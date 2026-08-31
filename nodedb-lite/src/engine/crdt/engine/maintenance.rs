// SPDX-License-Identifier: BUSL-1.1

//! CRDT history compaction and retained memory accounting.

use crate::error::LiteError;

use super::types::CrdtEngine;

impl CrdtEngine {
    /// Compact Loro history on every collection to prevent unbounded growth.
    ///
    /// Replaces each internal LoroDoc with a shallow snapshot. Historical
    /// operations are discarded. Current state is fully preserved.
    pub fn compact_history(&mut self) -> Result<(), LiteError> {
        // Only collections whose frontier has moved since their last
        // compaction. Compaction discards history behind the frontier, so a
        // collection that has taken no operations since has none to discard.
        //
        // Skipping them is the whole point. The bookkeeping below forces the
        // next flush to rewrite a compacted collection's base snapshot, and a
        // snapshot export is O(document). Doing that for every collection on
        // every periodic tick rewrote the entire store's snapshot set on a
        // fixed interval whether anything had changed or not — measured at
        // ~124 MB every five minutes on an idle dogfood store, none of which
        // is reclaimed until a restart.
        let due: Vec<String> = self
            .states
            .iter()
            .filter(|(collection, state)| {
                let frontier = state.oplog_version_vector();
                self.compacted_versions.get(*collection) != Some(&frontier)
            })
            .map(|(collection, _)| collection.clone())
            .collect();

        for collection in &due {
            let Some(state) = self.states.get_mut(collection) else {
                continue;
            };
            state.compact_history().map_err(|e| {
                LiteError::crdt(&format!("history compaction for '{collection}'"), e)
            })?;
            let frontier = state.oplog_version_vector();
            self.compacted_versions.insert(collection.clone(), frontier);
        }

        // Compaction rewrites the document without advancing its frontier, so
        // neither the persisted base nor the updates on top of it describe the
        // document any more, and the discarded history means an update export
        // from the old frontier may not even be possible. Dropping both marks
        // forces a fresh checkpoint, which also deletes the stale updates —
        // `next_delta_seq` is deliberately kept, since it is the count of the
        // entries that checkpoint has to delete.
        //
        // Advancing the epoch is what keeps a flush that is committing right
        // now from putting the marks back: its writes were exported from the
        // document this call just replaced.
        //
        // Both are per collection, and only for the ones actually compacted —
        // clearing the maps wholesale also discarded the marks of collections
        // this call never touched.
        for collection in &due {
            self.flushed_versions.remove(collection);
            self.checkpoint_bytes.remove(collection);
            self.delta_bytes.remove(collection);
            self.advance_state_epoch(collection);
        }
        Ok(())
    }

    /// Estimated memory usage includes retained live IDs, not a hard import ceiling.
    pub fn estimated_memory_bytes(&self) -> usize {
        let state_bytes: usize = self.states.values().fold(0usize, |bytes, state| {
            bytes.saturating_add(state.estimated_memory_bytes())
        });
        let delta_bytes: usize = self.pending_deltas.iter().fold(0usize, |bytes, delta| {
            bytes.saturating_add(delta.delta_bytes.len())
        });
        state_bytes
            .saturating_add(delta_bytes)
            .saturating_add(self.live_id_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::CrdtEngine;
    use loro::LoroValue;

    /// A second compaction with no writes in between must do nothing at all.
    ///
    /// Compaction drops a collection's checkpoint marks, which forces the next
    /// flush to rewrite its whole base snapshot — an O(document) export. A
    /// periodic tick that compacts unconditionally therefore rewrites the entire
    /// store's snapshot set on a fixed interval whether or not anything changed,
    /// which is what grew an idle store by ~124 MB every five minutes.
    #[test]
    fn compacting_twice_without_writes_leaves_the_second_pass_with_nothing_to_do() {
        let mut engine = CrdtEngine::new(1).unwrap();
        for i in 0..10 {
            engine
                .upsert("items", &format!("i{i}"), &[("val", LoroValue::I64(i))])
                .unwrap();
        }

        engine.compact_history().unwrap();
        let epoch_after_first = engine.state_epoch("items");

        // No writes in between.
        engine.compact_history().unwrap();
        assert_eq!(
            engine.state_epoch("items"),
            epoch_after_first,
            "an unchanged collection must not be compacted again, or its next \
             flush rewrites a base snapshot that did not change"
        );

        // A write makes it due again.
        engine
            .upsert("items", "i10", &[("val", LoroValue::I64(10))])
            .unwrap();
        engine.compact_history().unwrap();
        assert!(
            engine.state_epoch("items") > epoch_after_first,
            "a collection that took writes must still be compacted"
        );
    }

    #[test]
    fn compact_history_preserves_state() {
        let mut engine = CrdtEngine::new(1).unwrap();
        for i in 0..50 {
            engine
                .upsert(
                    "items",
                    &format!("i{i}"),
                    &[("val", LoroValue::I64(i as i64))],
                )
                .unwrap();
        }

        let mem_before = engine.estimated_memory_bytes();
        engine.compact_history().unwrap();

        // State should be preserved.
        assert!(engine.exists("items", "i0"));
        assert!(engine.exists("items", "i49"));

        // New operations should still work.
        engine
            .upsert("items", "i50", &[("val", LoroValue::I64(50))])
            .unwrap();
        assert!(engine.exists("items", "i50"));

        // Memory should be reduced (or at least not much larger).
        let mem_after = engine.estimated_memory_bytes();
        // History compaction should not increase memory significantly.
        assert!(
            mem_after <= mem_before * 2,
            "memory after compact ({mem_after}) should not be much larger than before ({mem_before})"
        );
    }

    #[test]
    fn memory_estimation() {
        let mut engine = CrdtEngine::new(1).unwrap();
        let before = engine.estimated_memory_bytes();

        for i in 0..100 {
            engine
                .upsert("big", &format!("k{i}"), &[("data", LoroValue::I64(i))])
                .unwrap();
        }

        let after = engine.estimated_memory_bytes();
        assert!(after > before);
    }
}
