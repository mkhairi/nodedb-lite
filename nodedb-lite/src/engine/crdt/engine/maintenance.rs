// SPDX-License-Identifier: BUSL-1.1

//! CRDT history compaction and retained memory accounting.

use crate::error::LiteError;

use super::types::{CrdtEngine, HistoryCompaction};

impl CrdtEngine {
    /// Compact Loro history on every collection to prevent unbounded growth.
    ///
    /// Replaces each internal LoroDoc with a shallow snapshot. Historical
    /// operations are discarded. Current state is fully preserved.
    ///
    /// Compaction never writes to storage, and it costs the next flush nothing
    /// for a collection whose disk form is current. Each collection takes one
    /// of three paths, decided by its persisted frontier:
    ///
    /// - Persisted at exactly the current frontier: compacted, with its
    ///   persistence marks kept. Base plus updates on disk already replay to
    ///   this state and version vector. A shallow snapshot keeps every
    ///   operation after its root, so a later update exported from that
    ///   frontier still applies on top of the disk form. Disk history is
    ///   trimmed at the next checkpoint the updates earn on their own.
    /// - Persisted at an older frontier: deferred, left uncompacted and not
    ///   recorded as compacted. Its pending update is exported from the older
    ///   frontier, which compaction can discard. It becomes eligible once a
    ///   flush records the current frontier.
    /// - No persisted frontier: compacted, marks dropped. The next flush
    ///   writes a full checkpoint whether or not compaction runs.
    ///
    /// Dropping the marks of a persisted collection instead forced a full
    /// snapshot of every compacted collection into one commit. A store
    /// compacting every 30 minutes wrote 250-500 MB per run, more than the
    /// file reuses per commit, so it grew 400-600 MB an hour.
    ///
    /// Crash safety: this function changes only memory. Disk holds exactly
    /// what the last committed flush wrote, so a crash at any point restores
    /// that. A kept-marks collection was persisted at its current frontier, so
    /// restore reproduces the pre-compaction state and loses nothing. The next
    /// update or checkpoint reaches disk through the normal flush batch.
    ///
    /// Compacts every collection that took any operation since its last
    /// compaction. [`compact_history_min_ops`](Self::compact_history_min_ops)
    /// raises that bar.
    pub fn compact_history(&mut self) -> Result<HistoryCompaction, LiteError> {
        self.compact_history_min_ops(1)
    }

    /// [`compact_history`](Self::compact_history), compacting only the
    /// collections that took at least `min_ops` operations since their last
    /// compaction. The rest are skipped and counted in the report.
    ///
    /// Compaction rebuilds the document, at a cost set by the collection's
    /// size rather than by how much history it discards. On a large
    /// collection taking a steady trickle of writes, compacting on every tick
    /// holds the CRDT lock for seconds to discard a few operations. A
    /// threshold makes it wait until there is history worth discarding.
    /// A `min_ops` of 0 is treated as 1: a collection with no new operations
    /// has nothing to discard.
    pub fn compact_history_min_ops(
        &mut self,
        min_ops: u64,
    ) -> Result<HistoryCompaction, LiteError> {
        self.compact_history_inner(min_ops, false)
    }

    /// [`compact_history_min_ops`](Self::compact_history_min_ops), and the
    /// next flush rewrites each compacted collection as a checkpoint of its
    /// shallow state, deleting the updates it supersedes.
    ///
    /// The plain pass keeps a persisted collection's disk form, so disk keeps
    /// every operation its base and updates hold. Each open replays them all
    /// before a compaction discards them again: 19M operations per open on a
    /// 111k-document store. One rebasing pass trims disk to state, and the
    /// next open finds little to discard.
    ///
    /// The cost is one checkpoint write per compacted collection. Run it once
    /// per process, at open, never on a timer: forcing checkpoints on every
    /// periodic pass rewrote 250-500 MB per run.
    pub fn compact_history_and_rebase(
        &mut self,
        min_ops: u64,
    ) -> Result<HistoryCompaction, LiteError> {
        self.compact_history_inner(min_ops, true)
    }

    fn compact_history_inner(
        &mut self,
        min_ops: u64,
        rebase: bool,
    ) -> Result<HistoryCompaction, LiteError> {
        let mut report = HistoryCompaction::default();
        // The flag records whether the disk form is current at this frontier.
        let due: Vec<(String, bool, u64)> = self
            .states
            .iter()
            .filter_map(|(collection, state)| {
                let frontier = state.oplog_version_vector();
                let ops = ops_since(&frontier, self.compacted_versions.get(collection));
                if ops < min_ops.max(1) {
                    report.skipped += 1;
                    return None;
                }
                match self.flushed_versions.get(collection) {
                    Some(flushed) if *flushed == frontier => Some((collection.clone(), true, ops)),
                    Some(_) => {
                        report.deferred += 1;
                        None
                    }
                    None => Some((collection.clone(), false, ops)),
                }
            })
            .collect();

        for (collection, persisted_at_frontier, ops) in &due {
            let Some(state) = self.states.get_mut(collection) else {
                continue;
            };
            state.compact_history().map_err(|e| {
                LiteError::crdt(&format!("history compaction for '{collection}'"), e)
            })?;
            let frontier = state.oplog_version_vector();
            self.compacted_versions.insert(collection.clone(), frontier);

            // A collection with no persisted frontier owes a full checkpoint
            // already. Its marks are absent, and removing all three keeps them
            // in step. `next_delta_seq` stays: it is the count of updates that
            // checkpoint has to delete.
            if rebase || !persisted_at_frontier {
                self.flushed_versions.remove(collection);
                self.checkpoint_bytes.remove(collection);
                self.delta_bytes.remove(collection);
            }
            // A flush committing right now carries a write exported from the
            // document this call replaced. The epoch makes its acknowledgement
            // a no-op, so it cannot record marks for that form. For a
            // kept-marks collection this is harmless: a flush plans a write
            // only while the persisted frontier is behind, and such a
            // collection is deferred above.
            self.advance_state_epoch(collection);
            report.compacted += 1;
            report.ops_discarded += ops;
        }
        Ok(report)
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

/// Operations in `frontier` that `base` does not cover. With no base, every
/// operation the document has ever taken.
fn ops_since(frontier: &loro::VersionVector, base: Option<&loro::VersionVector>) -> u64 {
    frontier
        .iter()
        .map(|(peer, end)| {
            let start = base.and_then(|b| b.get(peer)).copied().unwrap_or(0);
            u64::try_from(end - start).unwrap_or(0)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::CrdtEngine;
    use loro::LoroValue;

    /// A second compaction with no writes in between must do nothing at all.
    ///
    /// Compacting again rebuilds the document for nothing. A periodic tick
    /// that compacts unconditionally repeats that work on every interval
    /// whether or not anything changed.
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
