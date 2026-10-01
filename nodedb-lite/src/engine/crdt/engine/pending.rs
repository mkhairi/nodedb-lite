// SPDX-License-Identifier: BUSL-1.1

//! Pending-delta queue management, acknowledgement, rejection, and the
//! local vector clock.

use std::collections::HashMap;

use super::types::{CrdtEngine, PendingDelta};

impl CrdtEngine {
    // ─── Sync: Delta Management ──────────────────────────────────────

    /// Get all pending (unsent) deltas.
    pub fn pending_deltas(&self) -> &[PendingDelta] {
        &self.pending_deltas
    }

    /// Number of unsent deltas.
    pub fn pending_count(&self) -> usize {
        self.pending_deltas.len()
    }

    /// Clear all pending deltas (used for partial flush recovery).
    /// The CRDT state is authoritative — pending deltas are regenerated on next mutation.
    pub fn clear_pending_deltas(&mut self) {
        self.pending_deltas.clear();
        self.unpersisted_deltas.clear();
    }

    /// Mark a queue entry as not matching its stored form, stamping it with a
    /// fresh revision.
    ///
    /// Every path that adds an entry or edits one in place goes through here,
    /// so an edit that lands while a flush is committing is distinguishable
    /// from the state that flush actually wrote.
    pub(in crate::engine::crdt) fn mark_delta_unpersisted(&mut self, mutation_id: u64) {
        self.delta_revision += 1;
        self.unpersisted_deltas
            .insert(mutation_id, self.delta_revision);
    }

    /// The pending deltas whose stored form may not match the queue, each with
    /// the revision to report back once it is durable.
    ///
    /// Entries already written under their own key are absent: the queue is
    /// append-only, so an unchanged entry does not need rewriting. Report the
    /// write back with [`Self::mark_pending_deltas_persisted`] once it has
    /// committed, passing the revision handed out here — not the entry's
    /// current one, which may have moved on since.
    pub fn pending_deltas_needing_write(&self) -> impl Iterator<Item = (&PendingDelta, u64)> {
        self.pending_deltas.iter().filter_map(|d| {
            self.unpersisted_deltas
                .get(&d.mutation_id)
                .map(|&revision| (d, revision))
        })
    }

    /// Number of queue entries written and acknowledged durable since this
    /// engine was created.
    pub fn pending_delta_write_count(&self) -> u64 {
        self.delta_writes
    }

    /// Whether any pending delta needs writing.
    pub fn has_unpersisted_deltas(&self) -> bool {
        !self.unpersisted_deltas.is_empty()
    }

    /// Retire the dirty marks for queue entries that are now durable.
    ///
    /// Each `(mutation_id, revision)` pair must be one handed out by
    /// [`Self::pending_deltas_needing_write`] for the batch that has just
    /// committed. An entry whose revision has moved on since was added or
    /// edited while that batch was in flight and so was never in it; its mark
    /// stays, and the next flush writes it.
    ///
    /// Call only after the batch has committed.
    pub fn mark_pending_deltas_persisted(&mut self, written: impl IntoIterator<Item = (u64, u64)>) {
        for (mutation_id, revision) in written {
            if self.unpersisted_deltas.get(&mutation_id) == Some(&revision) {
                self.unpersisted_deltas.remove(&mutation_id);
                self.delta_writes += 1;
            }
        }
    }

    /// Drop a single pending delta by `mutation_id` without touching CRDT state.
    ///
    /// Unlike [`reject_delta`](Self::reject_delta), this does **not** delete the
    /// document — the row stays in local CRDT state (so local reads/search work);
    /// it is simply never pushed to Origin. Used to keep a document local-only
    /// when the host's `SyncGate` rejects it for sync.
    pub fn drop_pending(&mut self, mutation_id: u64) {
        self.pending_deltas.retain(|d| d.mutation_id != mutation_id);
        self.unpersisted_deltas.remove(&mutation_id);
    }

    /// Assign a stable stream seq to a pending delta the first time it is sent.
    ///
    /// If the delta already has a non-zero seq (assigned on a previous send)
    /// the call is a no-op — the existing seq is reused on reconnect re-sends
    /// so Origin can deduplicate rather than double-apply.
    pub fn set_pending_delta_seq(&mut self, mutation_id: u64, seq: u64) {
        let assigned = match self
            .pending_deltas
            .iter_mut()
            .find(|d| d.mutation_id == mutation_id)
        {
            Some(d) if d.seq == 0 => {
                d.seq = seq;
                true
            }
            _ => false,
        };
        if assigned {
            // The stored entry now carries a stale seq.
            self.mark_delta_unpersisted(mutation_id);
        }
    }

    /// Retire the single delta Origin acknowledged (after DeltaAck received).
    ///
    /// Acks are per-mutation and are not ordered: an ack for a later mutation
    /// can arrive before one for an earlier mutation, and a non-applied status
    /// never produces an ack at all. Retiring the whole range at or below
    /// `acked_id` would therefore discard deltas Origin never acknowledged —
    /// one ack silently dropping the entire backlog behind it. Only the
    /// acknowledged mutation is removed; the rest stay queued until their own
    /// acks arrive.
    pub fn acknowledge(&mut self, acked_id: u64) {
        self.pending_deltas.retain(|d| d.mutation_id != acked_id);
        self.unpersisted_deltas.remove(&acked_id);
    }

    /// Roll back a specific pending delta (after DeltaReject with CompensationHint).
    ///
    /// This is a best-effort operation — Loro CRDTs don't support true undo.
    /// For document mutations, we delete the affected row and let the
    /// application re-create it with corrected values.
    ///
    /// Returns the rejected delta if found.
    pub fn reject_delta(&mut self, mutation_id: u64) -> Option<PendingDelta> {
        if let Some(pos) = self
            .pending_deltas
            .iter()
            .position(|d| d.mutation_id == mutation_id)
        {
            let delta = self.pending_deltas.remove(pos);
            self.unpersisted_deltas.remove(&mutation_id);
            // Best-effort rollback: delete the affected document from its own
            // collection's document. The application should handle the
            // CompensationHint and re-create with corrected values.
            if let Some(state) = self.states.get(&delta.collection) {
                let _ = state.delete(&delta.collection, &delta.document_id);
            }
            self.sync_indexes(&delta.collection, [delta.document_id.as_str()]);
            Some(delta)
        } else {
            None
        }
    }
    // ─── Vector Clock ────────────────────────────────────────────────

    /// Export the current vector clock as a serializable map.
    ///
    /// Format: `{ peer_id_hex: counter }` — matches the Loro version vector.
    ///
    /// Each collection owns its own document (and its own derived peer ID), so
    /// the returned clock is the merge of every collection's version vector.
    /// Peer IDs are per-collection-derived and therefore disjoint, but the
    /// merge takes the maximum counter so an id shared with a remote peer is
    /// never regressed.
    pub fn export_vector_clock(&self) -> HashMap<String, u64> {
        let mut clock: HashMap<String, u64> = HashMap::new();
        // Loro's VersionVector maps PeerID → Counter.
        // We encode PeerID as hex string for wire compatibility.
        for state in self.states.values() {
            for (peer, counter) in state.oplog_version_vector().iter() {
                let entry = clock.entry(format!("{peer:016x}")).or_insert(0);
                *entry = (*entry).max(*counter as u64);
            }
        }
        clock
    }

    /// Set the acked version for a collection (after sync handshake).
    pub fn set_acked_version(&mut self, collection: &str, version: u64) {
        self.acked_versions.insert(collection.to_string(), version);
    }

    /// Get the acked version for a collection.
    pub fn acked_version(&self, collection: &str) -> u64 {
        self.acked_versions.get(collection).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::CrdtEngine;
    use loro::LoroValue;

    #[test]
    fn acknowledge_removes_deltas() {
        let mut engine = CrdtEngine::new(1).unwrap();
        engine
            .upsert("a", "1", &[("x", LoroValue::I64(1))])
            .unwrap(); // mid=1
        engine
            .upsert("a", "2", &[("x", LoroValue::I64(2))])
            .unwrap(); // mid=2
        engine
            .upsert("a", "3", &[("x", LoroValue::I64(3))])
            .unwrap(); // mid=3

        assert_eq!(engine.pending_count(), 3);
        // Origin acknowledges mid=2. Acks are per-mutation, so only that delta is
        // retired — mid=1 has not been acknowledged and must stay queued.
        engine.acknowledge(2);
        let remaining: Vec<u64> = engine
            .pending_deltas()
            .iter()
            .map(|d| d.mutation_id)
            .collect();
        assert_eq!(remaining, vec![1, 3]);

        engine.acknowledge(1);
        engine.acknowledge(3);
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn reject_delta_rolls_back() {
        let mut engine = CrdtEngine::new(1).unwrap();
        let mid = engine
            .upsert(
                "users",
                "u1",
                &[("name", LoroValue::String("Alice".into()))],
            )
            .unwrap();

        assert!(engine.exists("users", "u1"));
        let rejected = engine.reject_delta(mid).unwrap();
        assert_eq!(rejected.collection, "users");
        assert!(!engine.exists("users", "u1"));
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn vector_clock_export() {
        let mut engine = CrdtEngine::new(1).unwrap();
        engine
            .upsert("x", "1", &[("v", LoroValue::I64(1))])
            .unwrap();

        let clock = engine.export_vector_clock();
        assert!(!clock.is_empty());
        // Each collection's document authors under its own derived peer ID.
        let our_key = format!(
            "{:016x}",
            CrdtEngine::collection_peer_id(engine.peer_id(), "x")
        );
        assert!(
            clock.contains_key(&our_key),
            "clock should contain peer {our_key}: {clock:?}"
        );
    }

    #[test]
    fn acked_version_tracking() {
        let mut engine = CrdtEngine::new(1).unwrap();
        assert_eq!(engine.acked_version("users"), 0);

        engine.set_acked_version("users", 42);
        assert_eq!(engine.acked_version("users"), 42);
    }

    /// Out-of-order acknowledgements retire only their matching mutation.
    #[test]
    fn acknowledge_retires_only_the_acknowledged_delta() {
        let mut engine = CrdtEngine::new(1).unwrap();

        let first = engine
            .upsert("notes", "a", &[("v", LoroValue::I64(1))])
            .unwrap();
        let second = engine
            .upsert("notes", "b", &[("v", LoroValue::I64(2))])
            .unwrap();
        let third = engine
            .upsert("notes", "c", &[("v", LoroValue::I64(3))])
            .unwrap();
        assert_eq!(engine.pending_deltas().len(), 3);

        // Origin acknowledges only the middle mutation.
        engine.acknowledge(second);

        let remaining: Vec<u64> = engine
            .pending_deltas()
            .iter()
            .map(|d| d.mutation_id)
            .collect();
        assert!(
            remaining.contains(&first),
            "acknowledging {second} also retired the un-acknowledged delta \
             {first}; its write is lost. remaining: {remaining:?}"
        );
        assert!(remaining.contains(&third), "remaining: {remaining:?}");
        assert!(!remaining.contains(&second));
    }

    /// A write queued while the batch was committing was not in it. Retiring its
    /// dirty mark strands it: an append-only queue is only revisited when it
    /// changes, so the entry would sit in memory until the process ended and the
    /// write it carries would never reach Origin.
    #[test]
    fn a_delta_queued_during_a_flush_is_not_retired_unwritten() {
        let mut engine = CrdtEngine::new(1).unwrap();
        engine
            .upsert("users", "u1", &[("n", LoroValue::I64(1))])
            .unwrap();

        let planned: Vec<(u64, u64)> = engine
            .pending_deltas_needing_write()
            .map(|(delta, revision)| (delta.mutation_id, revision))
            .collect();
        assert_eq!(planned.len(), 1);

        // The batch is committing. A second write lands before it is acknowledged.
        let queued_during = engine
            .upsert("users", "u2", &[("n", LoroValue::I64(2))])
            .unwrap();

        engine.mark_pending_deltas_persisted(planned);

        let still_dirty: Vec<u64> = engine
            .pending_deltas_needing_write()
            .map(|(delta, _)| delta.mutation_id)
            .collect();
        assert_eq!(
            still_dirty,
            vec![queued_during],
            "the entry queued while the batch was in flight was not in it, so it must still be \
             waiting to be written"
        );
        assert_eq!(
            engine.pending_delta_write_count(),
            1,
            "only the entry the batch actually carried counts as written"
        );
    }

    /// The same window catches an *edit*, not only an insertion: assigning a stream
    /// seq rewrites an entry that is already on disk, so its stored form goes stale
    /// and it has to be written again.
    #[test]
    fn a_delta_resequenced_during_a_flush_stays_dirty() {
        let mut engine = CrdtEngine::new(1).unwrap();
        let mid = engine
            .upsert("users", "u1", &[("n", LoroValue::I64(1))])
            .unwrap();

        let planned: Vec<(u64, u64)> = engine
            .pending_deltas_needing_write()
            .map(|(delta, revision)| (delta.mutation_id, revision))
            .collect();

        // The batch is committing. The delta is sent and is assigned its seq.
        engine.set_pending_delta_seq(mid, 7);

        engine.mark_pending_deltas_persisted(planned);

        assert!(
            engine.has_unpersisted_deltas(),
            "the stored entry carries seq 0 while the queue carries seq 7 — a resend after a \
             restart would use the wrong seq, so it must be rewritten"
        );
    }

    /// An acknowledgement that arrives with nothing to report leaves the queue
    /// alone: replaying one must not resurrect an entry or double-count a write.
    #[test]
    fn acknowledging_the_same_batch_twice_changes_nothing() {
        let mut engine = CrdtEngine::new(1).unwrap();
        engine
            .upsert("users", "u1", &[("n", LoroValue::I64(1))])
            .unwrap();

        let planned: Vec<(u64, u64)> = engine
            .pending_deltas_needing_write()
            .map(|(delta, revision)| (delta.mutation_id, revision))
            .collect();

        engine.mark_pending_deltas_persisted(planned.clone());
        engine.mark_pending_deltas_persisted(planned);

        assert!(!engine.has_unpersisted_deltas());
        assert_eq!(
            engine.pending_delta_write_count(),
            1,
            "the second report describes the same write, not another one"
        );
    }
}
