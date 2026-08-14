// SPDX-License-Identifier: BUSL-1.1

//! Pending-delta queue management, acknowledgement, rejection, and the
//! local vector clock.

use std::collections::HashMap;

use super::types::{CrdtEngine, PendingDelta};

impl CrdtEngine {
    // ─── Sync: Delta Management ──────────────────────────────────────

    /// The resident window of unsent deltas, oldest first.
    ///
    /// This is not necessarily the whole queue: entries beyond
    /// [`Self::pending_delta_window`] are held only under their `delta:` keys
    /// and are paged back in, oldest first, as the window drains. Use
    /// [`Self::pending_count`] for the true queue length and
    /// [`Self::spilled_pending_ids`] to page the rest back in.
    pub fn pending_deltas(&self) -> &[PendingDelta] {
        &self.pending_deltas
    }

    /// Number of unsent deltas — the whole queue, resident or not.
    pub fn pending_count(&self) -> usize {
        self.pending_deltas.len() + self.spill.len()
    }

    /// Number of unsent deltas currently held in memory.
    pub fn resident_pending_count(&self) -> usize {
        self.pending_deltas.len()
    }

    /// Number of queued deltas Origin has refused for a reason re-sending them
    /// cannot fix on its own.
    ///
    /// A subset of [`Self::pending_count`]: a blocked entry is still queued and
    /// still holds its row. Non-zero means replication is stalled on something
    /// outside this replica — most often a grant the Origin has not been given.
    pub fn blocked_delta_count(&self) -> usize {
        self.blocked_deltas.len()
    }

    /// Writes retired without applying — see [`CrdtEngine::dropped_writes`].
    pub fn dropped_write_count(&self) -> u64 {
        self.dropped_writes
    }

    /// Record that a write was retired without ever applying.
    ///
    /// Called from the ack paths that retire an entry on a terminal refusal,
    /// which is the only way a queued write leaves without landing.
    pub fn record_dropped_write(&mut self) {
        self.dropped_writes = self.dropped_writes.saturating_add(1);
    }

    /// Mark a queued delta as refused by Origin for a reason that is not about
    /// the row it carries.
    ///
    /// The entry stays queued and its row stays in local state: the refusal
    /// says nothing is wrong with the write, only that Origin will not take it
    /// as this session stands. Marking it is what keeps a stalled queue
    /// distinguishable from a draining one.
    pub fn mark_delta_blocked(&mut self, mutation_id: u64) {
        if self.pending_delta_is_live(mutation_id) {
            self.blocked_deltas.insert(mutation_id);
        }
    }

    /// The resident window with every blocked collection held back — what the
    /// push loop may send right now.
    ///
    /// A blocked entry is not skipped: Origin sequences each collection's
    /// stream, so sending the entries behind a refused one opens a gap it
    /// refuses in turn, and Loro deltas are causally chained, so those entries
    /// depend on the refused one in any case. The collection stalls at the
    /// refusal — which is what a refusal that is not about the row means — and
    /// the rest keep flowing. Without this the refused entry is re-pushed on
    /// every 100 ms tick for as long as the condition lasts.
    pub fn pushable_pending_deltas(&self) -> Vec<PendingDelta> {
        if self.blocked_deltas.is_empty() {
            return self.pending_deltas.clone();
        }
        let stalled: std::collections::HashSet<&str> = self
            .pending_deltas
            .iter()
            .filter(|d| self.blocked_deltas.contains(&d.mutation_id))
            .map(|d| d.collection.as_str())
            .collect();
        self.pending_deltas
            .iter()
            .filter(|d| !stalled.contains(d.collection.as_str()))
            .cloned()
            .collect()
    }

    /// Forget every blocked mark, so a fresh session re-attempts them.
    ///
    /// Called when a handshake completes: a grant added at the Origin between
    /// sessions is exactly what unblocks these, and nothing else tells the
    /// replica it happened.
    pub fn clear_blocked_deltas(&mut self) {
        self.blocked_deltas.clear();
    }

    /// Number of unsent deltas held only under their `delta:` keys.
    pub fn spilled_pending_count(&self) -> usize {
        self.spill.len()
    }

    /// Maximum number of unsent deltas held in memory at once.
    pub fn pending_delta_window(&self) -> usize {
        self.pending_window
    }

    /// The highest mutation id in the queue, resident or spilled.
    ///
    /// The partial-flush watermark is written from this: taking it from the
    /// resident window alone would make it regress every time the newest
    /// entries were evicted, and the next open would read that as a flush that
    /// tore.
    pub fn max_pending_mutation_id(&self) -> u64 {
        let resident = self
            .pending_deltas
            .iter()
            .map(|d| d.mutation_id)
            .max()
            .unwrap_or(0);
        resident.max(self.spill.max().unwrap_or(0))
    }

    /// Whether the entry stored under `delta:{mutation_id:016x}` is still
    /// queued — that is, whether its stored form must be kept.
    ///
    /// A spilled entry is live even though it is nowhere in memory, which is
    /// what stops the retirement sweep in `flush` from deleting the only copy
    /// of it that exists.
    pub fn pending_delta_is_live(&self, mutation_id: u64) -> bool {
        self.spill.contains(mutation_id)
            || self
                .pending_deltas
                .iter()
                .any(|d| d.mutation_id == mutation_id)
    }

    /// Of the mutation ids stored under `delta:` keys, the ones no longer
    /// queued — the entries whose stored form must be deleted.
    ///
    /// Batched rather than asked one id at a time: the stored set is the whole
    /// queue and the resident window is only part of it, so a per-id membership
    /// test would rescan the window once per stored entry.
    pub fn retired_delta_ids(&self, stored: impl IntoIterator<Item = u64>) -> Vec<u64> {
        let resident: std::collections::HashSet<u64> =
            self.pending_deltas.iter().map(|d| d.mutation_id).collect();
        stored
            .into_iter()
            .filter(|id| !resident.contains(id) && !self.spill.contains(*id))
            .collect()
    }

    /// Clear all pending deltas (used for partial flush recovery).
    /// The CRDT state is authoritative — pending deltas are regenerated on next mutation.
    pub fn clear_pending_deltas(&mut self) {
        self.pending_deltas.clear();
        self.unpersisted_deltas.clear();
        self.spill.clear();
        self.blocked_deltas.clear();
    }

    // ─── Sync: Resident Window ───────────────────────────────────────

    /// Evict durable queue entries from the tail of the resident window until
    /// it fits [`Self::pending_delta_window`], returning how many were evicted.
    ///
    /// Only entries whose stored form is known to match are evicted: an entry
    /// that has not been written yet exists nowhere else, so dropping it would
    /// lose a local mutation Origin has never seen. Call this after the flush
    /// that persisted them has committed and been acknowledged with
    /// [`Self::mark_pending_deltas_persisted`].
    ///
    /// The oldest entries are kept, so what stays resident is the head of the
    /// queue — the entries a connected Origin is about to be sent.
    pub fn evict_pending_overflow(&mut self) -> usize {
        if self.pending_deltas.len() <= self.pending_window {
            return 0;
        }

        // From the newest end, so the head — the next entries to push — stays.
        // A still-unwritten entry is skipped rather than stopping the sweep:
        // under a continuous write stream the newest entry is almost always
        // unwritten, and stopping there would mean never evicting at all.
        let mut evicting: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for delta in self.pending_deltas.iter().rev() {
            if self.pending_deltas.len() - evicting.len() <= self.pending_window {
                break;
            }
            if self.unpersisted_deltas.contains_key(&delta.mutation_id) {
                continue;
            }
            evicting.insert(delta.mutation_id);
        }

        if evicting.is_empty() {
            return 0;
        }
        for &mutation_id in &evicting {
            self.spill.insert(mutation_id);
        }
        self.pending_deltas
            .retain(|d| !evicting.contains(&d.mutation_id));
        evicting.len()
    }

    /// The mutation ids of the `limit` oldest spilled entries — the next ones
    /// to page back in.
    ///
    /// Returns nothing while the resident window is full: paging in beyond it
    /// is what the window exists to prevent.
    pub fn spilled_pending_ids(&self, limit: usize) -> Vec<u64> {
        if self.spill.is_empty() {
            return Vec::new();
        }
        let room = self
            .pending_window
            .saturating_sub(self.pending_deltas.len());
        self.spill.lowest(room.min(limit))
    }

    /// Page previously evicted entries back into the resident window.
    ///
    /// `deltas` must be entries decoded from their own `delta:` keys. Anything
    /// not currently spilled is ignored — it is either already resident or no
    /// longer queued, and re-adding it would resurrect an acknowledged
    /// mutation. Returns how many entries were made resident.
    ///
    /// The entries are not marked unpersisted: they came from storage, so their
    /// stored form matches by construction.
    pub fn hydrate_pending_deltas(
        &mut self,
        deltas: impl IntoIterator<Item = PendingDelta>,
    ) -> usize {
        let mut hydrated = 0;
        for delta in deltas {
            if !self.spill.remove(delta.mutation_id) {
                continue;
            }
            self.pending_deltas.push(delta);
            hydrated += 1;
        }
        if hydrated > 0 {
            self.pending_deltas.sort_by_key(|d| d.mutation_id);
        }
        hydrated
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
        self.spill.remove(mutation_id);
        self.blocked_deltas.remove(&mutation_id);
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
    ///
    /// An entry evicted from the resident window is acknowledged the same way:
    /// it leaves the queue, and the next flush deletes the `delta:` key that
    /// held it. Nothing has to be paged in first — an ack names the mutation,
    /// and retiring it needs nothing else.
    pub fn acknowledge(&mut self, acked_id: u64) {
        self.pending_deltas.retain(|d| d.mutation_id != acked_id);
        self.unpersisted_deltas.remove(&acked_id);
        self.spill.remove(acked_id);
        self.blocked_deltas.remove(&acked_id);
    }

    /// Roll back a specific pending delta (after DeltaReject with CompensationHint).
    ///
    /// This is a best-effort operation — Loro CRDTs don't support true undo.
    /// For document mutations, we delete the affected row and let the
    /// application re-create it with corrected values.
    ///
    /// Returns the rejected delta if found.
    ///
    /// Compensation needs the delta's collection and row, so this acts on the
    /// resident window only. An entry evicted while its push was in flight is
    /// reported as not found and stays queued; it is pushed again once it is
    /// paged back in, and is then resident for the rejection that follows.
    pub fn reject_delta(&mut self, mutation_id: u64) -> Option<PendingDelta> {
        if let Some(pos) = self
            .pending_deltas
            .iter()
            .position(|d| d.mutation_id == mutation_id)
        {
            let delta = self.pending_deltas.remove(pos);
            self.unpersisted_deltas.remove(&mutation_id);
            self.blocked_deltas.remove(&mutation_id);
            // The row is deleted below and the delta leaves the queue: this
            // write applied nowhere, and nothing else records that.
            self.dropped_writes = self.dropped_writes.saturating_add(1);
            // Best-effort rollback: delete the affected document from its own
            // collection's document. The application should handle the
            // CompensationHint and re-create with corrected values.
            if let Some(state) = self.states.get(&delta.collection) {
                let _ = state.delete(&delta.collection, &delta.document_id);
            }
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
    use loro::LoroValue;

    use super::super::types::CrdtEngine;

    /// Queue `count` mutations, each producing exactly one delta.
    fn engine_with_queue(window: usize, count: usize) -> CrdtEngine {
        let mut engine = CrdtEngine::new_with_pending_window(1, window).expect("engine");
        for i in 0..count {
            engine
                .upsert("docs", &format!("d{i}"), &[("n", LoroValue::I64(i as i64))])
                .expect("upsert");
        }
        engine
    }

    /// Stand in for the flush that writes the queue out: every entry's stored
    /// form now matches, which is what makes it evictable.
    fn persist_queue(engine: &mut CrdtEngine) {
        let written: Vec<(u64, u64)> = engine
            .pending_deltas_needing_write()
            .map(|(delta, revision)| (delta.mutation_id, revision))
            .collect();
        engine.mark_pending_deltas_persisted(written);
    }

    #[test]
    fn eviction_preserves_the_total_count() {
        let mut engine = engine_with_queue(8, 50);
        persist_queue(&mut engine);

        let evicted = engine.evict_pending_overflow();

        assert_eq!(evicted, 42);
        assert_eq!(
            engine.resident_pending_count(),
            8,
            "window must be honoured"
        );
        assert_eq!(engine.spilled_pending_count(), 42);
        assert_eq!(
            engine.pending_count(),
            50,
            "an entry paged out to its own key is still queued; health reporting \
             and the partial-flush watermark both read this"
        );
        assert_eq!(
            engine.max_pending_mutation_id(),
            50,
            "the watermark must not regress when the newest entries are paged out"
        );
    }

    #[test]
    fn eviction_keeps_the_oldest_entries_resident() {
        let mut engine = engine_with_queue(8, 50);
        persist_queue(&mut engine);
        engine.evict_pending_overflow();

        let resident: Vec<u64> = engine
            .pending_deltas()
            .iter()
            .map(|d| d.mutation_id)
            .collect();
        assert_eq!(
            resident,
            (1..=8).collect::<Vec<u64>>(),
            "the head of the queue is what a connected Origin is sent first"
        );
    }

    #[test]
    fn an_unwritten_entry_is_never_evicted() {
        let mut engine = engine_with_queue(1, 4);

        // Nothing has been flushed, so no entry has a stored form to fall back
        // on and dropping one would lose the mutation outright.
        assert_eq!(engine.evict_pending_overflow(), 0);
        assert_eq!(engine.resident_pending_count(), 4);
        assert_eq!(engine.spilled_pending_count(), 0);

        // A later write that has not been flushed stays resident while the
        // durable entries around it are paged out.
        persist_queue(&mut engine);
        engine
            .upsert("docs", "late", &[("n", LoroValue::I64(9))])
            .expect("upsert");
        engine.evict_pending_overflow();

        assert!(
            engine
                .pending_deltas()
                .iter()
                .any(|d| d.document_id == "late"),
            "an entry with no stored form must stay in memory"
        );
        assert_eq!(engine.pending_count(), 5);
    }

    #[test]
    fn an_evicted_entry_is_still_acknowledgeable() {
        let mut engine = engine_with_queue(8, 50);
        persist_queue(&mut engine);
        engine.evict_pending_overflow();
        assert!(
            !engine.pending_deltas().iter().any(|d| d.mutation_id == 30),
            "precondition: mutation 30 is not resident"
        );
        assert!(engine.pending_delta_is_live(30));

        engine.acknowledge(30);

        assert_eq!(engine.pending_count(), 49);
        assert_eq!(engine.spilled_pending_count(), 41);
        assert!(
            !engine.pending_delta_is_live(30),
            "an acknowledged entry must stop being live so its stored key is deleted"
        );
        assert_eq!(
            engine.retired_delta_ids(1..=50u64),
            vec![30],
            "only the acknowledged entry is retired; the rest are on disk and still queued"
        );
    }

    #[test]
    fn an_evicted_entry_is_readable_after_paging_it_back_in() {
        let mut engine = engine_with_queue(8, 50);
        persist_queue(&mut engine);

        // What storage holds for mutation 30, captured before it is paged out.
        let stored = engine
            .pending_deltas()
            .iter()
            .find(|d| d.mutation_id == 30)
            .map(CrdtEngine::serialize_delta)
            .expect("mutation 30 queued")
            .expect("serialize");

        engine.evict_pending_overflow();
        assert!(!engine.pending_deltas().iter().any(|d| d.mutation_id == 30));

        // Drain the head so the window has room, as an Origin's acks would.
        for id in 1..=8 {
            engine.acknowledge(id);
        }
        let next = engine.spilled_pending_ids(usize::MAX);
        assert_eq!(next.first(), Some(&9), "the oldest entry is paged in first");
        assert_eq!(next.len(), 8, "no more than the window is paged in");

        let delta = CrdtEngine::deserialize_delta(&stored).expect("decode");
        assert_eq!(engine.hydrate_pending_deltas([delta]), 1);

        let back = engine
            .pending_deltas()
            .iter()
            .find(|d| d.mutation_id == 30)
            .expect("mutation 30 is resident again");
        assert_eq!(back.document_id, "d29");
        assert!(
            !back.delta_bytes.is_empty(),
            "the payload survived the round trip"
        );
        assert_eq!(engine.pending_count(), 42, "paging in changes no total");
        assert_eq!(engine.spilled_pending_count(), 41);
    }

    #[test]
    fn paging_in_an_entry_that_is_no_longer_queued_is_refused() {
        let mut engine = engine_with_queue(8, 50);
        persist_queue(&mut engine);
        let stored = engine
            .pending_deltas()
            .iter()
            .find(|d| d.mutation_id == 30)
            .map(CrdtEngine::serialize_delta)
            .expect("queued")
            .expect("serialize");
        engine.evict_pending_overflow();
        engine.acknowledge(30);

        let delta = CrdtEngine::deserialize_delta(&stored).expect("decode");
        assert_eq!(
            engine.hydrate_pending_deltas([delta]),
            0,
            "a stale read must not resurrect an acknowledged mutation"
        );
        assert_eq!(engine.pending_count(), 49);
    }

    #[test]
    fn clearing_the_queue_forgets_the_evicted_entries_too() {
        let mut engine = engine_with_queue(8, 50);
        persist_queue(&mut engine);
        engine.evict_pending_overflow();

        engine.clear_pending_deltas();

        assert_eq!(engine.pending_count(), 0);
        assert_eq!(
            engine.retired_delta_ids(1..=50u64).len(),
            50,
            "every stored entry is retired once the queue is cleared"
        );
    }

    #[test]
    fn a_window_of_zero_is_refused() {
        assert!(
            CrdtEngine::new_with_pending_window(1, 0).is_err(),
            "a queue with nowhere to hold a fresh entry cannot persist one"
        );
    }

    /// With sync off, a mutation must still be applied but must not be staged.
    ///
    /// The queue exists to be drained by an Origin. Without one it is a leak
    /// paid for in RAM, in `delta:` keys, and in flush work — for every write,
    /// forever. What must NOT change is the local document: the mutation is
    /// applied, readable, and still mints a mutation id for the caller.
    #[test]
    fn sync_disabled_applies_the_mutation_without_staging_a_delta() {
        let mut engine = CrdtEngine::new_with_options(1, 10, false).expect("engine");
        let mutation_id = engine
            .upsert("docs", "d1", &[("n", LoroValue::I64(7))])
            .expect("upsert");

        assert_eq!(engine.pending_count(), 0, "no delta may be staged");
        assert!(!engine.has_unpersisted_deltas(), "nothing to write out");
        assert!(mutation_id > 0, "the caller still gets a mutation id");

        // The write itself landed: this is the half that must not regress.
        assert!(
            engine.exists("docs", "d1"),
            "the row must exist locally even though no delta was staged"
        );
        assert_eq!(
            engine.read_field("docs", "d1", "n"),
            Some(LoroValue::I64(7)),
            "the value must be readable"
        );
    }

    /// The same engine with sync on still stages — the flag is what decides,
    /// not some unrelated change in the write path.
    #[test]
    fn sync_enabled_still_stages_a_delta() {
        let mut engine = CrdtEngine::new_with_options(1, 10, true).expect("engine");
        engine
            .upsert("docs", "d1", &[("n", LoroValue::I64(7))])
            .expect("upsert");
        assert_eq!(engine.pending_count(), 1, "sync on must stage");
        assert!(engine.has_unpersisted_deltas());
    }
}
