// SPDX-License-Identifier: BUSL-1.1

//! Serialization of pending deltas and storage-key construction.

use std::sync::atomic::Ordering;

use super::types::{
    CrdtEngine, DELTA_KEY_PREFIX, PendingDelta, SNAPSHOT_KEY, STATE_DELTA_KEY, VCLOCK_KEY,
};

impl CrdtEngine {
    // ─── Persistence Helpers ─────────────────────────────────────────

    /// Serialize pending deltas to bytes for StorageEngine persistence.
    pub fn serialize_pending_deltas(&self) -> Result<Vec<u8>, crate::error::LiteError> {
        zerompk::to_msgpack_vec(&self.pending_deltas).map_err(|e| {
            crate::error::LiteError::Serialization {
                detail: format!("pending deltas: {e}"),
            }
        })
    }

    /// Restore pending deltas from bytes (cold start).
    pub fn restore_pending_deltas(&mut self, bytes: &[u8]) {
        match zerompk::from_msgpack::<Vec<PendingDelta>>(bytes) {
            Ok(deltas) => {
                // Advance mutation ID counter past any restored deltas.
                let max_id = deltas.iter().map(|d| d.mutation_id).max().unwrap_or(0);
                self.next_mutation_id.store(max_id + 1, Ordering::Relaxed);
                // The bulk blob is the only copy these came from, so none of
                // them is stored under its own key yet.
                self.unpersisted_deltas.clear();
                // The blob is one value: there are no per-entry keys to page
                // anything back in from, so the whole queue is resident until
                // the next flush writes the entries out individually and the
                // window is enforced over them.
                self.spill.clear();
                for delta in &deltas {
                    self.mark_delta_unpersisted(delta.mutation_id);
                }
                self.pending_deltas = deltas;
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to restore pending deltas, continuing with empty state");
            }
        }
    }

    /// Serialize a single pending delta to bytes (for append-only persistence).
    pub fn serialize_delta(delta: &PendingDelta) -> Result<Vec<u8>, crate::error::LiteError> {
        zerompk::to_msgpack_vec(delta).map_err(|e| crate::error::LiteError::Serialization {
            detail: format!("pending delta: {e}"),
        })
    }

    /// Decode a single pending delta stored by [`Self::serialize_delta`].
    pub fn deserialize_delta(bytes: &[u8]) -> Result<PendingDelta, crate::error::LiteError> {
        zerompk::from_msgpack::<PendingDelta>(bytes).map_err(|e| {
            crate::error::LiteError::Corrupted {
                detail: format!("queued CRDT mutation failed to decode: {e}"),
            }
        })
    }

    /// Build the KV key for a single pending delta: `delta:{mutation_id:016x}`.
    /// Zero-padded hex ensures lexicographic ordering matches numeric ordering.
    pub fn delta_storage_key(mutation_id: u64) -> Vec<u8> {
        format!("delta:{mutation_id:016x}").into_bytes()
    }

    /// Recover the mutation id from a key produced by
    /// [`Self::delta_storage_key`].
    ///
    /// Returns `None` for keys that do not carry the prefix or whose suffix is
    /// not a hex mutation id — such an entry cannot be matched against the
    /// queue, so the caller must decide its fate without guessing an id.
    pub fn mutation_id_from_delta_key(key: &[u8]) -> Option<u64> {
        let suffix = std::str::from_utf8(key.strip_prefix(DELTA_KEY_PREFIX)?).ok()?;
        u64::from_str_radix(suffix, 16).ok()
    }

    /// Restore pending deltas from individual KV entries (append-only format).
    ///
    /// Each entry is stored under `Namespace::Crdt` with key `delta:{mutation_id:016x}`.
    /// Falls back to legacy bulk restore if no individual entries found.
    ///
    /// A queued delta is a local mutation no Origin has acknowledged yet, so an
    /// entry that will not decode is unrecoverable data, not noise to step
    /// over. `allow_discard` reflects the caller's corruption policy: when it
    /// is false the undecodable entry is reported and every entry is left in
    /// storage untouched.
    pub fn restore_pending_deltas_incremental(
        &mut self,
        entries: &[(Vec<u8>, Vec<u8>)],
        allow_discard: bool,
    ) -> Result<(), crate::error::LiteError> {
        // Every one of these was just read from its own key, so the stored form
        // matches by construction.
        self.pending_deltas.clear();
        self.unpersisted_deltas.clear();
        self.spill.clear();
        self.absorb_restored_delta_chunk(entries, allow_discard)?;
        self.pending_deltas.sort_by_key(|d| d.mutation_id);
        Ok(())
    }

    /// Absorb one key-ordered chunk of stored `delta:` entries into the queue.
    ///
    /// Entries are decoded only while the resident window has room; past that,
    /// the mutation id is taken from the key and the entry is recorded as
    /// spilled — it stays queued, stays counted by
    /// [`CrdtEngine::pending_count`], and is paged back in when the window
    /// drains. This is what keeps opening a store with a million-entry outbox
    /// from materialising a million deltas.
    ///
    /// Chunks must arrive in ascending key order, so the oldest entries — the
    /// ones a connected Origin is sent first — are the ones that stay resident.
    ///
    /// A spilled entry is not decoded here, so an undecodable one is not found
    /// at open. It is found when the entry is paged in, and either way it is
    /// left in storage untouched.
    pub fn absorb_restored_delta_chunk(
        &mut self,
        entries: &[(Vec<u8>, Vec<u8>)],
        allow_discard: bool,
    ) -> Result<(), crate::error::LiteError> {
        let mut max_id = self.next_mutation_id.load(Ordering::Relaxed).saturating_sub(1);
        for (key, value) in entries {
            let key_id = Self::mutation_id_from_delta_key(key);
            if self.pending_deltas.len() < self.pending_window {
                match zerompk::from_msgpack::<PendingDelta>(value) {
                    Ok(delta) => {
                        max_id = max_id.max(delta.mutation_id);
                        self.pending_deltas.push(delta);
                    }
                    Err(e) => {
                        if !allow_discard {
                            return Err(crate::error::LiteError::Corrupted {
                                detail: format!(
                                    "queued CRDT mutation failed to decode: {e}. It carries a \
                                     local write that has not reached Origin, and has been left \
                                     in place."
                                ),
                            });
                        }
                        tracing::warn!(error = %e, "skipping corrupted pending delta entry");
                    }
                }
                continue;
            }

            let Some(mutation_id) = key_id else {
                // Without an id the entry can be neither ordered nor matched
                // against an acknowledgement, and it is past the window so its
                // payload was never read. Leaving it in storage is the only
                // answer that does not guess.
                tracing::warn!(
                    "stored CRDT delta key is not `delta:<mutation_id>` — leaving it in place"
                );
                continue;
            };
            max_id = max_id.max(mutation_id);
            self.spill.insert(mutation_id);
        }

        self.next_mutation_id
            .store(max_id.saturating_add(1), Ordering::Relaxed);
        Ok(())
    }

    /// Key for storing one collection's Loro snapshot in `StorageEngine`:
    /// `loro_snapshot:<collection>`.
    pub fn snapshot_key_for(collection: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(SNAPSHOT_KEY.len() + collection.len());
        key.extend_from_slice(SNAPSHOT_KEY);
        key.extend_from_slice(collection.as_bytes());
        key
    }

    /// Prefix shared by every per-collection snapshot key, for prefix scans.
    pub fn snapshot_key_prefix() -> &'static [u8] {
        SNAPSHOT_KEY
    }

    /// Key for one incremental update on top of a collection's snapshot:
    /// `loro_delta:<collection>:<seq:016x>`.
    ///
    /// Zero-padded hex so lexicographic key order is replay order, which is
    /// what a prefix scan returns them in.
    pub fn state_delta_key_for(collection: &str, seq: u64) -> Vec<u8> {
        let mut key = Vec::with_capacity(STATE_DELTA_KEY.len() + collection.len() + 17);
        key.extend_from_slice(STATE_DELTA_KEY);
        key.extend_from_slice(collection.as_bytes());
        key.extend_from_slice(format!(":{seq:016x}").as_bytes());
        key
    }

    /// Prefix shared by every state-update key, for prefix scans.
    pub fn state_delta_key_prefix() -> &'static [u8] {
        STATE_DELTA_KEY
    }

    /// Recover `(collection, seq)` from a key produced by
    /// [`Self::state_delta_key_for`].
    ///
    /// Returns `None` for keys that do not carry the prefix, whose collection
    /// is not UTF-8, or whose sequence does not parse — such an entry cannot be
    /// routed or ordered, so the caller must skip it rather than guess.
    pub fn state_delta_from_key(key: &[u8]) -> Option<(&str, u64)> {
        let suffix = std::str::from_utf8(key.strip_prefix(STATE_DELTA_KEY)?).ok()?;
        let (collection, seq) = suffix.rsplit_once(':')?;
        if collection.is_empty() {
            return None;
        }
        Some((collection, u64::from_str_radix(seq, 16).ok()?))
    }

    /// Recover the collection name from a snapshot key produced by
    /// [`Self::snapshot_key_for`].
    ///
    /// Returns `None` for keys that do not carry the prefix or whose suffix is
    /// not UTF-8 — such an entry cannot be routed to a document, so the caller
    /// must skip it rather than guess a collection.
    pub fn collection_from_snapshot_key(key: &[u8]) -> Option<&str> {
        let suffix = key.strip_prefix(SNAPSHOT_KEY)?;
        std::str::from_utf8(suffix).ok().filter(|s| !s.is_empty())
    }

    /// Key for storing pending deltas in `StorageEngine`.
    pub fn delta_key() -> &'static [u8] {
        DELTA_KEY_PREFIX
    }

    /// Key for storing the vector clock in `StorageEngine`.
    pub fn vclock_key() -> &'static [u8] {
        VCLOCK_KEY
    }
}

#[cfg(test)]
mod tests {
    use loro::LoroValue;

    use super::CrdtEngine;

    /// The `(key, value)` pairs storage holds for a queue of `count` entries.
    fn stored_entries(count: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut engine = CrdtEngine::new(1).expect("engine");
        for i in 0..count {
            engine
                .upsert(
                    "docs",
                    &format!("d{i}"),
                    &[("n", LoroValue::I64(i as i64))],
                )
                .expect("upsert");
        }
        engine
            .pending_deltas()
            .iter()
            .map(|d| {
                (
                    CrdtEngine::delta_storage_key(d.mutation_id),
                    CrdtEngine::serialize_delta(d).expect("serialize"),
                )
            })
            .collect()
    }

    #[test]
    fn restore_holds_no_more_than_the_window_in_memory() {
        let entries = stored_entries(500);
        let mut engine = CrdtEngine::new_with_pending_window(1, 10).expect("engine");

        engine
            .restore_pending_deltas_incremental(&entries, false)
            .expect("restore");

        assert_eq!(
            engine.resident_pending_count(),
            10,
            "opening a store must not materialise a queue that nothing drains"
        );
        assert_eq!(engine.spilled_pending_count(), 490);
        assert_eq!(engine.pending_count(), 500, "the queue is unchanged");
        assert_eq!(
            engine.max_pending_mutation_id(),
            500,
            "the partial-flush watermark must see the entries that were not read"
        );
        let resident: Vec<u64> = engine
            .pending_deltas()
            .iter()
            .map(|d| d.mutation_id)
            .collect();
        assert_eq!(
            resident,
            (1..=10).collect::<Vec<u64>>(),
            "the oldest entries stay resident — they are what Origin is sent first"
        );
        assert!(
            engine.retired_delta_ids(1..=500u64).is_empty(),
            "nothing restored may be retired: every entry is still queued"
        );
    }

    #[test]
    fn restore_absorbs_ordered_chunks_the_same_way() {
        let entries = stored_entries(50);
        let mut engine = CrdtEngine::new_with_pending_window(1, 10).expect("engine");

        for chunk in entries.chunks(7) {
            engine
                .absorb_restored_delta_chunk(chunk, false)
                .expect("absorb");
        }

        assert_eq!(engine.resident_pending_count(), 10);
        assert_eq!(engine.pending_count(), 50);
        assert_eq!(engine.max_pending_mutation_id(), 50);
    }

    #[test]
    fn a_queue_within_the_window_is_fully_resident() {
        let entries = stored_entries(5);
        let mut engine = CrdtEngine::new_with_pending_window(1, 10).expect("engine");

        engine
            .restore_pending_deltas_incremental(&entries, false)
            .expect("restore");

        assert_eq!(engine.resident_pending_count(), 5);
        assert_eq!(engine.spilled_pending_count(), 0);
    }

    #[test]
    fn a_corrupt_entry_inside_the_window_is_still_reported() {
        let mut entries = stored_entries(5);
        entries[2].1 = vec![0xff, 0xff, 0xff];
        let mut engine = CrdtEngine::new_with_pending_window(1, 10).expect("engine");

        assert!(
            engine
                .restore_pending_deltas_incremental(&entries, false)
                .is_err(),
            "an undecodable queued mutation must not be stepped over silently"
        );
    }

    #[test]
    fn delta_keys_round_trip_through_their_mutation_id() {
        for id in [0u64, 1, 42, u64::MAX] {
            let key = CrdtEngine::delta_storage_key(id);
            assert_eq!(CrdtEngine::mutation_id_from_delta_key(&key), Some(id));
        }
        assert_eq!(CrdtEngine::mutation_id_from_delta_key(b"loro_delta:x"), None);
        assert_eq!(CrdtEngine::mutation_id_from_delta_key(b"delta:zz"), None);
    }
}
