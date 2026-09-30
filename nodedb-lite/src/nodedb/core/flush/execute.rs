// SPDX-License-Identifier: Apache-2.0

//! `NodeDbLite::flush` — persist all in-memory state to storage.

use crate::storage::engine::{StorageEngine, WriteOp};
use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::crdt::{CrdtEngine, CrdtWriteKind};
use crate::engine::fts::coordinator::TextMutationPermit;
use crate::nodedb::flush_gens::FlushArtifact;
use crate::nodedb::lock_ext::LockExt;

use super::super::types::{META_CRDT_DELTAS, META_LAST_FLUSHED_MID, NodeDbLite};

impl<S: StorageEngine> NodeDbLite<S> {
    /// Number of full CRDT snapshot exports performed since this handle was
    /// opened.
    ///
    /// A snapshot export costs O(document), so this is the term that decides
    /// both flush latency and how much the file grows per tick. It is a
    /// counter, not a timing, so a caller can assert on it directly: an idle
    /// store must not advance it.
    pub fn crdt_snapshot_export_count(&self) -> u64 {
        self.crdt.lock_or_recover().snapshot_export_count()
    }

    /// Number of unsent-delta queue entries written since this handle was
    /// opened.
    ///
    /// The queue is append-only, so this advances by what was added, not by
    /// the queue's length. An idle store must not advance it at all.
    pub fn crdt_delta_write_count(&self) -> u64 {
        self.crdt.lock_or_recover().pending_delta_write_count()
    }

    /// Number of unsent CRDT deltas held only under their `delta:` keys.
    ///
    /// The queue itself is reported by `pending_count`; this is the part of it
    /// that costs a mutation id rather than a payload.
    pub fn crdt_spilled_delta_count(&self) -> usize {
        self.crdt.lock_or_recover().spilled_pending_count()
    }

    /// Number of successful writes of `artifact` for `collection` since this
    /// handle was opened.
    ///
    /// [`FlushArtifact::HnswIdMap`] is one store-wide blob: pass
    /// [`ID_MAP_KEY`](crate::nodedb::ID_MAP_KEY) (the empty string) as its
    /// collection. A counter, not a timing, so a caller can assert on it
    /// directly: an idle store must not advance it.
    pub fn flush_artifact_write_count(&self, artifact: FlushArtifact, collection: &str) -> u64 {
        self.flush_gens.write_count(artifact, collection)
    }

    /// Whether `artifact` for `collection` holds mutations that no flush has
    /// made durable yet.
    pub fn flush_artifact_is_dirty(&self, artifact: FlushArtifact, collection: &str) -> bool {
        self.flush_gens.is_dirty(artifact, collection)
    }

    /// Mutation ids of every queue entry currently stored under a `delta:` key.
    ///
    /// Read in bounded chunks so a queue that nothing acknowledges does not
    /// have to fit in memory to be swept. A key that does not carry a mutation
    /// id is skipped rather than deleted: it cannot be matched against the
    /// queue, and deleting a stored entry on the strength of a name we cannot
    /// read is how an unacknowledged local write disappears.
    async fn persisted_delta_ids(&self) -> NodeDbResult<Vec<u64>> {
        const CHUNK: usize = 4_096;
        let mut ids = Vec::new();
        let mut start = b"delta:".to_vec();
        loop {
            let chunk = self
                .storage
                .scan_range(Namespace::Crdt, &start, CHUNK)
                .await?;
            if chunk.is_empty() {
                break;
            }
            let scanned = chunk.len();
            let mut next_start = chunk[scanned - 1].0.clone();
            next_start.push(0);

            let mut ended = scanned < CHUNK;
            for (key, _) in chunk {
                if !key.starts_with(b"delta:") {
                    ended = true;
                    break;
                }
                match CrdtEngine::mutation_id_from_delta_key(&key) {
                    Some(id) => ids.push(id),
                    None => tracing::warn!(
                        "stored CRDT delta key is not `delta:<mutation_id>` — leaving it in place"
                    ),
                }
            }
            if ended {
                break;
            }
            start = next_start;
        }
        Ok(ids)
    }

    /// Bring the resident delta window back up to size from the `delta:` keys.
    ///
    /// Called after each flush: entries are paged out only once they are
    /// durable, so the window refills at flush granularity — which is also the
    /// granularity at which an Origin acknowledgement can empty it.
    async fn hydrate_crdt_delta_window(&self) -> NodeDbResult<usize> {
        let wanted = {
            let crdt = self.crdt.lock_or_recover();
            crdt.spilled_pending_ids(crdt.pending_delta_window())
        };
        let Some(&lowest) = wanted.first() else {
            return Ok(0);
        };

        // One ordered scan from the oldest missing entry rather than a read per
        // id. Anything returned that is not actually spilled is discarded by
        // `hydrate_pending_deltas`.
        let entries = self
            .storage
            .scan_range(
                Namespace::Crdt,
                &CrdtEngine::delta_storage_key(lowest),
                wanted.len(),
            )
            .await?;

        let mut deltas = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            if !key.starts_with(b"delta:") {
                break;
            }
            match CrdtEngine::deserialize_delta(&value) {
                Ok(delta) => deltas.push(delta),
                Err(e) => tracing::warn!(
                    error = %e,
                    "queued CRDT delta failed to decode while paging it back in — \
                     leaving it in storage"
                ),
            }
        }

        let mut crdt = self.crdt.lock_or_recover();
        Ok(crdt.hydrate_pending_deltas(deltas))
    }

    /// Persist all in-memory state to storage (call before shutdown).
    ///
    /// Dirty-aware: an HNSW graph, the vector id-map, a vector segment, a CSR
    /// graph checkpoint, a sparse index, a spatial R-tree or doc-map, or a
    /// meta or catalog entry that has not changed since this handle last made
    /// it durable is not written again.
    /// [`flush_full`](Self::flush_full) writes all of them regardless.
    pub async fn flush(&self) -> NodeDbResult<()> {
        let permit = self.fts_state.admit_exclusive().await;
        self.flush_admitted(&permit).await
    }

    /// Persist all in-memory state, writing every dirty-tracked artifact
    /// whether or not it changed, then mark each one flushed.
    ///
    /// For callers that do not trust the stored form, e.g. after an external
    /// tool touched the file. Artifacts outside the dirty tracking behave as in
    /// [`flush`](Self::flush).
    pub async fn flush_full(&self) -> NodeDbResult<()> {
        let permit = self.fts_state.admit_exclusive().await;
        self.flush_pass(&permit, true).await
    }

    pub(crate) async fn flush_admitted(&self, permit: &TextMutationPermit) -> NodeDbResult<()> {
        self.flush_pass(permit, false).await
    }

    /// One flush pass. `full` ignores the dirty state of tracked artifacts.
    async fn flush_pass(&self, _permit: &TextMutationPermit, full: bool) -> NodeDbResult<()> {
        // One flush at a time: the CRDT update sequence is allocated under the
        // `crdt` guard but committed after it is released, so concurrent
        // flushes would hand out the same numbers. See `flush_lock`.
        let _flush_guard = self.flush_lock.lock().await;
        // The checkpoint-state marker reads incomplete while the FTS state
        // on disk may lag memory. It is skipped when the FTS index has
        // nothing to write and the stored marker already reads what this
        // flush would end with: FTS mutations wait for the exclusive admission
        // this flush holds, so the stored index stays current throughout.
        let fts_markers = {
            use crate::engine::fts::checkpoint::{
                CHECKPOINT_STATE_KEY, encode_checkpoint_complete, encode_checkpoint_incomplete,
            };
            let fts = self.fts_state.manager.lock_or_recover();
            let end_state = if self.fts_state.checkpoint_trusted() {
                encode_checkpoint_complete(&fts.declaration_revisions())
            } else {
                encode_checkpoint_incomplete()
            }
            .map_err(NodeDbError::from)?;
            full || fts.checkpoint_pending(full)
                || self
                    .flush_gens
                    .meta_changed(CHECKPOINT_STATE_KEY, &end_state)
        };
        if fts_markers {
            crate::engine::fts::checkpoint::persist_checkpoint_incomplete(self.storage.as_ref())
                .await
                .map_err(NodeDbError::from)?;
            self.flush_gens.mark_meta_written(
                crate::engine::fts::checkpoint::CHECKPOINT_STATE_KEY,
                crate::engine::fts::checkpoint::encode_checkpoint_incomplete()
                    .map_err(NodeDbError::from)?,
            );
        }

        // Drain the buffered KV writes first — they have their own batch-commit
        // path. Without this, `flush()` (and the auto-flush timer) would not
        // persist KV `put`s, contradicting "persist all in-memory state".
        self.kv_flush_inner().await?;

        let mut ops = Vec::new();
        // Meta entries queued in `ops`, recorded as written once the batch
        // commits. An unchanged meta value is not written again.
        let mut meta_writes: Vec<(&'static [u8], Vec<u8>)> = Vec::new();

        // Delta entries already on disk. Restore prefers these over the bulk
        // blob, so any one that is no longer pending — acknowledged by Origin,
        // or replaced by a peer-id rotation that re-authored the row — would
        // come back on the next open and be pushed again. They are deleted in
        // the same batch that writes the current set.
        //
        // Read in chunks, keeping the mutation ids rather than the entries: an
        // outbox with no Origin to drain it holds every mutation ever made, and
        // reading it whole to look at its keys materialises every payload in it
        // once per flush tick.
        //
        // With sync off nothing stages deltas, so the queue is empty and every
        // stored `delta:` key would look retired — the sweep below would delete
        // the whole backlog left over from a period when sync was on. Skipping
        // the scan keeps that residue on disk, so re-enabling sync is a
        // decision an operator makes, not one a flush tick makes for them. It
        // also removes the scan from the tick entirely in the common case.
        let sync_enabled = self.crdt.lock_or_recover().sync_enabled();
        let persisted_delta_ids = if sync_enabled {
            self.persisted_delta_ids().await?
        } else {
            Vec::new()
        };

        // ── Persist one CRDT snapshot per collection (CRC32C wrapped) ──
        // Each collection owns its own Loro document, so each gets its own
        // storage entry under `loro_snapshot:<collection>`.
        //
        // A collection whose frontier has not moved is not written at all; one
        // that has moved is written as an update since its last persisted
        // frontier, and only periodically as a fresh snapshot. Exporting a full
        // snapshot per collection per tick cost O(document) regardless of the
        // write rate — unbounded file growth on an otherwise idle store, and an
        // export duty cycle that starved readers once the document outgrew the
        // flush interval.
        let (persisted, written_deltas, index_flush) = {
            let crdt = self.crdt.lock_or_recover();
            let plan = crdt.plan_persistence().map_err(NodeDbError::storage)?;
            let mut persisted = Vec::with_capacity(plan.len());
            for write in plan {
                persisted.push(write.persisted());
                match write.kind {
                    CrdtWriteKind::Checkpoint { superseded_deltas } => {
                        // In the same batch as the new base, so no restore ever
                        // sees a base with updates in front of it that it
                        // already contains.
                        for seq in 0..superseded_deltas {
                            ops.push(WriteOp::Delete {
                                ns: Namespace::LoroState,
                                key: CrdtEngine::state_delta_key_for(&write.collection, seq),
                            });
                        }
                        ops.push(WriteOp::Put {
                            ns: Namespace::LoroState,
                            key: CrdtEngine::snapshot_key_for(&write.collection),
                            value: crate::storage::checksum::wrap(&write.bytes),
                        });
                    }
                    CrdtWriteKind::Delta { seq } => {
                        ops.push(WriteOp::Put {
                            ns: Namespace::LoroState,
                            key: CrdtEngine::state_delta_key_for(&write.collection, seq),
                            value: crate::storage::checksum::wrap(&write.bytes),
                        });
                    }
                }
            }

            // Write pending deltas individually (append-only persistence).
            // Each delta is stored under `crdt:delta:{mutation_id:016x}`.
            //
            // Only the entries added or edited since the last flush are
            // written. The queue is append-only and each entry owns its key,
            // so rewriting an unchanged one stores bytes identical to the ones
            // already there — and a replica with no Origin to acknowledge its
            // deltas accumulates them without bound, which made that rewrite
            // the whole outbox, once per `auto_flush_ms`.
            // The watermark covers the whole queue, not the resident window:
            // taking it from memory alone would make it regress as soon as the
            // newest entries were paged out, and the next open reads a
            // regressed watermark as a flush that tore.
            let max_mid = crdt.max_pending_mutation_id();

            // A paged-out entry is still queued, and its stored key is the only
            // copy of it that exists — `pending_delta_is_live` answers for the
            // whole queue so the sweep below cannot delete it.
            let retired = if sync_enabled {
                crdt.retired_delta_ids(persisted_delta_ids)
            } else {
                Vec::new()
            };
            let retired_any = !retired.is_empty();
            for mutation_id in retired {
                ops.push(WriteOp::Delete {
                    ns: Namespace::Crdt,
                    key: CrdtEngine::delta_storage_key(mutation_id),
                });
            }

            // The revision each entry was written at travels with it: the
            // acknowledgement below happens after an await, and an entry queued
            // or re-sequenced in that window was never in this batch.
            let mut written_deltas: Vec<(u64, u64)> = Vec::new();
            for (delta, revision) in crdt.pending_deltas_needing_write() {
                let key = CrdtEngine::delta_storage_key(delta.mutation_id);
                let value = CrdtEngine::serialize_delta(delta).map_err(NodeDbError::storage)?;
                written_deltas.push((delta.mutation_id, revision));
                ops.push(WriteOp::Put {
                    ns: Namespace::Crdt,
                    key,
                    value,
                });
            }

            // The legacy bulk blob duplicated every entry above in a single
            // value, and restore prefers the per-entry keys whenever they
            // exist — which is always, since this loop writes them. Keeping it
            // current therefore cost a full rewrite of the whole queue on any
            // flush that changed it, and the pages superseded by each rewrite
            // are not immediately reusable. Where deltas are never acknowledged
            // the queue only grows, so that rewrite is O(queue) per tick with no
            // upper bound, and the file grows without the data doing so.
            //
            // It is deleted rather than left stale: a blob that no longer
            // matches the queue is worse than no blob. Restore falls back to it
            // only when the per-entry scan comes back empty, which now means
            // those entries are damaged or missing — and resurrecting a stale
            // queue is the wrong answer to that.
            if retired_any || !written_deltas.is_empty() {
                ops.push(WriteOp::Delete {
                    ns: Namespace::Crdt,
                    key: META_CRDT_DELTAS.to_vec(),
                });
            }

            // Index entries in the same batch and lock hold as the rows.
            let index_flush = self.query_engine.indexes.stage_flush(&mut ops);

            // Write the last-flushed mutation_id for partial flush safety.
            // Skipped when it equals the value this handle last committed:
            // rewriting it on an idle tick stores identical bytes.
            let max_mid_bytes = max_mid.to_le_bytes().to_vec();
            if full
                || self
                    .flush_gens
                    .meta_changed(META_LAST_FLUSHED_MID, &max_mid_bytes)
            {
                ops.push(WriteOp::Put {
                    ns: Namespace::Meta,
                    key: META_LAST_FLUSHED_MID.to_vec(),
                    value: max_mid_bytes.clone(),
                });
                meta_writes.push((META_LAST_FLUSHED_MID, max_mid_bytes));
            }

            (persisted, written_deltas, index_flush)
        };

        let mut jobs = self.stage_indexes(&mut ops, full, &mut meta_writes)?;
        self.storage
            .batch_write(&ops)
            .await
            .map_err(NodeDbError::storage)?;

        // The graph blobs, the id-map, and the meta entries in the batch are
        // durable now. Record the generations captured when they were
        // serialized, never the current ones: a mutation made since keeps its
        // artifact dirty. A failed batch returned above, so nothing is marked
        // and the next flush writes all of it again.
        for planned in std::mem::take(&mut jobs.batched) {
            self.flush_gens.mark_flushed(&planned);
            self.flush_gens.record_write(&planned);
        }
        for (key, value) in meta_writes {
            self.flush_gens.mark_meta_written(key, value);
        }

        // The CRDT writes are durable now, so advance the frontiers and the
        // checkpoint accounting. Doing this only after the write means a failed
        // batch leaves every collection outstanding and the next flush retries
        // it.
        let evicted = {
            let mut crdt = self.crdt.lock_or_recover();
            crdt.mark_persisted(persisted);
            crdt.mark_pending_deltas_persisted(written_deltas);
            // Only now are the entries written above safe to page out: an entry
            // that exists only in memory is the sole copy of a local mutation.
            crdt.evict_pending_overflow()
        };
        if evicted > 0 {
            tracing::debug!(
                evicted,
                "paged unsent CRDT deltas out of the resident window; they stay queued \
                 under their `delta:` keys"
            );
        }

        // Refill the window from storage when acknowledgements have drained it.
        // A failure here costs a slower backlog drain, not correctness: the
        // entries are on disk and the next flush tries again.
        if let Err(e) = self.hydrate_crdt_delta_window().await {
            tracing::warn!(
                error = %e,
                "paging queued CRDT deltas back into the resident window failed; \
                 the backlog stays on disk and the next flush retries"
            );
        }
        self.query_engine.indexes.mark_flushed(index_flush);

        #[cfg(not(target_arch = "wasm32"))]
        super::segments::write_segments(self.storage.as_ref(), jobs, &self.flush_gens).await?;
        #[cfg(target_arch = "wasm32")]
        let _ = jobs;

        // ── Persist spatial indices (separate batch — includes docmap) ────────
        // Only dirty R-trees and collection doc-maps are serialized, and each
        // generation is captured under the spatial lock the bytes are read
        // under. A mutation after this point leaves its artifact dirty.
        let spatial_flush = self.spatial.lock_or_recover().checkpoint_dirty(full)?;
        crate::engine::spatial::checkpoint::write_serialized_spatial(
            self.storage.as_ref(),
            spatial_flush,
            &self.flush_gens,
        )
        .await?;

        // ── Persist FTS indices (separate batch — potentially large) ──
        // Only dirty indexes, a changed surrogate map, a changed index list,
        // and dropped indexes are planned. Planning is synchronous (no I/O)
        // and runs inside the lock, so each captured generation describes
        // exactly the bytes serialized. The writes run after it is released.
        let fts_flush = self
            .fts_state
            .manager
            .lock_or_recover()
            .checkpoint_dirty(full)
            .map_err(|e| NodeDbError::storage(format!("fts serialize: {e}")))?;
        let fts_written = crate::engine::fts::checkpoint::write_planned_fts(
            self.storage.as_ref(),
            fts_flush,
            &self.flush_gens,
        )
        .await
        .map_err(|e| NodeDbError::storage(format!("fts flush: {e}")))?;
        self.fts_state
            .manager
            .lock_or_recover()
            .forget_dropped(&fts_written);

        // ── Persist sparse-vector inverted indices ────────────────────────────
        // Same shape as the FTS block: serialize synchronously under the lock,
        // then perform the storage write after releasing it. Only dirty
        // indexes are serialized, and each generation is captured under the
        // same lock, so an insert after this point leaves its index dirty.
        let sparse_flush = self
            .sparse_state
            .manager
            .lock_or_recover()
            .checkpoint_dirty(full)
            .map_err(|e| NodeDbError::storage(format!("sparse serialize: {e}")))?;
        crate::engine::sparse_vector::checkpoint::write_serialized_sparse(
            self.storage.as_ref(),
            sparse_flush.ops,
        )
        .await
        .map_err(|e| NodeDbError::storage(format!("sparse flush: {e}")))?;
        // The batch committed, or was empty and skipped. Record the captured
        // generations, never the current ones.
        for planned in &sparse_flush.planned {
            self.flush_gens.mark_flushed(planned);
            self.flush_gens.record_write(planned);
        }
        if let Some((key, value)) = sparse_flush.meta {
            self.flush_gens.mark_meta_written(key, value);
        }

        // ── Spill FTS + spatial staging buffers to durable queues ────────────
        // These queues accumulate sync entries written synchronously by
        // `index_document_text`, `remove_document_text`, `spatial_insert`, and
        // `spatial_delete`. Spilling here (async, ~every second) keeps the
        // staging buffers bounded and ensures entries are durable before the
        // next sync transport drain.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(q) = &self.fts_outbound {
            q.flush_staging().await.map_err(NodeDbError::from)?;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(q) = &self.spatial_outbound {
            q.flush_staging().await.map_err(NodeDbError::from)?;
        }

        if fts_markers && self.fts_state.checkpoint_trusted() {
            let revisions = self
                .fts_state
                .manager
                .lock_or_recover()
                .declaration_revisions();
            crate::engine::fts::checkpoint::persist_checkpoint_complete(
                self.storage.as_ref(),
                &revisions,
            )
            .await
            .map_err(NodeDbError::from)?;
            self.flush_gens.mark_meta_written(
                crate::engine::fts::checkpoint::CHECKPOINT_STATE_KEY,
                crate::engine::fts::checkpoint::encode_checkpoint_complete(&revisions)
                    .map_err(NodeDbError::from)?,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::nodedb::LockExt;
    use crate::{LiteConfig, NodeDbLite, PagedbStorageMem};
    use nodedb_client::NodeDb;
    use nodedb_types::Value;
    use nodedb_types::document::Document;

    const WINDOW: usize = 8;
    const WRITES: usize = 40;

    /// A store whose delta window is far smaller than what it is about to
    /// queue, with no Origin to acknowledge any of it.
    async fn db_with_small_window() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.expect("storage");
        let config = LiteConfig {
            crdt_pending_delta_window: WINDOW,
            // Flushing is what enforces the window; do it explicitly rather
            // than racing a timer.
            auto_flush_ms: 0,
            // These tests are about the outbound queue, which only exists when
            // this store replicates: with sync off nothing is staged at all.
            sync_enabled: true,
            ..LiteConfig::default()
        };
        NodeDbLite::open_with_config(storage, config)
            .await
            .expect("open")
    }

    async fn write_documents(db: &NodeDbLite<PagedbStorageMem>) {
        for i in 0..WRITES {
            db.document_put("docs", Document::new(format!("d{i}")))
                .await
                .expect("document_put");
        }
    }

    #[tokio::test]
    async fn flush_bounds_the_resident_delta_window_without_losing_the_queue() {
        let db = db_with_small_window().await;
        write_documents(&db).await;

        db.flush().await.expect("flush");

        let (resident, total) = {
            let crdt = db.crdt.lock_or_recover();
            (crdt.resident_pending_count(), crdt.pending_count())
        };
        assert!(
            resident <= WINDOW,
            "resident window is {resident}, must not exceed {WINDOW}"
        );
        assert_eq!(
            total, WRITES,
            "no Origin acknowledged anything, so the queue must still hold every mutation"
        );
        assert_eq!(db.crdt_spilled_delta_count(), WRITES - resident);
        assert_eq!(
            db.health().engines.pending_deltas,
            WRITES,
            "health reports the queue, not the window"
        );
    }

    #[tokio::test]
    async fn the_retirement_sweep_keeps_every_paged_out_entry_on_disk() {
        let db = db_with_small_window().await;
        write_documents(&db).await;

        // Twice: the first flush pages entries out, the second sweeps the
        // stored keys with those entries nowhere in memory. Deleting them there
        // is exactly how the backlog would be lost.
        db.flush().await.expect("first flush");
        db.flush().await.expect("second flush");

        let stored = db.persisted_delta_ids().await.expect("scan");
        assert_eq!(
            stored.len(),
            WRITES,
            "every queued mutation must still have its stored entry"
        );
        assert_eq!(
            db.crdt.lock_or_recover().pending_count(),
            WRITES,
            "the queue survives a sweep that cannot see most of it"
        );
    }

    #[tokio::test]
    async fn acknowledging_a_paged_out_entry_deletes_its_stored_entry() {
        let db = db_with_small_window().await;
        write_documents(&db).await;
        db.flush().await.expect("flush");

        let acked = {
            let crdt = db.crdt.lock_or_recover();
            let resident: std::collections::HashSet<u64> = crdt
                .pending_deltas()
                .iter()
                .map(|d| d.mutation_id)
                .collect();
            (1..=WRITES as u64)
                .find(|id| !resident.contains(id))
                .expect("some entry was paged out")
        };

        db.acknowledge_deltas(acked).expect("acknowledge");
        db.flush().await.expect("flush");

        let stored = db.persisted_delta_ids().await.expect("scan");
        assert!(
            !stored.contains(&acked),
            "the acknowledged entry's stored form must be deleted even though it was \
             never paged back in"
        );
        assert_eq!(stored.len(), WRITES - 1);
        assert_eq!(db.crdt.lock_or_recover().pending_count(), WRITES - 1);
    }

    #[tokio::test]
    async fn draining_the_window_pages_the_backlog_back_in_oldest_first() {
        let db = db_with_small_window().await;
        write_documents(&db).await;
        db.flush().await.expect("flush");

        let head: Vec<u64> = db
            .crdt
            .lock_or_recover()
            .pending_deltas()
            .iter()
            .map(|d| d.mutation_id)
            .collect();
        for id in &head {
            db.acknowledge_deltas(*id).expect("acknowledge");
        }
        assert_eq!(db.crdt.lock_or_recover().resident_pending_count(), 0);

        db.flush().await.expect("flush");

        let crdt = db.crdt.lock_or_recover();
        let resident: Vec<u64> = crdt
            .pending_deltas()
            .iter()
            .map(|d| d.mutation_id)
            .collect();
        assert_eq!(
            resident.len(),
            WINDOW,
            "the window refills from storage once acknowledgements drain it"
        );
        assert_eq!(
            resident.first().copied(),
            Some(head.len() as u64 + 1),
            "the oldest unacknowledged entry is paged in first, so replay stays in order"
        );
        assert!(
            crdt.pending_deltas()
                .iter()
                .all(|d| !d.delta_bytes.is_empty()),
            "a paged-in entry carries the payload that will be pushed"
        );
        assert_eq!(crdt.pending_count(), WRITES - head.len());
    }

    #[tokio::test]
    async fn successful_flush_preserves_interrupted_mutation_distrust() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let db = NodeDbLite::open_with_config(
            storage,
            LiteConfig {
                auto_flush_ms: 0,
                sync_enabled: false,
                ..LiteConfig::default()
            },
        )
        .await
        .unwrap();
        drop(db.fts_state.admit_mutation().await);
        let mut document = Document::new("one");
        document.set("body", Value::String("laterwrite".into()));
        db.document_put("articles", document).await.unwrap();
        db.flush().await.unwrap();
        let revisions = db
            .fts_state
            .manager
            .lock_or_recover()
            .declaration_revisions();
        assert!(!crate::engine::fts::checkpoint::checkpoint_compatible(
            db.storage.as_ref(), &revisions,
        ).await.unwrap());
        assert!(!db.fts_state.checkpoint_trusted());
    }
}
