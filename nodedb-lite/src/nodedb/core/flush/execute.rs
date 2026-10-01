// SPDX-License-Identifier: Apache-2.0

//! `NodeDbLite::flush` — persist all in-memory state to storage.

use crate::storage::engine::{StorageEngine, WriteOp};
use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::crdt::{CrdtEngine, CrdtWriteKind};
use crate::engine::fts::coordinator::TextMutationPermit;
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

    /// Persist all in-memory state to storage (call before shutdown).
    pub async fn flush(&self) -> NodeDbResult<()> {
        let permit = self.fts_state.admit_exclusive().await;
        self.flush_admitted(&permit).await
    }

    pub(crate) async fn flush_admitted(&self, _permit: &TextMutationPermit) -> NodeDbResult<()> {
        // One flush at a time: the CRDT update sequence is allocated under the
        // `crdt` guard but committed after it is released, so concurrent
        // flushes would hand out the same numbers. See `flush_lock`.
        let _flush_guard = self.flush_lock.lock().await;
        crate::engine::fts::checkpoint::persist_checkpoint_incomplete(self.storage.as_ref())
            .await
            .map_err(NodeDbError::from)?;

        // Drain the buffered KV writes first — they have their own batch-commit
        // path. Without this, `flush()` (and the auto-flush timer) would not
        // persist KV `put`s, contradicting "persist all in-memory state".
        self.kv_flush_inner().await?;

        let mut ops = Vec::new();

        // Delta entries already on disk. Restore prefers these over the bulk
        // blob, so any one that is no longer pending — acknowledged by Origin,
        // or replaced by a peer-id rotation that re-authored the row — would
        // come back on the next open and be pushed again. They are deleted in
        // the same batch that writes the current set.
        let persisted_delta_keys: Vec<Vec<u8>> = self
            .storage
            .scan_prefix(Namespace::Crdt, b"delta:")
            .await?
            .into_iter()
            .map(|(key, _)| key)
            .collect();

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
            let pending = crdt.pending_deltas();
            let max_mid = pending.iter().map(|d| d.mutation_id).max().unwrap_or(0);

            let live_keys: std::collections::HashSet<Vec<u8>> = pending
                .iter()
                .map(|d| CrdtEngine::delta_storage_key(d.mutation_id))
                .collect();
            let mut retired_any = false;
            for key in persisted_delta_keys {
                if !live_keys.contains(&key) {
                    retired_any = true;
                    ops.push(WriteOp::Delete {
                        ns: Namespace::Crdt,
                        key,
                    });
                }
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

            // Legacy bulk blob (for clients that haven't upgraded to incremental
            // restore). It duplicates every entry above, so it is rewritten only
            // when the queue actually changed rather than on every tick.
            let queue_changed = retired_any || crdt.has_unpersisted_deltas();
            if queue_changed {
                let deltas_bulk = crdt
                    .serialize_pending_deltas()
                    .map_err(NodeDbError::storage)?;
                ops.push(WriteOp::Put {
                    ns: Namespace::Crdt,
                    key: META_CRDT_DELTAS.to_vec(),
                    value: deltas_bulk,
                });
            }

            // Index entries in the same batch and lock hold as the rows.
            let index_flush = self.query_engine.indexes.stage_flush(&mut ops);

            // Write the last-flushed mutation_id for partial flush safety.
            ops.push(WriteOp::Put {
                ns: Namespace::Meta,
                key: META_LAST_FLUSHED_MID.to_vec(),
                value: max_mid.to_le_bytes().to_vec(),
            });

            (persisted, written_deltas, index_flush)
        };

        let jobs = self.stage_indexes(&mut ops)?;
        self.storage
            .batch_write(&ops)
            .await
            .map_err(NodeDbError::storage)?;

        // The CRDT writes are durable now, so advance the frontiers and the
        // checkpoint accounting. Doing this only after the write means a failed
        // batch leaves every collection outstanding and the next flush retries
        // it.
        {
            let mut crdt = self.crdt.lock_or_recover();
            crdt.mark_persisted(persisted);
            crdt.mark_pending_deltas_persisted(written_deltas);
        }
        self.query_engine.indexes.mark_flushed(index_flush);

        #[cfg(not(target_arch = "wasm32"))]
        super::segments::write_segments(self.storage.as_ref(), jobs).await?;
        #[cfg(target_arch = "wasm32")]
        let _ = jobs;

        // ── Persist spatial indices (separate batch — includes docmap) ────────
        let (spatial_checkpoints, spatial_doc_to_entry, spatial_next_id) =
            self.spatial.lock_or_recover().checkpoint_data();
        crate::engine::spatial::checkpoint::flush_spatial(
            self.storage.as_ref(),
            &spatial_checkpoints,
            &spatial_doc_to_entry,
            spatial_next_id,
        )
        .await?;

        // ── Persist FTS indices (separate batch — potentially large) ──
        // Serialize is synchronous (no I/O); do it inside the lock so we don't
        // need to clone FtsIndex.  The resulting ops + segment blobs are written
        // to storage after the lock is released.
        let (fts_ops, fts_segment_writes) = {
            let fts = self.fts_state.manager.lock_or_recover();
            let (indices, id_to_surrogate, next_surrogate) = fts.checkpoint_data();
            crate::engine::fts::checkpoint::serialize_fts(indices, id_to_surrogate, next_surrogate)
                .map_err(|e| NodeDbError::storage(format!("fts serialize: {e}")))?
        };
        crate::engine::fts::checkpoint::write_serialized_fts(
            self.storage.as_ref(),
            fts_ops,
            fts_segment_writes,
        )
        .await
        .map_err(|e| NodeDbError::storage(format!("fts flush: {e}")))?;

        // ── Persist sparse-vector inverted indices ────────────────────────────
        // Same shape as the FTS block: serialize synchronously under the lock,
        // then perform the storage write after releasing it.
        let sparse_ops = {
            let sparse = self.sparse_state.manager.lock_or_recover();
            crate::engine::sparse_vector::checkpoint::serialize_sparse(sparse.checkpoint_data())
                .map_err(|e| NodeDbError::storage(format!("sparse serialize: {e}")))?
        };
        crate::engine::sparse_vector::checkpoint::write_serialized_sparse(
            self.storage.as_ref(),
            sparse_ops,
        )
        .await
        .map_err(|e| NodeDbError::storage(format!("sparse flush: {e}")))?;

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

        if self.fts_state.checkpoint_trusted() {
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
