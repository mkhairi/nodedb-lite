// SPDX-License-Identifier: Apache-2.0

//! `NodeDbLite::flush` — persist all in-memory state to storage.

use crate::storage::engine::{StorageEngine, WriteOp};
use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::crdt::{CrdtEngine, CrdtWriteKind};
use crate::nodedb::lock_ext::LockExt;

use super::types::{
    META_CRDT_DELTAS, META_CSR_COLLECTIONS, META_HNSW_COLLECTIONS, META_LAST_FLUSHED_MID,
    NodeDbLite,
};

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
    pub async fn flush(&self) -> NodeDbResult<()> {
        // One flush at a time: the CRDT update sequence is allocated under the
        // `crdt` guard but committed after it is released, so concurrent
        // flushes would hand out the same numbers. See `flush_lock`.
        let _flush_guard = self.flush_lock.lock().await;

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
        let persisted_delta_ids = if self.sync_enabled {
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
        let (persisted, written_deltas) = {
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
            let retired = if self.sync_enabled {
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

            // Write the last-flushed mutation_id for partial flush safety.
            ops.push(WriteOp::Put {
                ns: Namespace::Meta,
                key: META_LAST_FLUSHED_MID.to_vec(),
                value: max_mid.to_le_bytes().to_vec(),
            });

            (persisted, written_deltas)
        };

        // ── Persist per-collection CSR indices ──
        // When the pagedb segment extension is available (native PagedbStorage):
        //   - CSR blob → pagedb segment (written after batch_write)
        //   - B+ tree receives only the collection-name index (META_CSR_COLLECTIONS)
        // Otherwise (WASM or non-pagedb native backends):
        //   - CSR blob → B+ tree (Namespace::Graph, CRC32C wrapped)
        #[cfg(not(target_arch = "wasm32"))]
        let graph_seg_ext = self.storage.as_graph_segment_ext();
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
        let csr_segment_data: Vec<(String, Vec<u8>)> = {
            let csr_map = self.csr.lock_or_recover();
            let names: Vec<String> = csr_map.keys().cloned().collect();
            let names_bytes = zerompk::to_msgpack_vec(&names)
                .map_err(|e| NodeDbError::serialization("msgpack", e))?;
            ops.push(WriteOp::Put {
                ns: Namespace::Meta,
                key: META_CSR_COLLECTIONS.to_vec(),
                value: names_bytes,
            });

            // Mutated only via the native segment-ext path, compiled out on wasm32.
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let mut segment_data: Vec<(String, Vec<u8>)> = Vec::new();
            for (name, index) in csr_map.iter() {
                match index.checkpoint_to_bytes() {
                    Ok(checkpoint) => {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            if graph_seg_ext.is_some() {
                                // Pagedb segment path: collect for post-batch write.
                                segment_data.push((name.clone(), checkpoint));
                            } else {
                                // Legacy B+ tree path.
                                let key = format!("csr:{name}");
                                ops.push(WriteOp::Put {
                                    ns: Namespace::Graph,
                                    key: key.into_bytes(),
                                    value: crate::storage::checksum::wrap(&checkpoint),
                                });
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            let key = format!("csr:{name}");
                            ops.push(WriteOp::Put {
                                ns: Namespace::Graph,
                                key: key.into_bytes(),
                                value: crate::storage::checksum::wrap(&checkpoint),
                            });
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            collection = %name,
                            error = %e,
                            "CSR checkpoint failed for collection; graph state not persisted"
                        );
                    }
                }
            }
            segment_data
        };

        // ── Persist HNSW vector_id_map ──
        // The id_map is a flat HashMap<composite_key, (doc_id, internal_id)>
        // serialized as one MessagePack blob. It must be written before any restart
        // so that vector_search can return real doc_ids (not HNSW integer strings).
        // Vector search with an empty id_map after restart is the bug this fixes.
        // Vectors are flush-only (no per-insert durability path); the id_map
        // follows the same durability contract — flush required.
        {
            let id_map = self.vector_state.vector_id_map.lock_or_recover();
            // Serialize as Vec<(composite_key, doc_id, internal_id)> for stable msgpack encoding.
            let entries: Vec<(&str, &str, u32)> = id_map
                .iter()
                .map(|(k, (doc_id, iid))| (k.as_str(), doc_id.as_str(), *iid))
                .collect();
            match zerompk::to_msgpack_vec(&entries) {
                Ok(bytes) => {
                    ops.push(WriteOp::Put {
                        ns: Namespace::Vector,
                        key: b"hnsw_id_map".to_vec(),
                        value: crate::storage::checksum::wrap(&bytes),
                    });
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "vector_id_map serialization failed; \
                         vector search after restart will fall back to HNSW integer IDs"
                    );
                }
            }
        }

        // ── Persist HNSW indices ──
        // When the pagedb segment extension is available (native PagedbStorage):
        //   - graph topology blob → B+ tree (graph_checkpoint_to_bytes; empty vector slots)
        //   - vector data → pagedb segment (written after batch_write)
        // Otherwise (WASM or legacy backends):
        //   - full checkpoint blob → B+ tree (checkpoint_to_bytes)
        #[cfg(not(target_arch = "wasm32"))]
        let seg_ext = self.storage.as_vector_segment_ext();
        #[cfg_attr(
            target_arch = "wasm32",
            allow(unused_variables, clippy::type_complexity)
        )]
        #[allow(clippy::type_complexity)]
        let hnsw_segment_names: Vec<String> = {
            let indices = self.vector_state.hnsw_indices.lock_or_recover();
            let names: Vec<String> = indices.keys().cloned().collect();
            let names_bytes = zerompk::to_msgpack_vec(&names)
                .map_err(|e| NodeDbError::serialization("msgpack", e))?;
            ops.push(WriteOp::Put {
                ns: Namespace::Meta,
                key: META_HNSW_COLLECTIONS.to_vec(),
                value: names_bytes,
            });

            // Mutated only via the native segment-ext path, compiled out on wasm32.
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let mut segment_names: Vec<String> = Vec::new();
            for (name, index) in indices.iter() {
                let key = format!("hnsw:{name}");

                #[cfg(not(target_arch = "wasm32"))]
                {
                    if seg_ext.is_some() {
                        // Graph-only blob (vector bytes are empty placeholders).
                        let graph_bytes = index
                            .graph_checkpoint_to_bytes()
                            .map_err(|e| NodeDbError::serialization("hnsw-graph-checkpoint", e))?;
                        ops.push(WriteOp::Put {
                            ns: Namespace::Vector,
                            key: key.into_bytes(),
                            value: crate::storage::checksum::wrap(&graph_bytes),
                        });
                        // The segment payload is sourced from the DURABLE vectors
                        // after this lock is released, NOT from `index` — see
                        // `engine::vector::durable::segment_payload`. Reading it
                        // from the index would serialize empty vectors whenever
                        // the index was restored from a graph-only checkpoint.
                        segment_names.push(name.clone());
                    } else {
                        // Non-pagedb native backend: full checkpoint blob path.
                        let checkpoint = index
                            .checkpoint_to_bytes()
                            .map_err(|e| NodeDbError::serialization("hnsw-checkpoint", e))?;
                        ops.push(WriteOp::Put {
                            ns: Namespace::Vector,
                            key: key.into_bytes(),
                            value: crate::storage::checksum::wrap(&checkpoint),
                        });
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    // WASM: full checkpoint blob path (no segment ops).
                    let checkpoint = index
                        .checkpoint_to_bytes()
                        .map_err(|e| NodeDbError::serialization("hnsw-checkpoint", e))?;
                    ops.push(WriteOp::Put {
                        ns: Namespace::Vector,
                        key: key.into_bytes(),
                        value: crate::storage::checksum::wrap(&checkpoint),
                    });
                }
            }
            segment_names
        };

        self.storage
            .batch_write(&ops)
            .await
            .map_err(NodeDbError::storage)?;

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

        // ── Write HNSW vector segments to pagedb (native PagedbStorage only) ──
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ext) = seg_ext {
            for name in &hnsw_segment_names {
                let payload =
                    match crate::engine::vector::durable::segment_payload(&*self.storage, name)
                        .await
                    {
                        Ok(Some(p)) => p,
                        // No durable vectors: nothing to publish. Leaving the
                        // existing segment untouched is deliberate — replacing it
                        // with an empty one is exactly the corruption this fixes.
                        Ok(None) => continue,
                        Err(e) => {
                            tracing::error!(
                                collection = %name,
                                error = %e,
                                "reading durable vectors for the segment write failed; \
                                 leaving the existing segment in place"
                            );
                            continue;
                        }
                    };
                let (dim, vectors, surrogates) = payload;
                if let Err(e) = ext
                    .write_vector_segment(name, dim, &vectors, &surrogates)
                    .await
                {
                    tracing::error!(
                        collection = %name,
                        error = %e,
                        "HNSW vector segment write failed; \
                         graph topology is persisted but vectors may be lost on cold restart"
                    );
                }
            }
        }

        // ── Write CSR adjacency segments to pagedb (native PagedbStorage only) ──
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ext) = graph_seg_ext {
            for (name, checkpoint) in &csr_segment_data {
                if let Err(e) = ext.write_graph_segment(name, checkpoint).await {
                    tracing::error!(
                        collection = %name,
                        error = %e,
                        "CSR adjacency segment write failed; \
                         graph state may be lost on cold restart"
                    );
                }
            }
        }

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
        if let Some(q) = &self.fts_outbound
            && let Err(e) = q.flush_staging().await
        {
            tracing::warn!(error = %e, "fts outbound flush_staging failed; \
                    staged entries remain and will be retried on next flush");
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(q) = &self.spatial_outbound
            && let Err(e) = q.flush_staging().await
        {
            tracing::warn!(error = %e, "spatial outbound flush_staging failed; \
                    staged entries remain and will be retried on next flush");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nodedb_client::NodeDb;
    use nodedb_types::document::Document;

    use crate::PagedbStorageMem;
    use crate::config::LiteConfig;

    use super::*;

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
}
