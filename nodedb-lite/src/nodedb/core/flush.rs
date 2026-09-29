// SPDX-License-Identifier: Apache-2.0

//! `NodeDbLite::flush` — persist all in-memory state to storage.

use crate::storage::engine::{StorageEngine, WriteOp};
use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::crdt::{CrdtEngine, CrdtWriteKind};
use crate::nodedb::flush_gens::{ArtifactFlush, FlushArtifact, ID_MAP_KEY};
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

    /// Number of successful writes of `artifact` for `collection` since this
    /// handle was opened.
    ///
    /// [`FlushArtifact::HnswIdMap`] is one store-wide blob: pass
    /// [`ID_MAP_KEY`] (the empty string) as its collection. A counter, not a
    /// timing, so a caller can assert on it directly: an idle store must not
    /// advance it.
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
    /// graph checkpoint, or a meta entry that has not changed since this
    /// handle last made it durable is not written again.
    /// [`flush_full`](Self::flush_full) writes all of them regardless.
    pub async fn flush(&self) -> NodeDbResult<()> {
        self.flush_pass(false).await
    }

    /// Persist all in-memory state, writing every dirty-tracked artifact
    /// whether or not it changed, then mark each one flushed.
    ///
    /// For callers that do not trust the stored form, e.g. after an external
    /// tool touched the file. Artifacts outside the dirty tracking behave as in
    /// [`flush`](Self::flush).
    pub async fn flush_full(&self) -> NodeDbResult<()> {
        self.flush_pass(true).await
    }

    /// One flush pass. `full` ignores the dirty state of tracked artifacts.
    async fn flush_pass(&self, full: bool) -> NodeDbResult<()> {
        // One flush at a time: the CRDT update sequence is allocated under the
        // `crdt` guard but committed after it is released, so concurrent
        // flushes would hand out the same numbers. See `flush_lock`.
        let _flush_guard = self.flush_lock.lock().await;

        // Drain the buffered KV writes first — they have their own batch-commit
        // path. Without this, `flush()` (and the auto-flush timer) would not
        // persist KV `put`s, contradicting "persist all in-memory state".
        self.kv_flush_inner().await?;

        let mut ops = Vec::new();
        // Meta entries queued in `ops`, recorded as written once the batch
        // commits. An unchanged meta value is not written again.
        let mut meta_writes: Vec<(&[u8], Vec<u8>)> = Vec::new();

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

            (persisted, written_deltas)
        };

        // ── Persist per-collection CSR indices ──
        // When the pagedb segment extension is available (native PagedbStorage):
        //   - CSR blob → pagedb segment (written after batch_write)
        //   - B+ tree receives only the collection-name index (META_CSR_COLLECTIONS)
        // Otherwise (WASM or non-pagedb native backends):
        //   - CSR blob → B+ tree (Namespace::Graph, CRC32C wrapped)
        //
        // Each collection's checkpoint is serialized and written only when
        // dirty. Its generation is captured here, under the CSR lock the bytes
        // are serialized under, so an edge added after this point leaves the
        // collection dirty for the next flush.
        #[cfg(not(target_arch = "wasm32"))]
        let graph_seg_ext = self.storage.as_graph_segment_ext();
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
        let (csr_blob_flushes, csr_segment_flushes) = {
            let csr_map = self.csr.lock_or_recover();
            // Sorted so the encoded list is a function of the set of names,
            // not of the map's iteration order, and compares equal across
            // ticks while the set is unchanged.
            let mut entries: Vec<_> = csr_map.iter().collect();
            entries.sort_by_key(|(name, _)| *name);
            let names: Vec<String> = entries.iter().map(|(name, _)| (*name).clone()).collect();
            let names_bytes = zerompk::to_msgpack_vec(&names)
                .map_err(|e| NodeDbError::serialization("msgpack", e))?;
            if full
                || self
                    .flush_gens
                    .meta_changed(META_CSR_COLLECTIONS, &names_bytes)
            {
                ops.push(WriteOp::Put {
                    ns: Namespace::Meta,
                    key: META_CSR_COLLECTIONS.to_vec(),
                    value: names_bytes.clone(),
                });
                meta_writes.push((META_CSR_COLLECTIONS, names_bytes));
            }

            // Blobs put in the batch below, marked flushed once it commits.
            let mut blob_flushes: Vec<ArtifactFlush> = Vec::new();
            // Segments written after the batch, each marked flushed once its
            // own write succeeds. Mutated only via the native segment-ext
            // path, compiled out on wasm32.
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let mut segment_flushes: Vec<(ArtifactFlush, Vec<u8>)> = Vec::new();
            for (name, index) in entries {
                let Some(planned) = self.flush_gens.plan(FlushArtifact::CsrGraph, name, full)
                else {
                    continue;
                };
                match index.checkpoint_to_bytes() {
                    Ok(checkpoint) => {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            if graph_seg_ext.is_some() {
                                // Pagedb segment path: collect for post-batch write.
                                segment_flushes.push((planned, checkpoint));
                            } else {
                                // Legacy B+ tree path.
                                let key = format!("csr:{name}");
                                ops.push(WriteOp::Put {
                                    ns: Namespace::Graph,
                                    key: key.into_bytes(),
                                    value: crate::storage::checksum::wrap(&checkpoint),
                                });
                                blob_flushes.push(planned);
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
                            blob_flushes.push(planned);
                        }
                    }
                    Err(e) => {
                        // Not planned as written, so it stays dirty and the
                        // next flush tries again.
                        tracing::error!(
                            collection = %name,
                            error = %e,
                            "CSR checkpoint failed for collection; graph state not persisted"
                        );
                    }
                }
            }
            (blob_flushes, segment_flushes)
        };

        // ── Persist HNSW vector_id_map ──
        // The id_map is a flat HashMap<composite_key, (doc_id, internal_id)>
        // serialized as one MessagePack blob. It must be written before any restart
        // so that vector_search can return real doc_ids (not HNSW integer strings).
        // Vector search with an empty id_map after restart is the bug this fixes.
        // Vectors are flush-only (no per-insert durability path); the id_map
        // follows the same durability contract — flush required.
        //
        // Written only when dirty. The generation is captured under the same
        // lock the entries are read under, so a bind made after this point
        // leaves the id-map dirty for the next flush.
        #[cfg(not(target_arch = "wasm32"))]
        let seg_ext = self.storage.as_vector_segment_ext();
        // Collections whose vector segment this flush can write: a segment is
        // rewritten when its rows or its graph changed. Their slot bindings
        // are read below under the id-map lock, in the same critical section
        // that serializes the id-map, so each segment's stamps name exactly the
        // bindings the stored id-map holds. The index lock is not held there:
        // the id-map lock is never taken inside the index lock in a new order.
        #[cfg(not(target_arch = "wasm32"))]
        let segment_candidates: std::collections::HashSet<String> = if seg_ext.is_some() {
            let indices = self.vector_state.hnsw_indices.lock_or_recover();
            indices
                .keys()
                .filter(|name| {
                    full || self.flush_gens.is_dirty(FlushArtifact::HnswGraph, name)
                        || self.flush_gens.is_dirty(FlushArtifact::VectorSegment, name)
                })
                .cloned()
                .collect()
        } else {
            std::collections::HashSet::new()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let mut segment_bindings: std::collections::HashMap<
            String,
            crate::engine::vector::durable::SlotBindings,
        > = std::collections::HashMap::new();
        let id_map_flush = {
            let id_map = self.vector_state.vector_id_map.lock_or_recover();
            #[cfg(not(target_arch = "wasm32"))]
            if !segment_candidates.is_empty() {
                segment_bindings =
                    crate::engine::vector::durable::slot_bindings(id_map.iter(), |index_key| {
                        segment_candidates.contains(index_key)
                    });
                // A candidate with no binding at all still gets an entry: its
                // segment is written with every slot stamped `TOMB`.
                for name in &segment_candidates {
                    segment_bindings.entry(name.clone()).or_default();
                }
            }
            match self
                .flush_gens
                .plan(FlushArtifact::HnswIdMap, ID_MAP_KEY, full)
            {
                None => None,
                Some(planned) => {
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
                            Some(planned)
                        }
                        Err(e) => {
                            // Not planned as written, so it stays dirty and the
                            // next flush tries again.
                            tracing::error!(
                                error = %e,
                                "vector_id_map serialization failed; \
                                 vector search after restart will fall back to HNSW integer IDs"
                            );
                            None
                        }
                    }
                }
            }
        };

        // ── Persist HNSW indices ──
        // When the pagedb segment extension is available (native PagedbStorage):
        //   - graph topology blob → B+ tree (graph_checkpoint_to_bytes; empty vector slots)
        //   - vector data → pagedb segment (written after batch_write)
        // Otherwise (WASM or legacy backends):
        //   - full checkpoint blob → B+ tree (checkpoint_to_bytes)
        //
        // Each collection's graph blob and vector segment is written only when
        // dirty. Both generations are captured here, under the index lock, and
        // both payloads are built under the same lock: a graph mutation after
        // this point, or a durable vector row written after it, leaves the
        // artifact dirty for the next flush.
        #[cfg_attr(
            target_arch = "wasm32",
            allow(unused_variables, clippy::type_complexity)
        )]
        #[allow(clippy::type_complexity)]
        let (graph_flushes, segment_flushes) = {
            let indices = self.vector_state.hnsw_indices.lock_or_recover();
            // Sorted so the encoded list is a function of the set of names,
            // not of the map's iteration order, and compares equal across
            // ticks while the set is unchanged.
            let mut names: Vec<String> = indices.keys().cloned().collect();
            names.sort();
            let names_bytes = zerompk::to_msgpack_vec(&names)
                .map_err(|e| NodeDbError::serialization("msgpack", e))?;
            if full
                || self
                    .flush_gens
                    .meta_changed(META_HNSW_COLLECTIONS, &names_bytes)
            {
                ops.push(WriteOp::Put {
                    ns: Namespace::Meta,
                    key: META_HNSW_COLLECTIONS.to_vec(),
                    value: names_bytes.clone(),
                });
                meta_writes.push((META_HNSW_COLLECTIONS, names_bytes));
            }

            let mut graph_flushes: Vec<ArtifactFlush> = Vec::new();
            // Mutated only via the native segment-ext path, compiled out on wasm32.
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            // Each entry is (plan, (dim, vectors, stamps)).
            let mut segment_flushes: Vec<(
                ArtifactFlush,
                (usize, Vec<Vec<f32>>, Vec<u64>),
            )> = Vec::new();
            for (name, index) in indices.iter() {
                let key = format!("hnsw:{name}");
                let graph_plan = self.flush_gens.plan(FlushArtifact::HnswGraph, name, full);

                #[cfg(not(target_arch = "wasm32"))]
                {
                    if seg_ext.is_some() {
                        let graph_dirty = graph_plan.is_some();
                        if let Some(planned) = graph_plan {
                            // Graph-only blob (vector bytes are empty placeholders).
                            let graph_bytes = index.graph_checkpoint_to_bytes().map_err(|e| {
                                NodeDbError::serialization("hnsw-graph-checkpoint", e)
                            })?;
                            ops.push(WriteOp::Put {
                                ns: Namespace::Vector,
                                key: key.into_bytes(),
                                value: crate::storage::checksum::wrap(&graph_bytes),
                            });
                            graph_flushes.push(planned);
                        }
                        // The segment is built from `index` in node order, here
                        // under the lock the graph blob is serialized under, so
                        // its vectors and stamps describe exactly that graph.
                        // See `engine::vector::durable::segment_payload_from_index`.
                        // A dirty graph rewrites the segment too: a graph whose
                        // nodes changed no longer matches the stored segment.
                        // A collection that became dirty after its bindings
                        // were read has none here and stays dirty.
                        if let Some(planned) = self.flush_gens.plan(
                            FlushArtifact::VectorSegment,
                            name,
                            full || graph_dirty,
                        ) && let Some(bindings) = segment_bindings.get(name)
                        {
                            if let Some(slot) =
                                crate::engine::vector::durable::unbound_live_slot(index, bindings)
                            {
                                // An insert is between its index write and its
                                // bind. The segment stays dirty and the next
                                // flush stamps it with the bind in place.
                                tracing::debug!(
                                    collection = %name,
                                    slot,
                                    "vector segment deferred: a live node is not bound yet"
                                );
                            } else {
                                match crate::engine::vector::durable::segment_payload_from_index(
                                    index, bindings,
                                ) {
                                    Ok(payload) => segment_flushes.push((planned, payload)),
                                    Err(e) => {
                                        // Not planned as written, so it stays
                                        // dirty and the stored segment is kept.
                                        tracing::error!(
                                            collection = %name,
                                            error = %e,
                                            "building the vector segment failed; \
                                             leaving the existing segment in place"
                                        );
                                    }
                                }
                            }
                        }
                    } else if let Some(planned) = graph_plan {
                        // Non-pagedb native backend: full checkpoint blob path.
                        let checkpoint = index
                            .checkpoint_to_bytes()
                            .map_err(|e| NodeDbError::serialization("hnsw-checkpoint", e))?;
                        ops.push(WriteOp::Put {
                            ns: Namespace::Vector,
                            key: key.into_bytes(),
                            value: crate::storage::checksum::wrap(&checkpoint),
                        });
                        graph_flushes.push(planned);
                    }
                }
                #[cfg(target_arch = "wasm32")]
                if let Some(planned) = graph_plan {
                    // WASM: full checkpoint blob path (no segment ops).
                    let checkpoint = index
                        .checkpoint_to_bytes()
                        .map_err(|e| NodeDbError::serialization("hnsw-checkpoint", e))?;
                    ops.push(WriteOp::Put {
                        ns: Namespace::Vector,
                        key: key.into_bytes(),
                        value: crate::storage::checksum::wrap(&checkpoint),
                    });
                    graph_flushes.push(planned);
                }
            }
            (graph_flushes, segment_flushes)
        };

        self.storage
            .batch_write(&ops)
            .await
            .map_err(NodeDbError::storage)?;

        // The graph blobs, the CSR blobs, the id-map, and the meta entries in
        // the batch are durable now. Record the generations captured when they
        // were serialized, never the current ones: a mutation made since keeps
        // its artifact dirty. A failed batch returned above, so nothing is
        // marked and the next flush writes all of it again.
        for planned in graph_flushes
            .iter()
            .chain(csr_blob_flushes.iter())
            .chain(id_map_flush.iter())
        {
            self.flush_gens.mark_flushed(planned);
            self.flush_gens.record_write(planned);
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

        // ── Write HNSW vector segments to pagedb (native PagedbStorage only) ──
        //
        // Only collections whose durable rows or graph changed since their
        // segment was last written are planned, with the payload built when
        // they were planned. A segment is marked flushed only after its write
        // succeeds; the error branch below leaves it dirty, so the next flush
        // retries it.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ext) = seg_ext {
            for (planned, (dim, vectors, stamps)) in &segment_flushes {
                let name = planned.key();
                match ext.write_vector_segment(name, *dim, vectors, stamps).await {
                    Ok(()) => {
                        self.flush_gens.mark_flushed(planned);
                        self.flush_gens.record_write(planned);
                    }
                    Err(e) => {
                        tracing::error!(
                            collection = %name,
                            error = %e,
                            "HNSW vector segment write failed; \
                             graph topology is persisted but vectors may be lost on cold restart; \
                             the segment stays dirty and the next flush retries it"
                        );
                    }
                }
            }
        }

        // ── Write CSR adjacency segments to pagedb (native PagedbStorage only) ──
        //
        // Only dirty collections are planned. A segment is marked flushed only
        // after its write succeeds; the error branch below leaves it dirty, so
        // the next flush retries it.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ext) = graph_seg_ext {
            for (planned, checkpoint) in &csr_segment_flushes {
                let name = planned.key();
                match ext.write_graph_segment(name, checkpoint).await {
                    Ok(()) => {
                        self.flush_gens.mark_flushed(planned);
                        self.flush_gens.record_write(planned);
                    }
                    Err(e) => {
                        tracing::error!(
                            collection = %name,
                            error = %e,
                            "CSR adjacency segment write failed; \
                             graph state may be lost on cold restart; \
                             the segment stays dirty and the next flush retries it"
                        );
                    }
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

    const GRAPH_A: &str = "graph_a";
    const GRAPH_B: &str = "graph_b";

    /// A store holding one flushed edge in each of two graph collections,
    /// both written through the query engine's graph ops.
    async fn db_with_two_clean_graphs() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.expect("storage");
        let config = LiteConfig {
            auto_flush_ms: 0,
            ..LiteConfig::default()
        };
        let db = NodeDbLite::open_with_config(storage, config)
            .await
            .expect("open");
        for collection in [GRAPH_A, GRAPH_B] {
            query_edge_put(&db, collection, "a", "b").await;
        }
        db.flush().await.expect("flush");
        for collection in [GRAPH_A, GRAPH_B] {
            assert!(
                !db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, collection),
                "a completed flush leaves {collection} clean"
            );
        }
        db
    }

    async fn query_edge_put(
        db: &NodeDbLite<PagedbStorageMem>,
        collection: &str,
        src: &str,
        dst: &str,
    ) {
        let memory = db.memory_for(nodedb_mem::EngineId::Graph);
        crate::query::graph_ops::edges::edge_put(
            &db.storage,
            &db.query_engine.csr,
            &memory,
            crate::query::graph_ops::edges::EdgePutArgs {
                collection,
                src_id: src,
                label: "LINK",
                dst_id: dst,
                properties: &[],
            },
        )
        .await
        .expect("edge_put");
    }

    /// The query engine shares the store's CSR map, so an edge it adds
    /// dirties the collection it names and no other.
    #[tokio::test]
    async fn query_path_edge_put_dirties_only_its_collection() {
        let db = db_with_two_clean_graphs().await;

        query_edge_put(&db, GRAPH_A, "b", "c").await;

        assert!(db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, GRAPH_A));
        assert!(
            !db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, GRAPH_B),
            "an edge in one collection must not dirty another"
        );
        let b_before = db.flush_artifact_write_count(FlushArtifact::CsrGraph, GRAPH_B);
        db.flush().await.expect("flush");
        assert!(!db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, GRAPH_A));
        assert_eq!(
            db.flush_artifact_write_count(FlushArtifact::CsrGraph, GRAPH_B),
            b_before,
            "the untouched collection is not rewritten"
        );
    }

    /// A node label added through the query engine dirties its collection.
    #[tokio::test]
    async fn query_path_node_label_dirties_its_collection() {
        let db = db_with_two_clean_graphs().await;
        let memory = db.memory_for(nodedb_mem::EngineId::Graph);

        crate::query::graph_ops::labels::set_node_labels(
            &db.query_engine.csr,
            &memory,
            GRAPH_B,
            "a",
            &["Person".to_string()],
        )
        .expect("set_node_labels");

        assert!(db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, GRAPH_B));
        assert!(!db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, GRAPH_A));
    }
}
