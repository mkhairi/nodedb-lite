//! Batch operations and memory eviction for NodeDbLite.

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::vector::nodes::{bind_node, encode_sidecar};
use crate::engine::vector::resident::{check_insert_widths, lock_resident_or_create};
use crate::engine::vector::row::EMBEDDING_DIM_FIELD;

use super::{LockExt, NodeDbLite};
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Batch insert vectors — O(1) CRDT delta export instead of O(N).
    ///
    /// Use this for bulk loading (cold-start hydration, benchmark setup, imports).
    /// Each vector is inserted into HNSW and tracked in the ID map. An id
    /// repeated in the batch, or already indexed, keeps one node: the latest
    /// vector. Each id's embedding dimension merges into its CRDT row, whose
    /// other fields are kept.
    pub async fn batch_vector_insert(
        &self,
        collection: &str,
        vectors: &[(&str, &[f32])],
    ) -> NodeDbResult<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        if self.governor.worst_engine_pressure() == nodedb_mem::PressureLevel::Emergency {
            return Err(NodeDbError::storage(
                crate::error::LiteError::Backpressure {
                    detail:
                        "batch vector insert rejected: memory governor is at Emergency pressure"
                            .into(),
                },
            ));
        }

        let dim = vectors[0].1.len();

        // Every vector must fit the collection's index, loaded back if it was
        // evicted, or the batch's first vector when there is none; a refusal
        // writes nothing.
        check_insert_widths(
            &self.vector_state,
            collection,
            vectors.iter().map(|(_, e)| e.len()),
        )
        .await
        .map_err(NodeDbError::from)?;

        // Durable rows first, in ONE batch write. They are the source of truth
        // the in-memory index and the pagedb segment are both derived from: a
        // flush that finds no durable row for a collection writes no segment for
        // it, so skipping this would make a bulk load non-durable.
        {
            let ops: Vec<_> = vectors
                .iter()
                .filter(|(_, embedding)| !embedding.is_empty())
                .map(|(id, embedding)| {
                    crate::engine::vector::durable::put_op(collection, id, embedding)
                })
                .collect();
            if !ops.is_empty() {
                self.storage
                    .batch_write(&ops)
                    .await
                    .map_err(NodeDbError::storage)?;
            }
        }

        // Each id's final node, for the codec sidecar: a node an id held
        // earlier in the batch is already tombstoned.
        let mut bound: std::collections::HashMap<&str, (u32, &[f32])> =
            std::collections::HashMap::with_capacity(vectors.len());
        {
            let mut resident = lock_resident_or_create(&self.vector_state, collection, dim)
                .await
                .map_err(NodeDbError::from)?;
            let index = resident.index();
            for &(id, embedding) in vectors {
                let node = bind_node(
                    &self.vector_state,
                    index,
                    collection,
                    id,
                    embedding.to_vec(),
                )
                .map_err(NodeDbError::from)?;
                bound.insert(id, (node, embedding));
            }
        }
        // Encoded after the index lock is released: installing a sidecar
        // trains on the index.
        for (node, embedding) in bound.into_values() {
            encode_sidecar(&self.vector_state, collection, node, embedding)
                .map_err(|e| NodeDbError::bad_request(e.to_string()))?;
        }

        {
            let mut crdt = self.crdt.lock_or_recover();

            use crate::engine::crdt::{CrdtField, CrdtRowOp, CrdtRowWrite};

            let fields: Vec<Vec<CrdtField<'_>>> = vectors
                .iter()
                .map(|&(_, emb)| {
                    vec![(EMBEDDING_DIM_FIELD, loro::LoroValue::I64(emb.len() as i64))]
                })
                .collect();

            // A merge: each vector attaches to its row and keeps its fields.
            let ops: Vec<CrdtRowOp<'_>> = vectors
                .iter()
                .zip(fields.iter())
                .map(|(&(id, _), f)| (CrdtRowWrite::SetFields, collection, id, f.as_slice()))
                .collect();

            crdt.batch_write(&ops).map_err(NodeDbError::storage)?;
        }

        self.update_memory_stats();
        Ok(())
    }

    /// Batch insert graph edges into a named collection — O(1) CRDT delta
    /// export instead of O(N). Edges are isolated to `collection`.
    pub fn batch_graph_insert_edges(
        &self,
        collection: &str,
        edges: &[(&str, &str, &str)],
    ) -> NodeDbResult<()> {
        if edges.is_empty() {
            return Ok(());
        }

        {
            let memory = self.memory_for(nodedb_mem::EngineId::Graph);
            let mut csr_map = self.csr.lock_or_recover();
            let csr = csr_map
                .entry(collection.to_string())
                .or_insert_with(|| crate::engine::graph::index::CsrIndex::new(memory));
            for &(src, dst, label) in edges {
                let _ = csr.add_edge(src, label, dst);
            }
        }

        {
            let mut crdt = self.crdt.lock_or_recover();

            use crate::engine::crdt::engine::{CrdtBatchOp, CrdtField};
            let edge_coll = format!("__edges__{collection}");

            let ops: Vec<(String, Vec<CrdtField<'_>>)> = edges
                .iter()
                .map(|&(src, dst, label)| {
                    let edge_id = format!("{src}--{label}-->{dst}");
                    let fields: Vec<CrdtField<'_>> = vec![
                        ("src", loro::LoroValue::String(src.into())),
                        ("dst", loro::LoroValue::String(dst.into())),
                        ("label", loro::LoroValue::String(label.into())),
                    ];
                    (edge_id, fields)
                })
                .collect();

            let refs: Vec<CrdtBatchOp<'_>> = ops
                .iter()
                .map(|(id, fields)| (edge_coll.as_str(), id.as_str(), fields.as_slice()))
                .collect();

            crdt.batch_upsert(&refs).map_err(NodeDbError::storage)?;
        }

        self.update_memory_stats();
        Ok(())
    }

    /// Compact all per-collection CSR graph indices (merge buffer into dense arrays).
    pub fn compact_graph(&self) -> NodeDbResult<()> {
        let mut csr_map = self.csr.lock_or_recover();
        for (name, csr) in csr_map.iter_mut() {
            csr.compact().map_err(|e| {
                NodeDbError::storage(format!("graph csr compact failed for '{name}': {e}"))
            })?;
        }
        Ok(())
    }

    /// Evict HNSW collections to reduce memory usage.
    ///
    /// Persists each evicted collection to storage first, then drops
    /// it from memory. Collections are evicted smallest-first.
    pub async fn evict_collections(&self, max_to_evict: usize) -> NodeDbResult<usize> {
        let mut evicted = 0;

        let candidates: Vec<(String, usize)> = {
            let indices = self.vector_state.hnsw_indices.lock_or_recover();
            let mut sorted: Vec<(String, usize)> = indices
                .iter()
                .map(|(name, idx)| (name.clone(), idx.len()))
                .collect();
            sorted.sort_by_key(|(_, size)| *size);
            sorted
        };

        // Check once whether the pagedb segment path is available.
        #[cfg(not(target_arch = "wasm32"))]
        let seg_ext = self.storage.as_vector_segment_ext();

        for (name, _) in candidates.into_iter().take(max_to_evict) {
            // Snapshot checkpoint while holding the lock.
            // On native with segment support: graph-only bytes, and the segment
            // payload is read from the DURABLE vectors after the lock is
            // released — never from the in-memory index, which carries empty
            // vector slots whenever it was itself restored from a graph-only
            // checkpoint.
            // Otherwise: full checkpoint blob (WASM and non-pagedb native backends).
            //
            // `snapshot` is the index's (node count, tombstone count) at the
            // checkpoint. A write between the checkpoint and the removal
            // changes it, and the index then stays resident: dropping it would
            // lose that write from the stored copy.
            #[cfg(not(target_arch = "wasm32"))]
            let (blob, segment_sources, snapshot) = {
                let indices = self.vector_state.hnsw_indices.lock_or_recover();
                match indices.get(&name) {
                    Some(idx) => {
                        let snapshot = (idx.len(), idx.tombstone_count());
                        if seg_ext.is_some() {
                            let graph_bytes = idx.graph_checkpoint_to_bytes().map_err(|e| {
                                NodeDbError::serialization("hnsw-graph-checkpoint", e)
                            })?;
                            // Segment slots follow node-id order; see
                            // `engine::vector::segment`.
                            let id_map = self.vector_state.vector_id_map.lock_or_recover();
                            let sources = crate::engine::vector::segment::node_sources(
                                idx,
                                id_map.index(&name),
                            );
                            (graph_bytes, sources, snapshot)
                        } else {
                            let blob = idx
                                .checkpoint_to_bytes()
                                .map_err(|e| NodeDbError::serialization("hnsw-checkpoint", e))?;
                            (blob, None, snapshot)
                        }
                    }
                    None => continue,
                }
            };
            #[cfg(target_arch = "wasm32")]
            let (blob, snapshot) = {
                let indices = self.vector_state.hnsw_indices.lock_or_recover();
                match indices.get(&name) {
                    Some(idx) => (
                        idx.checkpoint_to_bytes()
                            .map_err(|e| NodeDbError::serialization("hnsw-checkpoint", e))?,
                        (idx.len(), idx.tombstone_count()),
                    ),
                    None => continue,
                }
            };

            let key = format!("hnsw:{name}");
            self.storage
                .put(
                    Namespace::Vector,
                    key.as_bytes(),
                    &crate::storage::checksum::wrap(&blob),
                )
                .await
                .map_err(NodeDbError::storage)?;

            // Write vector segment on native targets when segment ext is available.
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(sources) = segment_sources
                && let Some(ext) = seg_ext
            {
                match crate::engine::vector::segment::segment_payload(
                    &*self.storage,
                    &name,
                    sources,
                )
                .await
                {
                    Ok(Some((dim, vectors, surrogates))) => {
                        if let Err(e) = ext
                            .write_vector_segment(&name, dim, &vectors, &surrogates)
                            .await
                        {
                            tracing::error!(
                                collection = %name,
                                error = %e,
                                "HNSW vector segment write failed during eviction; \
                                 graph blob is persisted but vectors may be lost on cold restart"
                            );
                        }
                    }
                    // No durable vectors: leave any existing segment untouched.
                    // Replacing it with an empty one is the corruption this avoids.
                    Ok(None) => {}
                    Err(e) => tracing::error!(
                        collection = %name,
                        error = %e,
                        "reading durable vectors for the eviction segment write failed; \
                         leaving the existing segment in place"
                    ),
                }
            }

            // Mark and drop under one lock, so a loader never sees the index
            // gone without the mark. The stored checkpoint is valid, so an
            // earlier unloadable verdict no longer holds.
            {
                let mut indices = self.vector_state.hnsw_indices.lock_or_recover();
                let unchanged = indices
                    .get(&name)
                    .is_some_and(|idx| (idx.len(), idx.tombstone_count()) == snapshot);
                if !unchanged {
                    tracing::debug!(
                        collection = %name,
                        "HNSW collection changed during eviction; kept in memory"
                    );
                    continue;
                }
                self.vector_state
                    .evicted
                    .lock_or_recover()
                    .insert(name.clone());
                self.vector_state.unloadable.lock_or_recover().remove(&name);
                indices.remove(&name);
            }

            tracing::info!(collection = %name, "HNSW collection evicted from memory");
            evicted += 1;
        }

        self.update_memory_stats();
        Ok(evicted)
    }

    /// Check memory pressure and evict if needed.
    ///
    /// Matches the old thresholds: nodedb_mem's Critical (85-95%) is where
    /// Lite's own Warning used to start, and Emergency (>95%) is where
    /// Lite's own Critical used to start. The new Warning tier (70-85%) is
    /// below both old thresholds, so it evicts nothing, same as Normal.
    pub async fn check_and_evict(&self) -> NodeDbResult<usize> {
        use nodedb_mem::PressureLevel;

        self.update_memory_stats();
        match self.governor.worst_engine_pressure() {
            PressureLevel::Emergency => self.evict_collections(2).await,
            PressureLevel::Critical => self.evict_collections(1).await,
            PressureLevel::Warning => Ok(0),
            PressureLevel::Normal => Ok(0),
        }
    }
}
