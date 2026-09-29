// SPDX-License-Identifier: Apache-2.0

//! Serialize graph and vector index persistence jobs.

use super::super::types::{META_CSR_COLLECTIONS, META_HNSW_COLLECTIONS};
use crate::{
    nodedb::{
        NodeDbLite,
        flush_gens::{ArtifactFlush, FlushArtifact, ID_MAP_KEY},
    },
    storage::engine::{StorageEngine, WriteOp},
};
use nodedb_types::{
    Namespace,
    error::{NodeDbError, NodeDbResult},
};

pub(super) struct IndexJobs {
    /// CSR graph segments due a write, each with the generation planned for it.
    #[cfg(not(target_arch = "wasm32"))]
    pub csr: Vec<(ArtifactFlush, Vec<u8>)>,
    /// Vector segments due a write, each with the generation planned for it.
    #[cfg(not(target_arch = "wasm32"))]
    pub vectors: Vec<(
        ArtifactFlush,
        Vec<crate::engine::vector::segment::NodeSource>,
    )>,
    /// HNSW graph blobs and the id-map queued in the batch, recorded as
    /// flushed once the batch commits.
    pub batched: Vec<ArtifactFlush>,
}

impl<S: StorageEngine> NodeDbLite<S> {
    /// Queue index persistence into `ops`.
    ///
    /// The HNSW graph blobs, the id-map, and the vector segments are queued
    /// only when dirty, unless `full`. Each generation is captured under the
    /// lock the artifact is serialized under, so a mutation after it leaves
    /// the artifact dirty for the next flush. A meta entry is queued only when
    /// its value changed, and is pushed to `meta_writes` to be recorded once
    /// the batch commits.
    pub(super) fn stage_indexes(
        &self,
        ops: &mut Vec<WriteOp>,
        full: bool,
        meta_writes: &mut Vec<(&'static [u8], Vec<u8>)>,
    ) -> NodeDbResult<IndexJobs> {
        let mut batched: Vec<ArtifactFlush> = Vec::new();
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
        #[cfg(not(target_arch = "wasm32"))]
        let csr_segment_data;
        {
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

            // Segments written after the batch, each marked flushed once its
            // own write succeeds. Mutated only via the native segment-ext
            // path, compiled out on wasm32.
            #[cfg(not(target_arch = "wasm32"))]
            let mut segment_data: Vec<(ArtifactFlush, Vec<u8>)> = Vec::new();
            for (name, index) in entries {
                let Some(planned) = self.flush_gens.plan(FlushArtifact::CsrGraph, name, full)
                else {
                    continue;
                };
                let checkpoint = index
                    .checkpoint_to_bytes()
                    .map_err(|e| NodeDbError::serialization("csr-checkpoint", e))?;
                #[cfg(not(target_arch = "wasm32"))]
                if graph_seg_ext.is_some() {
                    segment_data.push((planned, checkpoint));
                } else {
                    ops.push(WriteOp::Put {
                        ns: Namespace::Graph,
                        key: format!("csr:{name}").into_bytes(),
                        value: crate::storage::checksum::wrap(&checkpoint),
                    });
                    batched.push(planned);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    ops.push(WriteOp::Put {
                        ns: Namespace::Graph,
                        key: format!("csr:{name}").into_bytes(),
                        value: crate::storage::checksum::wrap(&checkpoint),
                    });
                    batched.push(planned);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                csr_segment_data = segment_data;
            }
        }

        // ── Persist HNSW vector_id_map ──
        // The id_map is serialized as one MessagePack blob of
        // `("{index_key}:{node}", doc_id, node)` entries. It must be written before any restart
        // so that vector_search can return real doc_ids (not HNSW integer strings)
        // instead of an empty id_map after restart.
        // Durable rows persist on insertion. The derived ID map persists during flush.
        //
        // Written only when dirty. The generation is captured under the same
        // lock the entries are read under, so a bind made after this point
        // leaves the id-map dirty for the next flush.
        {
            let id_map = self.vector_state.vector_id_map.lock_or_recover();
            if let Some(planned) = self
                .flush_gens
                .plan(FlushArtifact::HnswIdMap, ID_MAP_KEY, full)
            {
                let entries = id_map.to_entries();
                let bytes = zerompk::to_msgpack_vec(&entries)
                    .map_err(|e| NodeDbError::serialization("vector-id-map", e))?;
                ops.push(WriteOp::Put {
                    ns: Namespace::Vector,
                    key: b"hnsw_id_map".to_vec(),
                    value: crate::storage::checksum::wrap(&bytes),
                });
                batched.push(planned);
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
        // Per segment-backed index: where each node's segment vector comes
        // from, in node-id order, read under the index lock.
        #[cfg(not(target_arch = "wasm32"))]
        let mut segment_jobs: Vec<(
            ArtifactFlush,
            Vec<crate::engine::vector::segment::NodeSource>,
        )> = Vec::new();
        {
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

            for (name, index) in indices.iter() {
                let key = format!("hnsw:{name}");
                let graph_plan = self.flush_gens.plan(FlushArtifact::HnswGraph, name, full);

                #[cfg(not(target_arch = "wasm32"))]
                {
                    if seg_ext.is_some() {
                        // The segment's slots follow the graph's node order, so
                        // a graph that changed makes its segment due too. A
                        // durable row change marks the segment dirty itself.
                        let segment_plan = self.flush_gens.plan(
                            FlushArtifact::VectorSegment,
                            name,
                            full || graph_plan.is_some(),
                        );
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
                            batched.push(planned);
                        }
                        if let Some(planned) = segment_plan {
                            // The segment payload is laid out in node-id order. Bound
                            // nodes read the DURABLE vectors after this lock is
                            // released — see `engine::vector::segment::segment_payload`.
                            let id_map = self.vector_state.vector_id_map.lock_or_recover();
                            let sources = crate::engine::vector::segment::node_sources(index, id_map.index(name))
                                .ok_or_else(|| NodeDbError::from(crate::error::LiteError::Corrupted {
                                    detail: format!("vector index '{name}' has an unreadable unbound node: rebuild the index"),
                                }))?;
                            segment_jobs.push((planned, sources));
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
                        batched.push(planned);
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
                    batched.push(planned);
                }
            }
        }

        Ok(IndexJobs {
            #[cfg(not(target_arch = "wasm32"))]
            csr: csr_segment_data,
            #[cfg(not(target_arch = "wasm32"))]
            vectors: segment_jobs,
            batched,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{PagedbStorageMem, config::LiteConfig, engine::vector::HnswIndex};
    use nodedb_client::NodeDb;

    #[tokio::test]
    async fn flush_rejects_unreadable_unbound_graph_nodes() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let db = NodeDbLite::open_with_config(
            storage,
            LiteConfig {
                auto_flush_ms: 0,
                auto_compact_ms: 0,
                ..LiteConfig::default()
            },
        )
        .await
        .unwrap();
        db.vector_insert("vectors", "bound", &[1.0, 2.0], None)
            .await
            .unwrap();
        {
            let mut indices = db.vector_state.hnsw_indices.lock_or_recover();
            let index = indices.get("vectors").unwrap();
            let checkpoint = index.graph_checkpoint_to_bytes().unwrap();
            let restored = HnswIndex::from_checkpoint(&checkpoint).unwrap().unwrap();
            indices.insert("vectors".into(), restored);
        }
        db.vector_state
            .vector_id_map
            .lock_or_recover()
            .remove_index("vectors");
        let error = db.flush().await.unwrap_err();
        assert!(error.to_string().contains("unreadable unbound node"));
    }
}
