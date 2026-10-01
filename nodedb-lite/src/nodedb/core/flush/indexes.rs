// SPDX-License-Identifier: Apache-2.0

//! Serialize graph and vector index persistence jobs.

use super::super::types::{META_CSR_COLLECTIONS, META_HNSW_COLLECTIONS};
use crate::{
    nodedb::{LockExt, NodeDbLite},
    storage::engine::{StorageEngine, WriteOp},
};
use nodedb_types::{
    Namespace,
    error::{NodeDbError, NodeDbResult},
};

pub(super) struct IndexJobs {
    #[cfg(not(target_arch = "wasm32"))]
    pub csr: Vec<(String, Vec<u8>)>,
    #[cfg(not(target_arch = "wasm32"))]
    pub vectors: Vec<(String, Vec<crate::engine::vector::segment::NodeSource>)>,
}

impl<S: StorageEngine> NodeDbLite<S> {
    pub(super) fn stage_indexes(&self, ops: &mut Vec<WriteOp>) -> NodeDbResult<IndexJobs> {
        // ── Persist per-collection CSR indices ──
        // When the pagedb segment extension is available (native PagedbStorage):
        //   - CSR blob → pagedb segment (written after batch_write)
        //   - B+ tree receives only the collection-name index (META_CSR_COLLECTIONS)
        // Otherwise (WASM or non-pagedb native backends):
        //   - CSR blob → B+ tree (Namespace::Graph, CRC32C wrapped)
        #[cfg(not(target_arch = "wasm32"))]
        let graph_seg_ext = self.storage.as_graph_segment_ext();
        #[cfg(not(target_arch = "wasm32"))]
        let csr_segment_data;
        {
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
            #[cfg(not(target_arch = "wasm32"))]
            let mut segment_data: Vec<(String, Vec<u8>)> = Vec::new();
            for (name, index) in csr_map.iter() {
                let checkpoint = index
                    .checkpoint_to_bytes()
                    .map_err(|e| NodeDbError::serialization("csr-checkpoint", e))?;
                #[cfg(not(target_arch = "wasm32"))]
                if graph_seg_ext.is_some() {
                    segment_data.push((name.clone(), checkpoint));
                } else {
                    ops.push(WriteOp::Put {
                        ns: Namespace::Graph,
                        key: format!("csr:{name}").into_bytes(),
                        value: crate::storage::checksum::wrap(&checkpoint),
                    });
                }
                #[cfg(target_arch = "wasm32")]
                ops.push(WriteOp::Put {
                    ns: Namespace::Graph,
                    key: format!("csr:{name}").into_bytes(),
                    value: crate::storage::checksum::wrap(&checkpoint),
                });
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
        {
            let entries = self
                .vector_state
                .vector_id_map
                .lock_or_recover()
                .to_entries();
            let bytes = zerompk::to_msgpack_vec(&entries)
                .map_err(|e| NodeDbError::serialization("vector-id-map", e))?;
            ops.push(WriteOp::Put {
                ns: Namespace::Vector,
                key: b"hnsw_id_map".to_vec(),
                value: crate::storage::checksum::wrap(&bytes),
            });
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
            String,
            Vec<crate::engine::vector::segment::NodeSource>,
        )> = Vec::new();
        {
            let indices = self.vector_state.hnsw_indices.lock_or_recover();
            let names: Vec<String> = indices.keys().cloned().collect();
            let names_bytes = zerompk::to_msgpack_vec(&names)
                .map_err(|e| NodeDbError::serialization("msgpack", e))?;
            ops.push(WriteOp::Put {
                ns: Namespace::Meta,
                key: META_HNSW_COLLECTIONS.to_vec(),
                value: names_bytes,
            });

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
                        // The segment payload is laid out in node-id order. Bound
                        // nodes read the DURABLE vectors after this lock is
                        // released — see `engine::vector::segment::segment_payload`.
                        let id_map = self.vector_state.vector_id_map.lock_or_recover();
                        let sources = crate::engine::vector::segment::node_sources(index, id_map.index(name))
                            .ok_or_else(|| NodeDbError::from(crate::error::LiteError::Corrupted {
                                detail: format!("vector index '{name}' has an unreadable unbound node: rebuild the index"),
                            }))?;
                        segment_jobs.push((name.clone(), sources));
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
        }

        Ok(IndexJobs {
            #[cfg(not(target_arch = "wasm32"))]
            csr: csr_segment_data,
            #[cfg(not(target_arch = "wasm32"))]
            vectors: segment_jobs,
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
