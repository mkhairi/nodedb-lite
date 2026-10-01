// SPDX-License-Identifier: Apache-2.0

//! Combined document and vector source mutations.

use nodedb_types::document::Document;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::document::history::ops::{is_bitemporal, versioned_put};
use crate::nodedb::LockExt;
use crate::nodedb::NodeDbLite;
use crate::nodedb::convert::{document_to_msgpack, value_to_loro};
use crate::runtime::{monotonic_millis_i64, now_millis_i64};
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Upsert a document and insert its embedding vector under one CRDT lock.
    ///
    /// Performs two logical writes in one `batch_write` call: the document is a
    /// full-row upsert, the vector metadata a field merge into the row with the
    /// vector's id. When both name the same row, the document fields survive
    /// the vector write. Re-inserting an id replaces its vector. The HNSW
    /// insert and sidecar encoding run after the CRDT lock is released.
    ///
    /// `embedding` being empty is a no-op for the vector path; the document write
    /// proceeds normally.
    pub(super) async fn document_put_with_vector_impl(
        &self,
        doc_collection: &str,
        doc: Document,
        vector_collection: &str,
        id: &str,
        embedding: &[f32],
    ) -> NodeDbResult<()> {
        let guard = self.fts_state.admit_mutation().await;
        let result = async {
            use crate::engine::crdt::{CrdtRowOp, CrdtRowWrite};
            use crate::engine::vector::nodes::{encode_sidecar, upsert_node};
            use crate::engine::vector::resident::check_insert_widths;
            use crate::engine::vector::row::EMBEDDING_DIM_FIELD;

            let doc_id = if doc.id.is_empty() {
                nodedb_types::id_gen::uuid_v7()
            } else {
                doc.id.clone()
            };

            // Build field slices for both ops before acquiring the lock.
            let doc_fields: Vec<(&str, loro::LoroValue)> = doc
                .fields
                .iter()
                .map(|(k, v)| (k.as_str(), value_to_loro(v)))
                .collect();

            let vec_meta_field = loro::LoroValue::I64(embedding.len() as i64);
            let vec_fields: Vec<(&str, loro::LoroValue)> = if !embedding.is_empty() {
                vec![(EMBEDDING_DIM_FIELD, vec_meta_field)]
            } else {
                vec![]
            };

            let sync_doc = self.should_sync_doc(doc_collection, &doc.fields);

            let bitemporal = is_bitemporal(&*self.storage, doc_collection)
                .await
                .map_err(NodeDbError::storage)?;
            let _index_build = self.hold_bitemporal_build(bitemporal).await;
            self.query_engine.indexes.revive(doc_collection, &doc_id);

            // One CRDT lock — one batch_write — one delta per row.
            {
                let mut crdt = self.crdt.lock_or_recover();
                let mut ops: Vec<CrdtRowOp<'_>> = Vec::with_capacity(2);
                ops.push((
                    CrdtRowWrite::Upsert,
                    doc_collection,
                    doc_id.as_str(),
                    doc_fields.as_slice(),
                ));
                if !embedding.is_empty() {
                    ops.push((
                        CrdtRowWrite::SetFields,
                        vector_collection,
                        id,
                        vec_fields.as_slice(),
                    ));
                }
                let mutation_ids = crdt.batch_write(&ops).map_err(NodeDbError::from)?;
                // Keep local-only documents out of the outbound CRDT delta stream.
                if !sync_doc {
                    for mutation_id in mutation_ids {
                        crdt.drop_pending(mutation_id);
                    }
                }
            }

            // For bitemporal collections, record versioned history (outside the CRDT lock).
            if bitemporal {
                let now_ms = monotonic_millis_i64();
                let body = document_to_msgpack(&doc);
                versioned_put(
                    &*self.storage,
                    doc_collection,
                    &doc_id,
                    &body,
                    now_ms,
                    // See note above: monotonic system-time key, wall-clock valid_from.
                    Some(now_millis_i64()),
                    None,
                )
                .await
                .map_err(NodeDbError::storage)?;
            }
            // Make the vector durable in the SAME write that makes the document
            // durable. Before this, a vector lived only in the in-memory HNSW
            // until some later flush wrote the segment, so an acknowledged write
            // could still lose its vector on an unclean exit — and the segment
            // being the only copy meant an unreadable one was unrecoverable.
            // Written BEFORE the in-memory index below so the durable row can
            // never be the thing that is missing after a crash.
            if !embedding.is_empty() {
                // A vector of another width than the loaded index is refused
                // before its durable row is written.
                check_insert_widths(&self.vector_state, vector_collection, [embedding.len()])
                    .await
                    .map_err(NodeDbError::from)?;
                let op = crate::engine::vector::durable::put_op(vector_collection, id, embedding);
                self.storage
                    .batch_write(std::slice::from_ref(&op))
                    .await
                    .map_err(NodeDbError::storage)?;
            }

            self.index_document_text(doc_collection, &doc_id, &doc.fields)?;
            self.index_document_sparse(doc_collection, &doc_id, &doc.fields);

            // HNSW insert (no CRDT lock needed — vector_state uses its own locks).
            if !embedding.is_empty() {
                // Replaces the node `id` held before, so the id stays one node.
                let internal_id = upsert_node(&self.vector_state, vector_collection, id, embedding)
                    .await
                    .map_err(NodeDbError::from)?;
                encode_sidecar(
                    &self.vector_state,
                    vector_collection,
                    internal_id,
                    embedding,
                )
                .map_err(|e| NodeDbError::bad_request(e.to_string()))?;

                #[cfg(not(target_arch = "wasm32"))]
                if sync_doc && let Some(q) = &self.vector_outbound {
                    crate::sync::reconcile_outbound_enqueue(
                        q.enqueue_insert(
                            vector_collection,
                            id,
                            embedding.to_vec(),
                            embedding.len(),
                            "",
                        )
                        .await,
                        "vector insert (with document)",
                        vector_collection,
                        id,
                    )
                    .map_err(NodeDbError::storage)?;
                }

                self.update_memory_stats();
            }

            Ok(())
        }
        .await;
        guard.finish(result)
    }
}
