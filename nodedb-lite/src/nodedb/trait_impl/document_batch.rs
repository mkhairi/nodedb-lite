// SPDX-License-Identifier: Apache-2.0

//! Batch document + vector ingest for `NodeDbLite`.
//!
//! `document_put_with_vector_batch_impl` takes a slice of
//! `(doc_collection, doc, vector_collection, id, embedding)` items and
//! acquires the CRDT lock exactly **once** for the whole batch — that is the
//! win over the single-item path, which takes and releases the lock per item.
//!
//! The batch does **not** collapse into one delta: `CrdtEngine::batch_write`
//! emits one `PendingDelta` per CRDT row, tagged with that row's real
//! collection and document ID. A delta spanning several rows (or several
//! collections) is not independently applicable by a receiver, which commits
//! per row and stores documents per collection. So a batch of N items emits
//! N document deltas plus one vector-metadata delta per item that carries a
//! non-empty embedding. See `CrdtEngine::batch_write` for the contract.
//!
//! Each document is a full-row upsert; each vector-metadata write is a field
//! merge, so a vector attached to its own document's row keeps that
//! document's fields. An id repeated within the batch, or already indexed,
//! keeps one HNSW node: the latest vector.
//!
//! FTS indexing, bitemporal history writes, and HNSW inserts run after
//! the CRDT lock is released — matching the ordering of the single-item path.

use nodedb_types::document::Document;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::crdt::{CrdtRowOp, CrdtRowWrite};
use crate::engine::document::history::ops::{is_bitemporal, versioned_put};
use crate::engine::vector::nodes::{encode_sidecar, upsert_node};
use crate::engine::vector::resident::check_insert_widths;
use crate::engine::vector::row::EMBEDDING_DIM_FIELD;
use crate::nodedb::LockExt;
use crate::nodedb::NodeDbLite;
use crate::nodedb::convert::{document_to_msgpack, value_to_loro};
use crate::runtime::now_millis_i64;
use crate::storage::engine::StorageEngine;

/// One item in a batch ingest call.
pub struct BatchItem<'a> {
    pub doc_collection: &'a str,
    pub doc: Document,
    pub vector_collection: &'a str,
    pub id: &'a str,
    pub embedding: Option<&'a [f32]>,
}

/// A list of CRDT fields for one upsert: borrowed field name → Loro value.
type LoroFields<'a> = Vec<(&'a str, loro::LoroValue)>;

/// A batch item resolved before the CRDT lock is taken:
/// `(document id, document fields, vector-metadata fields)`.
type ResolvedBatchItem<'a> = (String, LoroFields<'a>, LoroFields<'a>);

impl<S: StorageEngine> NodeDbLite<S> {
    /// Batch upsert of documents with optional embeddings.
    ///
    /// Acquires the CRDT lock once and runs every per-item Loro mutation
    /// under that single hold. Per `CrdtEngine::batch_write`, each CRDT row
    /// still exports its own delta — one per document, plus one per item with
    /// a non-empty embedding for the vector-metadata row — because a delta
    /// covering several rows or collections is not independently applicable by
    /// the receiver.
    ///
    /// FTS indexing, bitemporal history, and HNSW inserts happen after the
    /// lock is released, in the same relative order as the single-item path.
    ///
    /// Returns the list of document IDs written, in input order.
    pub async fn document_put_with_vector_batch_impl(
        &self,
        items: &[BatchItem<'_>],
    ) -> NodeDbResult<Vec<String>> {
        let guard = self.fts_state.admit_mutation().await;
        let result = async {
            if items.is_empty() {
                return Ok(Vec::new());
            }

            // Reject the whole batch up front under critical memory pressure, matching
            // the single-item `document_put_impl` / `vector_insert_impl` guard. A batch
            // can ingest many documents plus embeddings at once, so the early gate is
            // even more important here than on the single-item path.
            if self.governor.worst_engine_pressure() == nodedb_mem::PressureLevel::Emergency {
                return Err(NodeDbError::storage(
                    crate::error::LiteError::Backpressure {
                        detail: "batch ingest rejected: memory governor is at Emergency pressure"
                            .into(),
                    },
                ));
            }

            // Every embedding must fit its loaded index, or the first embedding
            // bound for the same index when none is loaded. A refusal writes
            // nothing: the check runs before the CRDT upsert and the durable rows.
            let mut widths_by_index: std::collections::HashMap<&str, Vec<usize>> =
                std::collections::HashMap::new();
            for item in items {
                if let Some(emb) = item.embedding
                    && !emb.is_empty()
                {
                    widths_by_index
                        .entry(item.vector_collection)
                        .or_default()
                        .push(emb.len());
                }
            }
            for (index_key, widths) in &widths_by_index {
                check_insert_widths(&self.vector_state, index_key, widths.iter().copied())
                    .await
                    .map_err(NodeDbError::from)?;
            }

            // A batch may write bitemporal collections: hold off any index build
            // reading history until every item's history version is written.
            let _index_build = self.hold_bitemporal_build(true).await;

            // Pre-compute doc IDs and field vecs before taking the lock.
            let mut resolved: Vec<ResolvedBatchItem<'_>> = Vec::with_capacity(items.len());

            for item in items {
                let doc_id = if item.doc.id.is_empty() {
                    nodedb_types::id_gen::uuid_v7()
                } else {
                    item.doc.id.clone()
                };

                let doc_fields: Vec<(&str, loro::LoroValue)> = item
                    .doc
                    .fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), value_to_loro(v)))
                    .collect();

                let vec_fields: Vec<(&str, loro::LoroValue)> = match item.embedding {
                    Some(emb) if !emb.is_empty() => {
                        vec![(EMBEDDING_DIM_FIELD, loro::LoroValue::I64(emb.len() as i64))]
                    }
                    _ => vec![],
                };

                resolved.push((doc_id, doc_fields, vec_fields));
            }

            // A written document is live again, even one deleted in history.
            for (item, (doc_id, _, _)) in items.iter().zip(&resolved) {
                self.query_engine
                    .indexes
                    .revive(item.doc_collection, doc_id);
            }

            // Build the ops slice for batch_write — one CRDT lock hold, one
            // exported delta per row. Documents replace their row; vector
            // metadata merges into its row.
            {
                let mut crdt = self.crdt.lock_or_recover();

                let mut ops: Vec<CrdtRowOp<'_>> = Vec::with_capacity(items.len() * 2);
                for (item, (doc_id, doc_fields, vec_fields)) in items.iter().zip(&resolved) {
                    ops.push((
                        CrdtRowWrite::Upsert,
                        item.doc_collection,
                        doc_id.as_str(),
                        doc_fields.as_slice(),
                    ));
                    if !vec_fields.is_empty() {
                        ops.push((
                            CrdtRowWrite::SetFields,
                            item.vector_collection,
                            item.id,
                            vec_fields.as_slice(),
                        ));
                    }
                }

                crdt.batch_write(&ops).map_err(NodeDbError::from)?;
            }

            // Post-lock work: bitemporal history + FTS + HNSW (matches single-item ordering).
            let now_ms = now_millis_i64();

            for (item, (doc_id, _, _)) in items.iter().zip(&resolved) {
                if is_bitemporal(&*self.storage, item.doc_collection)
                    .await
                    .map_err(NodeDbError::storage)?
                {
                    let body = document_to_msgpack(&item.doc);
                    versioned_put(
                        &*self.storage,
                        item.doc_collection,
                        doc_id,
                        &body,
                        now_ms,
                        None,
                        None,
                    )
                    .await
                    .map_err(NodeDbError::storage)?;
                }
                // Same durability contract as the single-document path: the
                // vector is persisted in the same write that makes the document
                // durable, so the in-memory HNSW below is a derived index rather
                // than the only copy.
                if let Some(embedding) = item.embedding
                    && !embedding.is_empty()
                {
                    let op = crate::engine::vector::durable::put_op(
                        item.vector_collection,
                        item.id,
                        embedding,
                    );
                    self.storage
                        .batch_write(std::slice::from_ref(&op))
                        .await
                        .map_err(NodeDbError::storage)?;
                }

                self.index_document_text(item.doc_collection, doc_id, &item.doc.fields)?;
                self.index_document_sparse(item.doc_collection, doc_id, &item.doc.fields);

                if let Some(embedding) = item.embedding
                    && !embedding.is_empty()
                {
                    // Replaces the node the id held before, including one bound
                    // earlier in this batch, so the id stays one node.
                    let internal_id = upsert_node(
                        &self.vector_state,
                        item.vector_collection,
                        item.id,
                        embedding,
                    )
                    .await
                    .map_err(NodeDbError::from)?;
                    encode_sidecar(
                        &self.vector_state,
                        item.vector_collection,
                        internal_id,
                        embedding,
                    )
                    .map_err(|e| NodeDbError::bad_request(e.to_string()))?;

                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(q) = &self.vector_outbound {
                        crate::sync::reconcile_outbound_enqueue(
                            q.enqueue_insert(
                                item.vector_collection,
                                item.id,
                                embedding.to_vec(),
                                embedding.len(),
                                "",
                            )
                            .await,
                            "vector insert (batch)",
                            item.vector_collection,
                            item.id,
                        )
                        .map_err(nodedb_types::error::NodeDbError::storage)?;
                    }
                }
            }

            if items
                .iter()
                .any(|it| it.embedding.is_some_and(|e| !e.is_empty()))
            {
                self.update_memory_stats();
            }

            Ok(resolved.into_iter().map(|(id, _, _)| id).collect())
        }
        .await;
        guard.finish(result)
    }
}
