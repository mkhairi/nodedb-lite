// SPDX-License-Identifier: Apache-2.0

//! Graph operation dispatcher for the Lite physical visitor.
//!
//! Exhaustively matches all 21 `GraphOp` variants. `RagFusion` and `Match`
//! are wired to their writer-2 placeholder stubs. `MatchContinuation`,
//! `MatchVarLenResume`, `BspSuperstep`, and `WccSuperstep` are cross-shard
//! distributed primitives with no single-node equivalent and return
//! `LiteError::Unsupported`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use nodedb_physical::physical_plan::GraphOp;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::graph_ops::{
    algorithms, edges, fusion, labels, match_engine, stats, temporal, traversal,
};
use crate::storage::engine::StorageEngine;

use super::graph_resolve::resolve_collection_for_nodes;
use super::policy::{deny_policy, deny_write_check};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type GraphFut<'a> =
    Pin<Box<dyn Future<Output = Result<QueryResult, LiteError>> + Send + 'a>>;

#[cfg(target_arch = "wasm32")]
pub(crate) type GraphFut<'a> = Pin<Box<dyn Future<Output = Result<QueryResult, LiteError>> + 'a>>;

/// Dispatch a `GraphOp` to the correct Lite handler.
pub(crate) fn dispatch<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    op: &GraphOp,
) -> Result<GraphFut<'a>, LiteError> {
    let fut: GraphFut<'a> = match op {
        GraphOp::EdgePut {
            collection,
            src_id,
            label,
            dst_id,
            properties,
            ..
        } => {
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let collection = collection.to_string();
            let src_id = src_id.clone();
            let label = label.clone();
            let dst_id = dst_id.clone();
            let properties = properties.clone();
            Box::pin(async move {
                edges::edge_put(
                    &storage,
                    &csr_map,
                    &collection,
                    &src_id,
                    &label,
                    &dst_id,
                    &properties,
                )
                .await
            })
        }

        GraphOp::EdgePutBatch { edges: batch_edges } => {
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let batch_edges = batch_edges.clone();
            Box::pin(async move { edges::edge_put_batch(&storage, &csr_map, &batch_edges).await })
        }

        GraphOp::EdgeDelete {
            collection,
            src_id,
            label,
            dst_id,
            rls_write_check,
            ..
        } => {
            deny_write_check("GraphOp::EdgeDelete", &[rls_write_check])?;
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let collection = collection.to_string();
            let src_id = src_id.clone();
            let label = label.clone();
            let dst_id = dst_id.clone();
            Box::pin(async move {
                edges::edge_delete(&storage, &csr_map, &collection, &src_id, &label, &dst_id).await
            })
        }

        GraphOp::EdgeDeleteBatch { edges: batch_edges } => {
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let batch_edges = batch_edges.clone();
            Box::pin(
                async move { edges::edge_delete_batch(&storage, &csr_map, &batch_edges).await },
            )
        }

        GraphOp::Hop {
            start_nodes,
            edge_label,
            direction,
            depth,
            options,
            frontier_bitmap,
            rls_filters,
            ..
        } => {
            deny_policy("GraphOp::Hop", None, &[rls_filters.as_slice()])?;
            let csr_map = engine.csr.clone();
            // Hop is scoped to a single collection; collection is implicit in Lite
            // as all edges share the same CSR map keyed by collection. The caller
            // must pass start_nodes that are collection-scoped. We use a default
            // sentinel to indicate "traverse the first collection" — but in practice
            // the collection is embedded in the node keys when the caller is the
            // Origin SQL planner. For Lite, use a special lookup in the first key
            // found in start_nodes against the CSR map.
            //
            // Because `GraphOp::Hop` carries no explicit collection field, Lite
            // resolves the collection by iterating csr_map entries for the first
            // collection that contains any of the start nodes.
            let start_nodes = start_nodes.clone();
            let edge_label = edge_label.clone();
            let direction = *direction;
            let depth = *depth;
            let options = options.clone();
            let frontier_bitmap = frontier_bitmap.clone();
            Box::pin(async move {
                // Resolve collection from csr_map.
                let collection = resolve_collection_for_nodes(&csr_map, &start_nodes);
                traversal::hop(
                    &csr_map,
                    &collection,
                    &start_nodes,
                    edge_label.as_deref(),
                    direction,
                    depth,
                    &options,
                    frontier_bitmap.as_ref(),
                )
            })
        }

        GraphOp::Neighbors {
            node_id,
            edge_label,
            direction,
            rls_filters,
            ..
        } => {
            deny_policy("GraphOp::Neighbors", None, &[rls_filters.as_slice()])?;
            let csr_map = engine.csr.clone();
            let node_id = node_id.clone();
            let edge_label = edge_label.clone();
            let direction = *direction;
            Box::pin(async move {
                let collection =
                    resolve_collection_for_nodes(&csr_map, std::slice::from_ref(&node_id));
                traversal::neighbors(
                    &csr_map,
                    &collection,
                    &node_id,
                    edge_label.as_deref(),
                    direction,
                )
            })
        }

        GraphOp::NeighborsMulti {
            node_ids,
            edge_label,
            direction,
            max_results,
            rls_filters,
            ..
        } => {
            deny_policy("GraphOp::NeighborsMulti", None, &[rls_filters.as_slice()])?;
            let csr_map = engine.csr.clone();
            let node_ids = node_ids.clone();
            let edge_label = edge_label.clone();
            let direction = *direction;
            let max_results = *max_results;
            Box::pin(async move {
                let collection = resolve_collection_for_nodes(&csr_map, &node_ids);
                traversal::neighbors_multi(
                    &csr_map,
                    &collection,
                    &node_ids,
                    edge_label.as_deref(),
                    direction,
                    max_results,
                )
            })
        }

        GraphOp::Path {
            src,
            dst,
            edge_label,
            max_depth,
            options,
            frontier_bitmap,
            rls_filters,
            ..
        } => {
            deny_policy("GraphOp::Path", None, &[rls_filters.as_slice()])?;
            let csr_map = engine.csr.clone();
            let src = src.clone();
            let dst = dst.clone();
            let edge_label = edge_label.clone();
            let max_depth = *max_depth;
            let options = options.clone();
            let frontier_bitmap = frontier_bitmap.clone();
            Box::pin(async move {
                let collection =
                    resolve_collection_for_nodes(&csr_map, &[src.clone(), dst.clone()]);
                traversal::path(
                    &csr_map,
                    &collection,
                    &src,
                    &dst,
                    edge_label.as_deref(),
                    max_depth,
                    &options,
                    frontier_bitmap.as_ref(),
                )
            })
        }

        GraphOp::Subgraph {
            start_nodes,
            edge_label,
            depth,
            options,
            rls_filters,
            ..
        } => {
            deny_policy("GraphOp::Subgraph", None, &[rls_filters.as_slice()])?;
            let csr_map = engine.csr.clone();
            let start_nodes = start_nodes.clone();
            let edge_label = edge_label.clone();
            let depth = *depth;
            let options = options.clone();
            Box::pin(async move {
                let collection = resolve_collection_for_nodes(&csr_map, &start_nodes);
                traversal::subgraph(
                    &csr_map,
                    &collection,
                    &start_nodes,
                    edge_label.as_deref(),
                    depth,
                    &options,
                )
            })
        }

        GraphOp::Algo { algorithm, params } => {
            let csr_map = engine.csr.clone();
            let algorithm = *algorithm;
            let params = params.clone();
            Box::pin(async move { algorithms::run_algo(&csr_map, algorithm, &params) })
        }

        GraphOp::SetNodeLabels { node_id, labels } => {
            let csr_map = engine.csr.clone();
            let node_id = node_id.clone();
            let labels = labels.clone();
            Box::pin(async move {
                // SetNodeLabels carries no collection field; resolve via node presence.
                let collection =
                    resolve_collection_for_nodes(&csr_map, std::slice::from_ref(&node_id));
                labels::set_node_labels(&csr_map, &collection, &node_id, &labels)
            })
        }

        GraphOp::RemoveNodeLabels { node_id, labels } => {
            let csr_map = engine.csr.clone();
            let node_id = node_id.clone();
            let labels = labels.clone();
            Box::pin(async move {
                let collection =
                    resolve_collection_for_nodes(&csr_map, std::slice::from_ref(&node_id));
                labels::remove_node_labels(&csr_map, &collection, &node_id, &labels)
            })
        }

        GraphOp::TemporalNeighbors {
            collection,
            node_id,
            edge_label,
            direction,
            system_time,
            valid_at_ms,
            rls_filters,
            ..
        } => {
            deny_policy(
                "GraphOp::TemporalNeighbors",
                None,
                &[rls_filters.as_slice()],
            )?;
            use nodedb_types::SystemTimeScope;
            // Mirror Origin: AllVersions is not supported on the graph engine.
            if system_time.is_all_versions() {
                return Err(LiteError::Unsupported {
                    detail: "AS OF SYSTEM TIME NULL (all-versions) is not supported on \
                             the graph engine in Lite"
                        .into(),
                });
            }
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let collection = collection.to_string();
            let node_id = node_id.clone();
            let edge_label = edge_label.clone();
            let direction = *direction;
            // Only an explicit `AS OF SYSTEM TIME <ts>` narrows the read; every
            // other scope (`Current`, and the all-versions case already rejected
            // above) means "no system-time filter" → read the latest version.
            let system_as_of_ms: Option<i64> = match system_time {
                SystemTimeScope::AsOf(ms) => Some(*ms),
                _ => None,
            };
            let valid_at_ms = *valid_at_ms;
            Box::pin(async move {
                temporal::temporal_neighbors(
                    &storage,
                    &csr_map,
                    &collection,
                    &node_id,
                    edge_label.as_deref(),
                    direction,
                    system_as_of_ms,
                    valid_at_ms,
                )
                .await
            })
        }

        GraphOp::TemporalAlgorithm {
            algorithm,
            params,
            system_time,
        } => {
            use nodedb_types::SystemTimeScope;
            // Mirror Origin: AllVersions is not supported on the graph engine.
            if system_time.is_all_versions() {
                return Err(LiteError::Unsupported {
                    detail: "AS OF SYSTEM TIME NULL (all-versions) is not supported on \
                             the graph engine in Lite"
                        .into(),
                });
            }
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let algorithm = *algorithm;
            let params = params.clone();
            // Only an explicit `AS OF SYSTEM TIME <ts>` narrows the read; every
            // other scope (`Current`, and the all-versions case already rejected
            // above) means "no system-time filter" → read the latest version.
            let system_as_of_ms: Option<i64> = match system_time {
                SystemTimeScope::AsOf(ms) => Some(*ms),
                _ => None,
            };
            Box::pin(async move {
                temporal::temporal_algorithm(
                    &storage,
                    &csr_map,
                    algorithm,
                    &params,
                    system_as_of_ms,
                )
                .await
            })
        }

        GraphOp::Stats { collection, as_of } => {
            let storage = engine.storage.clone();
            let csr_map = engine.csr.clone();
            let collection = collection.as_ref().map(|c| c.to_string());
            let as_of = *as_of;
            Box::pin(async move {
                stats::graph_stats(&storage, &csr_map, collection.as_deref(), as_of).await
            })
        }

        GraphOp::RagFusion {
            collection,
            query_vector,
            vector_top_k,
            edge_label,
            direction,
            expansion_depth,
            final_top_k,
            rrf_k,
            rrf_k_triple,
            vector_field,
            options: _,
            bm25_query,
            bm25_field,
        } => {
            let vector_state = Arc::clone(&engine.vector_state);
            let crdt = Arc::clone(&engine.crdt);
            let fts_state = Arc::clone(&engine.fts_state);
            let csr_map = Arc::clone(&engine.csr);
            let collection = collection.to_string();
            let query_vector = query_vector.clone();
            let vector_top_k = *vector_top_k;
            let edge_label = edge_label.clone();
            let direction = *direction;
            let expansion_depth = *expansion_depth;
            let final_top_k = *final_top_k;
            let rrf_k = *rrf_k;
            let rrf_k_triple = *rrf_k_triple;
            let vector_field = vector_field.clone();
            let bm25_query = bm25_query.clone();
            let bm25_field = bm25_field.clone();
            Box::pin(async move {
                fusion::rag_fusion(
                    &vector_state,
                    &crdt,
                    &fts_state,
                    &csr_map,
                    &collection,
                    &query_vector,
                    &vector_field,
                    vector_top_k,
                    edge_label.as_deref(),
                    direction,
                    expansion_depth,
                    final_top_k,
                    rrf_k,
                    rrf_k_triple,
                    bm25_query.as_deref(),
                    bm25_field.as_deref(),
                )
                .await
            })
        }

        // Cross-shard MATCH continuation / var-len resume and the BSP
        // superstep primitives (PageRank/WCC) exist to let a distributed
        // coordinator round-trip partial state across owning shards. Lite is
        // single-node — there are no shards to resume on or stitch together
        // — so these have no local execution path.
        GraphOp::MatchContinuation { .. } => Box::pin(async move {
            Err(LiteError::Unsupported {
                detail: "MatchContinuation is a cross-shard MATCH resume primitive; \
                         unsupported on the single-node Lite engine"
                    .into(),
            })
        }),

        GraphOp::MatchVarLenResume { .. } => Box::pin(async move {
            Err(LiteError::Unsupported {
                detail: "MatchVarLenResume is a cross-shard MATCH resume primitive; \
                         unsupported on the single-node Lite engine"
                    .into(),
            })
        }),

        GraphOp::BspSuperstep(_) => Box::pin(async move {
            Err(LiteError::Unsupported {
                detail: "BspSuperstep is a distributed PageRank BSP primitive; \
                         unsupported on the single-node Lite engine"
                    .into(),
            })
        }),

        GraphOp::WccSuperstep(_) => Box::pin(async move {
            Err(LiteError::Unsupported {
                detail: "WccSuperstep is a distributed WCC contraction primitive; \
                         unsupported on the single-node Lite engine"
                    .into(),
            })
        }),

        GraphOp::Match {
            query,
            frontier_bitmap,
            ..
        } => {
            let csr_map = Arc::clone(&engine.csr);
            let crdt = Arc::clone(&engine.crdt);
            let query = query.clone();
            let frontier_bitmap = frontier_bitmap.clone();
            Box::pin(async move {
                match_engine::graph_match(&csr_map, &query, frontier_bitmap.as_ref(), Some(&crdt))
                    .await
            })
        }

        GraphOp::ResolveEdgeDelete(_) => {
            return Err(LiteError::Unsupported {
                detail:
                    "GraphOp::ResolveEdgeDelete: the governed resolve/apply write path belongs \
                         to the Origin Control Plane and has no equivalent on the \
                         single-node Lite engine"
                        .to_string(),
            });
        }
    };

    Ok(fut)
}
