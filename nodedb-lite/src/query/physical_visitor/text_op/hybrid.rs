// SPDX-License-Identifier: Apache-2.0

//! Hybrid reads: vector + BM25 text fused via RRF, with an optional graph
//! BFS leg.

use std::sync::Arc;

use nodedb_graph::Direction;
use nodedb_graph::traversal::DEFAULT_MAX_VISITED;
use nodedb_query::fusion::{FusedResult, RankedResult, reciprocal_rank_fusion_weighted};
use nodedb_types::result::{QueryResult, SearchResult};
use nodedb_types::value::Value;

use crate::engine::fts::{TextSearchRequest, run_text_search};
use crate::engine::vector::search::run_vector_search;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::adapter::LitePhysicalFut;
use super::search::{decode_rls_filter, or_params};

/// `TextOp::HybridSearch` inputs.
pub(super) struct HybridArgs<'o> {
    pub collection: &'o str,
    /// Field the text leg searches. Empty searches every string field.
    pub field: String,
    pub query_vector: &'o [f32],
    pub query_text: &'o str,
    pub top_k: usize,
    pub fuzzy: bool,
    pub vector_weight: f32,
    pub rls_filters: &'o [u8],
    pub score_alias: Option<&'o str>,
}

/// `TextOp::HybridSearchTriple` inputs.
pub(super) struct HybridTripleArgs<'o> {
    pub collection: &'o str,
    /// Field the text leg searches. Empty searches every string field.
    pub field: String,
    pub query_vector: &'o [f32],
    pub query_text: &'o str,
    pub graph_seed_id: &'o str,
    pub graph_depth: usize,
    pub graph_edge_label: Option<&'o str>,
    pub top_k: usize,
    pub fuzzy: bool,
    pub rrf_k: (f64, f64, f64),
    pub rls_filters: &'o [u8],
    pub score_alias: Option<&'o str>,
}

/// Rank search results in their returned order under `source`.
fn ranked(results: &[SearchResult], source: &'static str) -> Vec<RankedResult> {
    results
        .iter()
        .enumerate()
        .map(|(rank, r)| RankedResult {
            document_id: r.id.clone(),
            rank,
            score: 1.0 - r.distance,
            source,
        })
        .collect()
}

/// `id` + fused-score rows.
fn fused_result(score_alias: Option<String>, fused: Vec<FusedResult>) -> QueryResult {
    QueryResult {
        columns: vec![
            "id".to_string(),
            score_alias.unwrap_or_else(|| "rrf_score".to_string()),
        ],
        rows: fused
            .into_iter()
            .map(|f| vec![Value::String(f.document_id), Value::Float(f.rrf_score)])
            .collect(),
        rows_affected: 0,
        command: None,
    }
}

/// `TextOp::HybridSearch`: vector and text legs fused with weighted RRF.
pub(super) fn hybrid_search<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: HybridArgs<'_>,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let metadata_filter = decode_rls_filter(args.rls_filters)?;
    let collection = args.collection.to_owned();
    let field = args.field;
    let query_vector = args.query_vector.to_vec();
    let query_text = args.query_text.to_owned();
    let top_k = args.top_k;
    let fuzzy = args.fuzzy;
    let vector_weight = args.vector_weight;
    let score_alias = args.score_alias.map(str::to_owned);
    let fts_state = Arc::clone(&engine.fts_state);
    let crdt = Arc::clone(&engine.crdt);
    let vector_state = Arc::clone(&engine.vector_state);
    Ok(Box::pin(async move {
        let text_params = or_params(fuzzy);
        let text_results = run_text_search(
            &fts_state,
            &crdt,
            TextSearchRequest {
                collection: &collection,
                field: &field,
                query: &query_text,
                top_k: top_k * 3,
                params: &text_params,
                allowed_ids: None,
            },
        )?;
        let vector_results = run_vector_search(
            &vector_state,
            &crdt,
            &collection,
            &collection,
            &query_vector,
            top_k * 3,
            metadata_filter.as_ref(),
            &[],
            None,
            None,
            false,
            None,
            None,
        )
        .await
        .map_err(|e| LiteError::Query(e.to_string()))?;

        let text_k = 60.0 * (1.0 - vector_weight as f64);
        let vector_k = 60.0 * vector_weight as f64;
        let fused = reciprocal_rank_fusion_weighted(
            &[
                ranked(&vector_results, "vector"),
                ranked(&text_results, "text"),
            ],
            &[vector_k, text_k],
            top_k,
        );
        Ok(fused_result(score_alias, fused))
    }))
}

/// `TextOp::HybridSearchTriple`: vector, text, and graph BFS legs fused with
/// per-source RRF constants.
pub(super) fn hybrid_search_triple<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: HybridTripleArgs<'_>,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let metadata_filter = decode_rls_filter(args.rls_filters)?;
    let collection = args.collection.to_owned();
    let field = args.field;
    let query_vector = args.query_vector.to_vec();
    let query_text = args.query_text.to_owned();
    let graph_seed_id = args.graph_seed_id.to_owned();
    let graph_depth = args.graph_depth;
    let graph_edge_label = args.graph_edge_label.map(str::to_owned);
    let top_k = args.top_k;
    let fuzzy = args.fuzzy;
    let (kv, kt, kg) = args.rrf_k;
    let score_alias = args.score_alias.map(str::to_owned);
    let fts_state = Arc::clone(&engine.fts_state);
    let crdt = Arc::clone(&engine.crdt);
    let vector_state = Arc::clone(&engine.vector_state);
    let csr = Arc::clone(&engine.csr);
    Ok(Box::pin(async move {
        // Leg 1: text search.
        let text_params = or_params(fuzzy);
        let text_results = run_text_search(
            &fts_state,
            &crdt,
            TextSearchRequest {
                collection: &collection,
                field: &field,
                query: &query_text,
                top_k: top_k * 3,
                params: &text_params,
                allowed_ids: None,
            },
        )?;

        // Leg 2: vector search.
        let vector_results = run_vector_search(
            &vector_state,
            &crdt,
            &collection,
            &collection,
            &query_vector,
            top_k * 3,
            metadata_filter.as_ref(),
            &[],
            None,
            None,
            false,
            None,
            None,
        )
        .await
        .map_err(|e| LiteError::Query(e.to_string()))?;

        // Leg 3: graph BFS from seed node.
        let graph_ranked: Vec<RankedResult> = if graph_depth > 0 {
            let csr_guard = csr.lock().map_err(|_| LiteError::LockPoisoned)?;
            match csr_guard.get(collection.as_str()) {
                Some(csr_idx) => {
                    let max_vis = graph_depth
                        .saturating_mul(top_k * 3)
                        .max(DEFAULT_MAX_VISITED);
                    csr_idx
                        .traverse_bfs(
                            nodedb_graph::BfsParams {
                                start_nodes: &[graph_seed_id.as_str()],
                                label_filter: graph_edge_label.as_deref(),
                                direction: Direction::Out,
                                max_depth: graph_depth,
                                max_visited: max_vis,
                                frontier_bitmap: None,
                            },
                            None,
                        )
                        .into_iter()
                        .enumerate()
                        .map(|(rank, id)| RankedResult {
                            document_id: id,
                            rank,
                            score: 0.0,
                            source: "graph",
                        })
                        .collect()
                }
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };

        let fused = reciprocal_rank_fusion_weighted(
            &[
                ranked(&vector_results, "vector"),
                ranked(&text_results, "text"),
                graph_ranked,
            ],
            &[kv, kt, kg],
            top_k,
        );
        Ok(fused_result(score_alias, fused))
    }))
}
