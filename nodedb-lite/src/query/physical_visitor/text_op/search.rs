// SPDX-License-Identifier: Apache-2.0

//! Text reads: BM25 search, the all-documents score scan, and phrase search.

use std::sync::Arc;

use nodedb_types::filter::MetadataFilter;
use nodedb_types::result::{QueryResult, SearchResult};
use nodedb_types::text_search::{QueryMode, TextSearchParams};
use nodedb_types::value::Value;

use crate::engine::fts::{TextSearchRequest, run_text_search};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::adapter::LitePhysicalFut;

/// Plain OR-mode search params with the given fuzzy flag.
pub(super) fn or_params(fuzzy: bool) -> TextSearchParams {
    TextSearchParams {
        fuzzy,
        mode: QueryMode::Or,
    }
}

/// Decode the serialized RLS post-filter of a text op. Empty bytes mean no
/// filter.
pub(super) fn decode_rls_filter(rls_filters: &[u8]) -> Result<Option<MetadataFilter>, LiteError> {
    if rls_filters.is_empty() {
        return Ok(None);
    }
    zerompk::from_msgpack(rls_filters)
        .map(Some)
        .map_err(|e| LiteError::Serialization {
            detail: format!("decode MetadataFilter: {e}"),
        })
}

/// Keep only the results whose metadata passes `filter`.
///
/// A result whose metadata cannot be encoded for the filter fails the
/// search: dropping it would claim it did not match.
pub(super) fn retain_matching(
    results: Vec<SearchResult>,
    filter: &MetadataFilter,
) -> Result<Vec<SearchResult>, LiteError> {
    let mut kept = Vec::with_capacity(results.len());
    for r in results {
        let json_doc = serde_json::to_value(&r.metadata).map_err(|e| LiteError::Serialization {
            detail: format!("encode metadata of '{}' for the RLS filter: {e}", r.id),
        })?;
        if nodedb_query::metadata_filter::matches_metadata_filter(&json_doc, filter) {
            kept.push(r);
        }
    }
    Ok(kept)
}

/// `id` + score rows.
fn id_score_result(score_column: String, rows: Vec<(String, f64)>) -> QueryResult {
    QueryResult {
        columns: vec!["id".to_string(), score_column],
        rows: rows
            .into_iter()
            .map(|(id, score)| vec![Value::String(id), Value::Float(score)])
            .collect(),
        rows_affected: 0,
        command: None,
    }
}

/// `TextOp::Search`: ranked BM25 hits, RLS-filtered.
pub(super) fn text_search<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    field: String,
    query: &str,
    top_k: usize,
    fuzzy: bool,
    rls_filters: &[u8],
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let metadata_filter = decode_rls_filter(rls_filters)?;
    let collection = collection.to_owned();
    let query = query.to_owned();
    let fts_state = Arc::clone(&engine.fts_state);
    let crdt = Arc::clone(&engine.crdt);
    Ok(Box::pin(async move {
        let params = or_params(fuzzy);
        let mut results = run_text_search(
            &fts_state,
            &crdt,
            TextSearchRequest {
                collection: &collection,
                field: &field,
                query: &query,
                top_k,
                params: &params,
                allowed_ids: None,
            },
        )?;
        if let Some(filter) = metadata_filter {
            results = retain_matching(results, &filter)?;
        }
        Ok(id_score_result(
            "score".to_string(),
            results
                .into_iter()
                .map(|r| (r.id, (1.0 - r.distance) as f64))
                .collect(),
        ))
    }))
}

/// `TextOp::BM25ScoreScan`: every indexed document with its BM25 score.
pub(super) fn bm25_score_scan<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    field: String,
    query: &str,
    score_alias: &str,
    fuzzy: bool,
) -> LitePhysicalFut<'a> {
    let collection = collection.to_owned();
    let query = query.to_owned();
    let score_alias = score_alias.to_owned();
    let fts_state = Arc::clone(&engine.fts_state);
    Box::pin(async move {
        let scored = fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .scan_all_with_scores(&collection, &field, &query, &or_params(fuzzy))?;
        Ok(id_score_result(
            score_alias,
            scored
                .into_iter()
                .map(|(doc_id, score)| (doc_id, score as f64))
                .collect(),
        ))
    })
}

/// `TextOp::PhraseSearch`: documents holding the terms consecutively.
pub(super) fn phrase_search<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    field: String,
    terms: &[String],
    top_k: usize,
) -> LitePhysicalFut<'a> {
    let collection = collection.to_owned();
    let terms = terms.to_vec();
    let fts_state = Arc::clone(&engine.fts_state);
    Box::pin(async move {
        let results = fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .phrase_search(&collection, &field, &terms, top_k, &or_params(false))?;
        Ok(id_score_result(
            "score".to_string(),
            results
                .into_iter()
                .map(|r| (r.doc_id, r.score as f64))
                .collect(),
        ))
    })
}
