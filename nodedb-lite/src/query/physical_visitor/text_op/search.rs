// SPDX-License-Identifier: Apache-2.0

//! Text reads: BM25 search, the all-documents score scan, and phrase search.

use std::sync::Arc;

use nodedb_physical::physical_plan::{ScoreScanBound, TextScoreSpec};
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

/// Refuse residual `ScanFilter` predicates on a text op.
///
/// Lite's planner carries WHERE predicates as an RLS `MetadataFilter` and
/// applies the rest after the scan, so it never fills `filters`. An op that
/// arrives with them would otherwise rank rows they exclude.
pub(super) fn refuse_scan_filters(op: &str, filters: &[u8]) -> Result<(), LiteError> {
    if filters.is_empty() {
        return Ok(());
    }
    Err(LiteError::Unsupported {
        detail: format!(
            "{op} with residual ScanFilter predicates: NodeDB-Lite applies WHERE after the \
             text scan, so the planner must not push them into the op"
        ),
    })
}

/// The output column of a search's score.
///
/// `score`, unless a `bm25_score(...)` column names the same field and query
/// as the match: the score then lands under that column's alias. A score of
/// another field or query has no Lite execution path.
pub(super) fn match_score_column(
    field: &str,
    query: &str,
    scores: &[TextScoreSpec],
) -> Result<String, LiteError> {
    match scores {
        [] => Ok("score".to_string()),
        [score] if score.field.as_deref().unwrap_or("") == field && score.query == query => {
            Ok(score.alias.clone())
        }
        _ => Err(LiteError::Unsupported {
            detail: "bm25_score(...) of a field or query other than the text_match: \
                     NodeDB-Lite scores only the matched index and query"
                .to_string(),
        }),
    }
}

/// `TextOp::Search` inputs.
pub(super) struct TextSearchArgs<'o> {
    pub collection: &'o str,
    /// Field index the query reads. Empty reads the whole-document index.
    pub field: String,
    pub query: &'o str,
    pub top_k: usize,
    pub fuzzy: bool,
    pub rls_filters: &'o [u8],
    /// Output column the score lands in.
    pub score_column: String,
}

/// `TextOp::Search`: ranked BM25 hits, RLS-filtered.
pub(super) fn text_search<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: TextSearchArgs<'_>,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let metadata_filter = decode_rls_filter(args.rls_filters)?;
    let collection = args.collection.to_owned();
    let field = args.field;
    let query = args.query.to_owned();
    let top_k = args.top_k;
    let fuzzy = args.fuzzy;
    let score_column = args.score_column;
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
            score_column,
            results
                .into_iter()
                .map(|r| (r.id, (1.0 - r.distance) as f64))
                .collect(),
        ))
    }))
}

/// `TextOp::BM25ScoreScan` inputs.
pub(super) struct ScoreScanArgs<'o> {
    pub collection: &'o str,
    /// Field index the score reads. Empty reads the whole-document index.
    pub field: String,
    pub query: &'o str,
    pub score_alias: &'o str,
    pub fuzzy: bool,
    pub bound: Option<ScoreScanBound>,
}

/// Keep the first `bound.rows` rows, in the bound's order when it has one.
fn apply_bound(rows: &mut Vec<(String, f64)>, bound: &ScoreScanBound) {
    if let Some(order) = &bound.order {
        rows.sort_by(|a, b| {
            let ord = a.1.total_cmp(&b.1);
            if order.ascending { ord } else { ord.reverse() }
        });
    }
    rows.truncate(bound.rows);
}

/// `TextOp::BM25ScoreScan`: every indexed document with its BM25 score.
pub(super) fn bm25_score_scan<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: ScoreScanArgs<'_>,
) -> LitePhysicalFut<'a> {
    let collection = args.collection.to_owned();
    let field = args.field;
    let query = args.query.to_owned();
    let score_alias = args.score_alias.to_owned();
    let fuzzy = args.fuzzy;
    let bound = args.bound;
    let fts_state = Arc::clone(&engine.fts_state);
    Box::pin(async move {
        let scored = fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .scan_all_with_scores(&collection, &field, &query, &or_params(fuzzy))?;
        let mut rows: Vec<(String, f64)> = scored
            .into_iter()
            .map(|(doc_id, score)| (doc_id, score as f64))
            .collect();
        if let Some(bound) = &bound {
            apply_bound(&mut rows, bound);
        }
        Ok(id_score_result(score_alias, rows))
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
