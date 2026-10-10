// SPDX-License-Identifier: Apache-2.0

//! Text search lowering: converts an `FtsQuery` to a `TextOp` and runs it
//! against the index of the field the SQL names (`text_match(field, q)`,
//! `bm25_score(field, q)`). No field searches the whole-document index.
//! Empty phrase queries short-circuit to an empty `QueryResult`.
//! `FtsQuery::Not` is rejected — Lite does not support standalone NOT FTS
//! queries.

use nodedb_physical::physical_plan::{TextOp, TextScoreSpec};
use nodedb_sql::TextSearchVisitArgs;
use nodedb_sql::fts_types::FtsQuery;
use nodedb_sql::types::TextSearchShape;
use nodedb_sql::types::filter::Filter;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::filter_convert::sql_filters_to_metadata;
use crate::query::physical_visitor::execute_text_op_on_field;
use crate::storage::engine::StorageEngine;

use super::visitor::LiteFut;

/// Lite holds a bare collection name. `DatabaseId::DEFAULT` keeps it unqualified.
fn qualified(collection: &str) -> nodedb_types::QualifiedCollection {
    nodedb_types::QualifiedCollection::new(nodedb_types::DatabaseId::DEFAULT, collection)
}

/// The WHERE predicates a `MetadataFilter` can express, msgpack-encoded for
/// the op's RLS slot.
///
/// Complex QExpr predicates cannot be serialized to the FTS physical visitor.
/// They are dropped from the pre-filter here and applied at post-scan time by
/// `apply_scan_post_processing` on the caller side.
fn metadata_prefilter(filters: &[Filter]) -> Result<Vec<u8>, LiteError> {
    if filters.is_empty() {
        return Ok(Vec::new());
    }
    match sql_filters_to_metadata(filters, &[])?.meta {
        None => Ok(Vec::new()),
        Some(mf) => zerompk::to_msgpack_vec(&mf).map_err(|e| LiteError::Serialization {
            detail: format!("encode MetadataFilter: {e}"),
        }),
    }
}

pub(super) fn lower_text_search<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: TextSearchVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    let TextSearchVisitArgs {
        collection,
        shape,
        filters,
        scores,
    } = args;
    let scores: Vec<TextScoreSpec> = scores
        .iter()
        .map(|s| TextScoreSpec {
            field: s.field.clone(),
            query: s.query.clone(),
            mode: s.mode,
            fuzzy: s.fuzzy,
            alias: s.alias.clone(),
        })
        .collect();
    let text_op = match shape {
        // `bm25_score(...)` with no `text_match`: the score of every indexed
        // row. WHERE applies after the scan, and the relational tail applies
        // ORDER BY and LIMIT, so no bound is pushed.
        TextSearchShape::ScoreScan => TextOp::BM25ScoreScan {
            collection: qualified(collection),
            filters: Vec::new(),
            rls_filters: Vec::new(),
            scores,
            bound: None,
        },
        TextSearchShape::Match {
            field,
            query,
            mode,
            top_k,
        } => {
            let top_k = top_k.unwrap_or(usize::MAX);
            match query {
                FtsQuery::Phrase(terms) => {
                    // Phrase queries with no analyzed terms produce no results.
                    if terms.is_empty() {
                        return Ok(Box::pin(async move {
                            Ok(QueryResult {
                                columns: vec!["id".to_string(), "score".to_string()],
                                rows: vec![],
                                rows_affected: 0,
                                command: None,
                            })
                        }));
                    }
                    TextOp::PhraseSearch {
                        collection: qualified(collection),
                        field: field.clone(),
                        terms: terms.clone(),
                        top_k,
                        prefilter: None,
                        filters: Vec::new(),
                        rls_filters: Vec::new(),
                        scores,
                    }
                }
                FtsQuery::Not(_) => {
                    return Err(LiteError::BadRequest {
                        detail: "FTS NOT queries are not supported".to_string(),
                    });
                }
                other => {
                    let Some(plain) = other.to_plain_string() else {
                        return Err(LiteError::BadRequest {
                            detail: "FTS query cannot be expressed as a plain text search"
                                .to_string(),
                        });
                    };
                    TextOp::Search {
                        collection: qualified(collection),
                        field: field.clone(),
                        query: plain,
                        top_k,
                        mode: *mode,
                        fuzzy: other.is_fuzzy(),
                        prefilter: None,
                        filters: Vec::new(),
                        rls_filters: metadata_prefilter(filters)?,
                        scores,
                    }
                }
            }
        }
    };

    execute_text_op_on_field(engine, &text_op, "").map(|fut| Box::pin(fut) as LiteFut<'a>)
}
