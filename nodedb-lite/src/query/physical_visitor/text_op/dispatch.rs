// SPDX-License-Identifier: Apache-2.0

//! `TextOp` variant routing.

use nodedb_physical::physical_plan::TextOp;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::adapter::LitePhysicalFut;
use super::super::text_config::text_set_config;
use super::hybrid::{HybridArgs, HybridTripleArgs, hybrid_search, hybrid_search_triple};
use super::search::{
    ScoreScanArgs, TextSearchArgs, bm25_score_scan, match_score_column, phrase_search,
    refuse_scan_filters, text_search,
};
use super::sync::{fts_delete_doc, fts_index_doc};

/// Dispatch a `TextOp` against the whole-document index of its collection.
///
/// `TextOp` carries no field, so an op that arrives without one searches
/// every string field. Returns a pinned future that resolves to a
/// `QueryResult`.
pub(crate) fn execute_text_op<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    op: &TextOp,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    execute_text_op_on_field(engine, op, "")
}

/// Dispatch a `TextOp`, scoping every text read to the index of `field`.
///
/// An empty `field` names the whole-document index. Ops that write or
/// configure the index ignore `field`.
pub(crate) fn execute_text_op_admitted<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: Option<&'a crate::engine::fts::coordinator::TextMutationPermit>,
    op: &TextOp,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    execute_text_op_on_field_admitted(engine, permit, op, "")
}

pub(crate) fn execute_text_op_on_field<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    op: &TextOp,
    field: &str,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    execute_text_op_on_field_admitted(engine, None, op, field)
}

pub(crate) fn execute_text_op_on_field_admitted<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: Option<&'a crate::engine::fts::coordinator::TextMutationPermit>,
    op: &TextOp,
    field: &str,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let field = field.to_owned();
    match op {
        TextOp::Search {
            collection,
            field: op_field,
            query,
            top_k,
            mode: _,
            fuzzy,
            prefilter: _,
            filters,
            rls_filters,
            scores,
        } => {
            refuse_scan_filters("TextOp::Search", filters)?;
            let field = op_field.clone().unwrap_or(field);
            let score_column = match_score_column(&field, query, scores)?;
            text_search(
                engine,
                TextSearchArgs {
                    collection: collection.as_str(),
                    field,
                    query,
                    top_k: *top_k,
                    fuzzy: *fuzzy,
                    rls_filters,
                    score_column,
                },
            )
        }

        TextOp::BM25ScoreScan {
            collection,
            filters,
            rls_filters: _,
            scores,
            bound,
        } => {
            refuse_scan_filters("TextOp::BM25ScoreScan", filters)?;
            let [score] = scores.as_slice() else {
                return Err(LiteError::Unsupported {
                    detail: format!(
                        "TextOp::BM25ScoreScan with {} score columns: NodeDB-Lite scores one \
                         bm25_score(...) per scan",
                        scores.len()
                    ),
                });
            };
            Ok(bm25_score_scan(
                engine,
                ScoreScanArgs {
                    collection: collection.as_str(),
                    field: score.field.clone().unwrap_or(field),
                    query: &score.query,
                    score_alias: &score.alias,
                    fuzzy: score.fuzzy,
                    bound: bound.clone(),
                },
            ))
        }

        TextOp::PhraseSearch {
            collection,
            field: op_field,
            terms,
            top_k,
            prefilter: _,
            filters,
            rls_filters: _,
            scores: _,
        } => {
            refuse_scan_filters("TextOp::PhraseSearch", filters)?;
            Ok(phrase_search(
                engine,
                collection.as_str(),
                op_field.clone().unwrap_or(field),
                terms,
                *top_k,
            ))
        }

        TextOp::HybridSearch {
            collection,
            vector_field,
            query_vector,
            text_field,
            query_text,
            filters,
            top_k,
            ef_search: _,
            mode: _,
            fuzzy,
            vector_weight,
            filter_bitmap: _,
            rls_filters,
            score_alias,
        } => {
            refuse_scan_filters("TextOp::HybridSearch", filters)?;
            hybrid_search(
                engine,
                HybridArgs {
                    collection: collection.as_str(),
                    field: text_field.clone().unwrap_or(field),
                    vector_field,
                    query_vector,
                    query_text,
                    top_k: *top_k,
                    fuzzy: *fuzzy,
                    vector_weight: *vector_weight,
                    rls_filters,
                    score_alias: score_alias.as_deref(),
                },
            )
        }

        TextOp::HybridSearchTriple {
            collection,
            vector_field,
            query_vector,
            text_field,
            query_text,
            filters,
            graph_seed_id,
            graph_depth,
            graph_edge_label,
            top_k,
            ef_search: _,
            mode: _,
            fuzzy,
            rrf_k,
            filter_bitmap: _,
            rls_filters,
            score_alias,
        } => {
            refuse_scan_filters("TextOp::HybridSearchTriple", filters)?;
            hybrid_search_triple(
                engine,
                HybridTripleArgs {
                    collection: collection.as_str(),
                    field: text_field.clone().unwrap_or(field),
                    vector_field,
                    query_vector,
                    query_text,
                    graph_seed_id,
                    graph_depth: *graph_depth,
                    graph_edge_label: graph_edge_label.as_deref(),
                    top_k: *top_k,
                    fuzzy: *fuzzy,
                    rrf_k: *rrf_k,
                    rls_filters,
                    score_alias: score_alias.as_deref(),
                },
            )
        }

        // Lite keeps one whole-document index for these frames: the fields
        // join in field-name order, as Origin's whole-document index does.
        TextOp::FtsIndexDoc {
            collection,
            surrogate,
            fields,
            provenance: _,
        } => Ok(fts_index_doc(
            engine,
            permit,
            collection.as_str(),
            *surrogate,
            &nodedb_fts::DocumentText::from_fields(fields.iter().cloned()).whole(),
        )),

        // No bound surrogate names no document: the delete removes nothing.
        TextOp::FtsDeleteDoc {
            surrogate: None, ..
        } => Ok(Box::pin(async { Ok(QueryResult::empty()) })),
        TextOp::FtsDeleteDoc {
            collection,
            surrogate: Some(surrogate),
            provenance: _,
        } => Ok(fts_delete_doc(
            engine,
            permit,
            collection.as_str(),
            *surrogate,
        )),

        // ── Config write ──────────────────────────────────────────────────────
        TextOp::SetTextConfig {
            collection,
            analyzer_name,
            fuzzy_default,
        } => text_set_config(
            engine,
            permit,
            collection.as_str().to_string(),
            analyzer_name.clone(),
            *fuzzy_default,
        ),
    }
}
