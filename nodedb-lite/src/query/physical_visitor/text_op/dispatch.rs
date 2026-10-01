// SPDX-License-Identifier: Apache-2.0

//! `TextOp` variant routing.

use nodedb_physical::physical_plan::TextOp;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::adapter::LitePhysicalFut;
use super::super::text_config::text_set_config;
use super::hybrid::{HybridArgs, HybridTripleArgs, hybrid_search, hybrid_search_triple};
use super::search::{bm25_score_scan, phrase_search, text_search};
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
            query,
            top_k,
            fuzzy,
            rls_filters,
            ..
        } => text_search(
            engine,
            collection.as_str(),
            field,
            query,
            *top_k,
            *fuzzy,
            rls_filters,
        ),

        TextOp::BM25ScoreScan {
            collection,
            query,
            score_alias,
            fuzzy,
        } => Ok(bm25_score_scan(
            engine,
            collection.as_str(),
            field,
            query,
            score_alias,
            *fuzzy,
        )),

        TextOp::PhraseSearch {
            collection,
            terms,
            top_k,
            ..
        } => Ok(phrase_search(
            engine,
            collection.as_str(),
            field,
            terms,
            *top_k,
        )),

        TextOp::HybridSearch {
            collection,
            query_vector,
            query_text,
            top_k,
            fuzzy,
            vector_weight,
            rls_filters,
            score_alias,
            ..
        } => hybrid_search(
            engine,
            HybridArgs {
                collection: collection.as_str(),
                field,
                query_vector,
                query_text,
                top_k: *top_k,
                fuzzy: *fuzzy,
                vector_weight: *vector_weight,
                rls_filters,
                score_alias: score_alias.as_deref(),
            },
        ),

        TextOp::HybridSearchTriple {
            collection,
            query_vector,
            query_text,
            graph_seed_id,
            graph_depth,
            graph_edge_label,
            top_k,
            fuzzy,
            rrf_k,
            rls_filters,
            score_alias,
            ..
        } => hybrid_search_triple(
            engine,
            HybridTripleArgs {
                collection: collection.as_str(),
                field,
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
        ),

        TextOp::FtsIndexDoc {
            collection,
            surrogate,
            text,
            provenance: _,
        } => Ok(fts_index_doc(
            engine,
            permit,
            collection.as_str(),
            *surrogate,
            text,
        )),

        TextOp::FtsDeleteDoc {
            collection,
            surrogate,
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
