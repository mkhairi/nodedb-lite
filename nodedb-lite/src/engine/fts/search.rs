// SPDX-License-Identifier: Apache-2.0

//! Free-function FTS search callable from both `NodeDbLite` and
//! `LiteDataPlaneVisitor` without depending on either concrete type.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use nodedb_types::result::SearchResult;
use nodedb_types::text_search::TextSearchParams;

use crate::engine::crdt::CrdtEngine;
use crate::engine::fts::state::FtsState;
use crate::error::LiteError;
use crate::nodedb::convert::loro_value_to_document;
use crate::nodedb::lock_ext::LockExt;

/// One BM25 text query.
pub(crate) struct TextSearchRequest<'a> {
    pub collection: &'a str,
    /// Field whose index the query runs against. Empty searches the
    /// whole-document index, which covers every string field.
    pub field: &'a str,
    pub query: &'a str,
    pub top_k: usize,
    pub params: &'a TextSearchParams,
    /// When `Some`, only documents whose string ID is in the set are returned.
    pub allowed_ids: Option<&'a HashSet<String>>,
}

/// Run a BM25 text query against the in-memory FTS index and hydrate each
/// hit with the document's fields from CRDT storage.
///
/// The FTS score is converted to a `distance` in `[0.0, 1.0]` via
/// `1.0 - min(score / 20.0, 1.0)` so callers can rank text and vector hits
/// on the same axis (lower = better).
///
/// With `allowed_ids`, the filter is applied after an over-fetch (8×
/// multiplier) so that haystack-scoped queries surface the full relevant
/// candidate set even when relevant documents rank lower than the global
/// top-k.
///
/// A collection nothing was text-indexed in returns an empty list, as does a
/// query no document matches. Fails with [`LiteError::TextIndexMissing`] when
/// the collection has text-indexed documents but none under the named field,
/// and with the read error when the index read fails.
pub(crate) fn run_text_search(
    fts_state: &Arc<FtsState>,
    crdt: &Arc<Mutex<CrdtEngine>>,
    req: TextSearchRequest<'_>,
) -> Result<Vec<SearchResult>, LiteError> {
    let raw = {
        let mgr = fts_state.manager.lock_or_recover();
        match req.allowed_ids {
            Some(ids) => mgr.search_with_allowed(
                req.collection,
                req.field,
                req.query,
                req.top_k,
                req.params,
                ids,
            )?,
            None => mgr.search(req.collection, req.field, req.query, req.top_k, req.params)?,
        }
    };
    let crdt_guard = crdt.lock_or_recover();
    Ok(raw
        .into_iter()
        .map(|r| {
            let metadata = match crdt_guard.read(req.collection, &r.doc_id) {
                Some(loro_val) => loro_value_to_document(&r.doc_id, &loro_val).fields,
                None => HashMap::new(),
            };
            SearchResult {
                id: r.doc_id,
                node_id: None,
                distance: 1.0 - (r.score / 20.0).min(1.0),
                metadata,
            }
        })
        .collect())
}
