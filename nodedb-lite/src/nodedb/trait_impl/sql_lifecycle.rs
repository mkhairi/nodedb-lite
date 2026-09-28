// SPDX-License-Identifier: Apache-2.0

//! SQL execution and text-search helpers for `NodeDbLite`.

use std::collections::HashSet;

use nodedb_types::error::{NodeDbError, NodeDbResult};
use nodedb_types::result::{QueryResult, SearchResult};
use nodedb_types::text_search::TextSearchParams;
use nodedb_types::value::Value;

use crate::engine::fts::{TextSearchRequest, run_text_search};
use crate::nodedb::NodeDbLite;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Execute a SQL statement against the embedded query engine.
    ///
    /// `params` binds `$1`, `$2`, … placeholders in `query` at the AST level
    /// before planning. Supported `Value` variants: `Null`, `Bool`, `Integer`,
    /// `Float`, `String`, `Uuid`. Pass an empty slice when no parameters are
    /// needed.
    pub(super) async fn execute_sql_impl(
        &self,
        query: &str,
        params: &[Value],
    ) -> NodeDbResult<QueryResult> {
        self.query_engine
            .execute_sql_with_params(query, params)
            .await
            .map_err(NodeDbError::from)
    }

    /// Run a BM25 text query against the FTS index for `field` of
    /// `collection` and hydrate each hit with the document's fields from
    /// CRDT storage. An empty `field` searches the whole-document index,
    /// which covers every string field.
    ///
    /// The FTS score is converted to a `distance` in `[0.0, 1.0]` via
    /// `1.0 - min(score / 20.0, 1.0)` so callers can rank text and vector hits
    /// on the same axis (lower = better). The `20.0` divisor matches the BM25
    /// score range produced by the bundled analyzer pipeline.
    ///
    /// A known collection with no text-indexed documents returns an empty
    /// list. Fails with `collection_not_found` for a collection no engine or
    /// catalog knows, with a bad-request error for a named field no document
    /// holds while other fields are indexed, and when the index read fails.
    pub(super) async fn text_search_impl(
        &self,
        collection: &str,
        field: &str,
        query: &str,
        top_k: usize,
        params: TextSearchParams,
        allowed_ids: Option<&HashSet<String>>,
    ) -> NodeDbResult<Vec<SearchResult>> {
        let indexed = self
            .fts_state
            .manager
            .lock_or_recover()
            .has_text_index(collection);
        if !indexed && !self.collection_known(collection).await? {
            return Err(NodeDbError::collection_not_found(collection));
        }
        run_text_search(
            &self.fts_state,
            &self.crdt,
            TextSearchRequest {
                collection,
                field,
                query,
                top_k,
                params: &params,
                allowed_ids,
            },
        )
        .map_err(NodeDbError::from)
    }
}
