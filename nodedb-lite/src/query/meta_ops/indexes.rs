// SPDX-License-Identifier: Apache-2.0
//! Index meta-ops: RebuildIndex.

use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

/// `RebuildIndex` — rebuild index entries from the collection's rows: the
/// index named `index_name`, or every index on `collection` when `None`.
///
/// The `concurrent` flag has no effect on Lite: a rebuild holds the CRDT lock
/// for its duration, so writes wait for it rather than run beside it.
pub async fn handle_rebuild_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    index_name: Option<&str>,
    _concurrent: bool,
) -> Result<QueryResult, LiteError> {
    crate::query::document_ops::indexes::reindex(engine, collection, index_name).await
}
