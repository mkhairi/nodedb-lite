// SPDX-License-Identifier: Apache-2.0
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;

/// BulkDelete dispatch target.
///
/// `DocumentOp::BulkDelete` carries a msgpack-encoded filter predicate produced
/// by Origin's Calvin/OLLP planner. Lite's SQL visitor resolves DELETE to
/// point-key `PointDelete` ops via `target_keys`, and CRDT sync plans carry no
/// bulk-predicate deletes, so Lite has no evaluator for this op and refuses it.
pub async fn bulk_delete<S: StorageEngine>(
    _engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    Err(LiteError::Unsupported {
        detail: format!(
            "predicate bulk delete on '{collection}': Lite deletes by key; \
             issue DELETE ... WHERE id = ... instead"
        ),
    })
}
