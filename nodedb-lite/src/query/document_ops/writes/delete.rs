// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::write_helpers::affected;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::text_index::{reindex_documents, reindex_strict_rows};
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

/// PointDelete: remove a document by ID.
pub async fn point_delete<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
) -> Result<QueryResult, LiteError> {
    point_delete_coordinated(engine, None, collection, document_id).await
}

pub(crate) async fn point_delete_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    document_id: &str,
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return point_delete_admitted(engine, permit, collection, document_id).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = point_delete_admitted(engine, guard.permit(), collection, document_id).await;
    guard.finish(result)
}

pub(crate) async fn point_delete_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    document_id: &str,
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    if is_strict(engine, collection) {
        let pk = Value::String(document_id.to_string());
        let deleted = engine.strict.delete(collection, &pk).await?;
        reindex_strict_rows(engine, collection, &[pk]).await?;
        Ok(affected(if deleted { 1 } else { 0 }, "DELETE"))
    } else {
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        if !crdt.exists(collection, document_id) {
            return Ok(affected(0, "DELETE"));
        }
        crdt.delete(collection, document_id)
            .map_err(|e| LiteError::Storage {
                detail: e.to_string(),
            })?;
        drop(crdt);
        reindex_documents(engine, collection, [document_id])?;
        Ok(affected(1, "DELETE"))
    }
}
