// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::reads::msgpack_bytes_to_crdt_fields;
use super::super::write_helpers::affected;
use super::super::write_helpers::{decode_strict_fields, fields_to_values, strict_schema};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::text_index::{index_strict_rows, reindex_documents, reindex_strict_rows};
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;
use std::collections::HashMap;

/// PointPut: unconditional overwrite (upsert semantics).
pub async fn point_put<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    point_put_coordinated(engine, None, collection, document_id, value_bytes).await
}

pub(crate) async fn point_put_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return point_put_admitted(engine, permit, collection, document_id, value_bytes).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result =
        point_put_admitted(engine, guard.permit(), collection, document_id, value_bytes).await;
    guard.finish(result)
}

pub(crate) async fn point_put_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    if is_strict(engine, collection) {
        let fields = decode_strict_fields(value_bytes)?;
        let existing_pk = Value::String(document_id.to_string());
        if engine.strict.get(collection, &existing_pk).await?.is_some() {
            let updates: HashMap<String, Value> = fields.into_iter().collect();
            engine
                .strict
                .update(collection, &existing_pk, &updates)
                .await?;
        } else {
            let schema = strict_schema(engine, collection)?;
            let values = fields_to_values(&fields, &schema.columns);
            engine.strict.insert(collection, &values).await?;
        }
        reindex_strict_rows(engine, collection, &[existing_pk]).await?;
    } else {
        let crdt_fields = msgpack_bytes_to_crdt_fields(value_bytes)?;
        let loro_fields: Vec<(&str, loro::LoroValue)> = crdt_fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        engine
            .crdt
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .upsert(collection, document_id, &loro_fields)?;
        reindex_documents(engine, collection, [document_id])?;
    }
    Ok(affected(1, "INSERT"))
}

/// PointInsert: insert-only, fail on duplicate PK (or skip if `if_absent`).
pub async fn point_insert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    if_absent: bool,
) -> Result<QueryResult, LiteError> {
    point_insert_coordinated(
        engine,
        None,
        collection,
        document_id,
        value_bytes,
        if_absent,
    )
    .await
}

pub(crate) async fn point_insert_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    if_absent: bool,
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return point_insert_admitted(
            engine,
            permit,
            collection,
            document_id,
            value_bytes,
            if_absent,
        )
        .await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = point_insert_admitted(
        engine,
        guard.permit(),
        collection,
        document_id,
        value_bytes,
        if_absent,
    )
    .await;
    guard.finish(result)
}

pub(crate) async fn point_insert_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    if_absent: bool,
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    if is_strict(engine, collection) {
        let pk = Value::String(document_id.to_string());
        if engine.strict.get(collection, &pk).await?.is_some() {
            if if_absent {
                return Ok(affected(0, "INSERT"));
            }
            return Err(LiteError::BadRequest {
                detail: format!(
                    "duplicate key value violates unique constraint on '{collection}' (id = '{document_id}')"
                ),
            });
        }
        let fields = decode_strict_fields(value_bytes)?;
        let schema = strict_schema(engine, collection)?;
        let values = fields_to_values(&fields, &schema.columns);
        engine.strict.insert(collection, &values).await?;
        index_strict_rows(engine, collection, [values.as_slice()])?;
    } else {
        let crdt_fields = msgpack_bytes_to_crdt_fields(value_bytes)?;
        let loro_fields: Vec<(&str, loro::LoroValue)> = crdt_fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        if crdt.exists(collection, document_id) {
            if if_absent {
                return Ok(affected(0, "INSERT"));
            }
            return Err(LiteError::BadRequest {
                detail: format!(
                    "duplicate key value violates unique constraint on '{collection}' (id = '{document_id}')"
                ),
            });
        }
        crdt.upsert(collection, document_id, &loro_fields)?;
        drop(crdt);
        reindex_documents(engine, collection, [document_id])?;
    }
    Ok(affected(1, "INSERT"))
}
