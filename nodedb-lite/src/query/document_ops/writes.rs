// SPDX-License-Identifier: Apache-2.0
//! Write operations for the Document engine physical visitor.

use std::collections::HashMap;

use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::truncate::{clear_spatial, truncated};
use crate::storage::engine::{StorageEngine, WriteOp};

use super::is_strict;
use super::reads::{loro_value_to_ndb_value, msgpack_bytes_to_crdt_fields, ndb_value_to_loro};
use super::write_helpers::{
    affected, decode_literal_updates, decode_strict_fields, fields_to_values, strict_schema,
};
use crate::query::text_index::{index_strict_rows, reindex_documents, reindex_strict_rows};

pub(super) type UpdateValue = nodedb_physical::physical_plan::document::types::UpdateValue;

/// PointPut: unconditional overwrite (upsert semantics).
pub async fn point_put<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
) -> Result<QueryResult, LiteError> {
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
            .upsert(collection, document_id, &loro_fields)
            .map_err(|e| LiteError::Storage {
                detail: e.to_string(),
            })?;
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
        crdt.upsert(collection, document_id, &loro_fields)
            .map_err(|e| LiteError::Storage {
                detail: e.to_string(),
            })?;
        drop(crdt);
        reindex_documents(engine, collection, [document_id])?;
    }
    Ok(affected(1, "INSERT"))
}

/// PointUpdate: read-modify-write with field-level changes.
pub async fn point_update<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let pk = Value::String(document_id.to_string());
        let field_updates = decode_literal_updates(updates)?;
        let updated = engine
            .strict
            .update(collection, &pk, &field_updates)
            .await?;
        reindex_strict_rows(engine, collection, &[pk]).await?;
        Ok(affected(if updated { 1 } else { 0 }, "UPDATE"))
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        if !crdt.exists(collection, document_id) {
            return Ok(affected(0, "UPDATE"));
        }
        let existing_val = crdt.read(collection, document_id);
        drop(crdt);
        let mut merged: HashMap<String, loro::LoroValue> = if let Some(val) = existing_val {
            match loro_value_to_ndb_value(&val) {
                Value::Object(map) => map
                    .into_iter()
                    .map(|(k, v)| (k, ndb_value_to_loro(v)))
                    .collect(),
                _ => HashMap::new(),
            }
        } else {
            HashMap::new()
        };
        for (field, update_val) in updates {
            if let UpdateValue::Literal(bytes) = update_val {
                let val: Value =
                    zerompk::from_msgpack(bytes).map_err(|e| LiteError::Serialization {
                        detail: format!("decode update literal: {e}"),
                    })?;
                merged.insert(field.clone(), ndb_value_to_loro(val));
            }
        }
        let loro_fields: Vec<(&str, loro::LoroValue)> = merged
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        engine
            .crdt
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .upsert(collection, document_id, &loro_fields)
            .map_err(|e| LiteError::Storage {
                detail: e.to_string(),
            })?;
        reindex_documents(engine, collection, [document_id])?;
        Ok(affected(1, "UPDATE"))
    }
}

/// PointDelete: remove a document by ID.
pub async fn point_delete<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
) -> Result<QueryResult, LiteError> {
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

/// BatchInsert: insert N documents in a single transaction.
pub async fn batch_insert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    documents: &[(String, Vec<u8>)],
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let schema = strict_schema(engine, collection)?;
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(documents.len());
        for (_doc_id, value_bytes) in documents {
            let fields = decode_strict_fields(value_bytes)?;
            let values = fields_to_values(&fields, &schema.columns);
            rows.push(values);
        }
        let affected_n = rows.len() as u64;
        engine.strict.insert_batch(collection, &rows).await?;
        index_strict_rows(engine, collection, rows.iter().map(Vec::as_slice))?;
        Ok(affected(affected_n, "INSERT"))
    } else {
        let mut decoded: Vec<(String, Vec<(String, loro::LoroValue)>)> =
            Vec::with_capacity(documents.len());
        for (doc_id, value_bytes) in documents {
            let crdt_fields = msgpack_bytes_to_crdt_fields(value_bytes)?;
            decoded.push((doc_id.clone(), crdt_fields));
        }
        let affected_n = decoded.len() as u64;
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        for (doc_id, fields) in &decoded {
            let loro_slice: Vec<(&str, loro::LoroValue)> = fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            crdt.upsert_deferred(collection, doc_id, &loro_slice)
                .map_err(|e| LiteError::Storage {
                    detail: e.to_string(),
                })?;
        }
        crdt.flush_deltas().map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
        drop(crdt);
        reindex_documents(
            engine,
            collection,
            decoded.iter().map(|(doc_id, _)| doc_id.as_str()),
        )?;
        Ok(affected(affected_n, "INSERT"))
    }
}

/// Upsert: insert or update. When `on_conflict_updates` is non-empty, applies
/// those assignments on conflict instead of merging the new value.
pub async fn upsert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    on_conflict_updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let pk = Value::String(document_id.to_string());
        let existed = engine.strict.get(collection, &pk).await?.is_some();
        if existed && !on_conflict_updates.is_empty() {
            let field_updates = decode_literal_updates(on_conflict_updates)?;
            engine
                .strict
                .update(collection, &pk, &field_updates)
                .await?;
        } else {
            let fields = decode_strict_fields(value_bytes)?;
            if existed {
                let updates: HashMap<String, Value> = fields.into_iter().collect();
                engine.strict.update(collection, &pk, &updates).await?;
            } else {
                let schema = strict_schema(engine, collection)?;
                let values = fields_to_values(&fields, &schema.columns);
                engine.strict.insert(collection, &values).await?;
            }
        }
        reindex_strict_rows(engine, collection, &[pk]).await?;
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
            .upsert(collection, document_id, &loro_fields)
            .map_err(|e| LiteError::Storage {
                detail: e.to_string(),
            })?;
        reindex_documents(engine, collection, [document_id])?;
    }
    Ok(affected(1, "UPSERT"))
}

/// Truncate: delete ALL documents in a collection, then the overlays they
/// fed: the FTS index, the sparse-vector postings, the R-tree entries, and
/// every vector bucket of the collection. Answers with the bare `TRUNCATE`
/// tag, never a row count.
pub async fn truncate<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let prefix = format!("{collection}:");
        let all_entries = engine
            .storage
            .scan_prefix(Namespace::Strict, prefix.as_bytes())
            .await?;
        let mut ops: Vec<WriteOp> = Vec::with_capacity(all_entries.len());
        for (key, _) in all_entries {
            ops.push(WriteOp::Delete {
                ns: Namespace::Strict,
                key,
            });
        }
        if !ops.is_empty() {
            engine.storage.batch_write(&ops).await?;
        }
    } else {
        let ids = {
            let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
            let ids = crdt.list_ids(collection);
            crdt.clear_collection(collection)
                .map_err(|e| LiteError::Storage {
                    detail: e.to_string(),
                })?;
            ids
        };
        let mut sparse = engine
            .sparse_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?;
        for id in &ids {
            sparse.remove_document_all_fields(collection, id);
        }
    }
    engine
        .fts_state
        .manager
        .lock()
        .map_err(|_| LiteError::LockPoisoned)?
        .drop_collection(collection);
    clear_spatial(engine, collection)?;
    crate::query::physical_visitor::clear_collection_indexes(&engine.vector_state, collection)
        .await?;
    Ok(truncated())
}

/// BulkUpdate: scan matching documents and apply field updates to all.
///
/// Lite does not yet evaluate residual scan filters — every document in the
/// collection receives the update. Callers that need filtered bulk updates
/// should compose `Scan` + per-row `PointUpdate` at the application layer.
pub async fn bulk_update<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    let field_updates = decode_literal_updates(updates)?;

    if is_strict(engine, collection) {
        let schema = strict_schema(engine, collection)?;
        let pk_idx = schema
            .columns
            .iter()
            .position(|c| c.primary_key)
            .ok_or_else(|| LiteError::BadRequest {
                detail: format!("strict collection '{collection}' has no primary key"),
            })?;
        let all_rows = engine.strict.list_rows(collection).await?;
        let mut affected_n: u64 = 0;
        let mut pks: Vec<Value> = Vec::with_capacity(all_rows.len());
        for row in &all_rows {
            let Some(pk) = row.get(pk_idx) else {
                continue;
            };
            if engine.strict.update(collection, pk, &field_updates).await? {
                affected_n += 1;
            }
            pks.push(pk.clone());
        }
        reindex_strict_rows(engine, collection, &pks).await?;
        Ok(affected(affected_n, "UPDATE"))
    } else {
        let loro_updates: Vec<(String, loro::LoroValue)> = field_updates
            .into_iter()
            .map(|(k, v)| (k, ndb_value_to_loro(v)))
            .collect();
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        let ids = crdt.list_ids(collection);
        let loro_slice: Vec<(&str, loro::LoroValue)> = loro_updates
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        let mut affected_n: u64 = 0;
        for id in &ids {
            crdt.upsert_deferred(collection, id, &loro_slice)
                .map_err(|e| LiteError::Storage {
                    detail: e.to_string(),
                })?;
            affected_n += 1;
        }
        crdt.flush_deltas().map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
        drop(crdt);
        reindex_documents(engine, collection, ids.iter().map(String::as_str))?;
        Ok(affected(affected_n, "UPDATE"))
    }
}

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
