// SPDX-License-Identifier: Apache-2.0
//! Read operations for the Document engine physical visitor.

use std::collections::HashMap;

use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::value_utils::value_to_string;
use crate::storage::engine::StorageEngine;

use super::is_strict;

/// PointGet: fetch a single document by ID.
pub async fn point_get<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let columns = strict_columns(engine, collection);
        let pk = Value::String(document_id.to_string());
        match engine.strict.get(collection, &pk).await? {
            Some(values) => Ok(QueryResult {
                columns,
                rows: vec![values],
                rows_affected: 0,
                command: None,
            }),
            None => Ok(QueryResult::empty()),
        }
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        match crdt.read(collection, document_id) {
            Some(val) => {
                let bytes = crdt_value_to_msgpack(&val)?;
                drop(crdt);
                Ok(QueryResult {
                    columns: vec!["id".into(), "data".into()],
                    rows: vec![vec![
                        Value::String(document_id.to_string()),
                        Value::Bytes(bytes),
                    ]],
                    rows_affected: 0,
                    command: None,
                })
            }
            None => Ok(QueryResult::empty()),
        }
    }
}

/// Scan: full collection scan with limit/offset.
pub async fn scan<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    limit: usize,
    offset: usize,
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let columns = strict_columns(engine, collection);
        let all_rows = engine.strict.list_rows(collection).await?;
        let rows: Vec<Vec<Value>> = all_rows.into_iter().skip(offset).take(limit).collect();
        Ok(QueryResult {
            columns,
            rows,
            rows_affected: 0,
            command: None,
        })
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        let ids = crdt.list_ids(collection);
        let mut rows = Vec::with_capacity(ids.len().min(limit));
        for id in ids.iter().skip(offset).take(limit) {
            if let Some(val) = crdt.read(collection, id) {
                let bytes = crdt_value_to_msgpack(&val)?;
                rows.push(vec![Value::String(id.clone()), Value::Bytes(bytes)]);
            }
        }
        drop(crdt);
        Ok(QueryResult {
            columns: vec!["id".into(), "data".into()],
            rows,
            rows_affected: 0,
            command: None,
        })
    }
}

/// RangeScan: scan documents whose primary key lies within `[lower, upper]`.
///
/// For the strict path, materializes via `list_rows` once and filters by the
/// PK byte-range — avoiding the N+1 re-fetch that a `scan + get` composition
/// would incur (the strict-storage value encoding is internal to the strict
/// engine and not safe to decode here).
pub async fn range_scan<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    lower: Option<&[u8]>,
    upper: Option<&[u8]>,
    limit: usize,
) -> Result<QueryResult, LiteError> {
    if is_strict(engine, collection) {
        let schema = engine
            .strict
            .schema(collection)
            .ok_or_else(|| LiteError::collection_not_found("strict", collection))?;
        let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let pk_idx = schema
            .columns
            .iter()
            .position(|c| c.primary_key)
            .ok_or_else(|| LiteError::BadRequest {
                detail: format!("strict collection '{collection}' has no primary key"),
            })?;
        let all_rows = engine.strict.list_rows(collection).await?;
        let mut rows = Vec::new();
        for row in all_rows {
            let pk = row.get(pk_idx).ok_or_else(|| LiteError::Storage {
                detail: format!(
                    "strict collection '{collection}' row omits primary key at column {pk_idx}"
                ),
            })?;
            let pk_str = value_to_string(pk);
            let pk_bytes = pk_str.as_bytes();
            if let Some(lo) = lower
                && pk_bytes < lo
            {
                continue;
            }
            if let Some(hi) = upper
                && pk_bytes > hi
            {
                continue;
            }
            rows.push(row);
            if rows.len() >= limit {
                break;
            }
        }
        Ok(QueryResult {
            columns,
            rows,
            rows_affected: 0,
            command: None,
        })
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        let all_ids = crdt.list_ids(collection);
        let mut rows = Vec::new();
        for id in all_ids.iter() {
            if let Some(lo) = lower
                && id.as_bytes() < lo
            {
                continue;
            }
            if let Some(hi) = upper
                && id.as_bytes() > hi
            {
                continue;
            }
            if let Some(val) = crdt.read(collection, id) {
                let bytes = crdt_value_to_msgpack(&val)?;
                rows.push(vec![Value::String(id.clone()), Value::Bytes(bytes)]);
                if rows.len() >= limit {
                    break;
                }
            }
        }
        drop(crdt);
        Ok(QueryResult {
            columns: vec!["id".into(), "data".into()],
            rows,
            rows_affected: 0,
            command: None,
        })
    }
}

/// EstimateCount: exact document count for the collection.
pub async fn estimate_count<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    let count: u64 = if is_strict(engine, collection) {
        engine.strict.count(collection).await? as u64
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        crdt.list_ids(collection).len() as u64
    };
    Ok(QueryResult {
        columns: vec!["count".into()],
        rows: vec![vec![Value::Integer(count as i64)]],
        rows_affected: 0,
        command: None,
    })
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn strict_columns<S: StorageEngine>(engine: &LiteQueryEngine<S>, collection: &str) -> Vec<String> {
    engine
        .strict
        .schema(collection)
        .map(|s| s.columns.iter().map(|c| c.name.clone()).collect())
        .unwrap_or_default()
}

fn crdt_value_to_msgpack(val: &loro::LoroValue) -> Result<Vec<u8>, LiteError> {
    let ndb_val = loro_value_to_ndb_value(val);
    zerompk::to_msgpack_vec(&ndb_val).map_err(|e| LiteError::Serialization {
        detail: format!("serialize crdt value: {e}"),
    })
}

pub(crate) fn loro_value_to_ndb_value(v: &loro::LoroValue) -> Value {
    match v {
        loro::LoroValue::Null => Value::Null,
        loro::LoroValue::Bool(b) => Value::Bool(*b),
        loro::LoroValue::I64(n) => Value::Integer(*n),
        loro::LoroValue::Double(f) => Value::Float(*f),
        loro::LoroValue::String(s) => Value::String(s.to_string()),
        loro::LoroValue::Binary(b) => Value::Bytes(b.to_vec()),
        loro::LoroValue::Map(m) => {
            let mut map = HashMap::new();
            for (k, v) in m.iter() {
                map.insert(k.to_string(), loro_value_to_ndb_value(v));
            }
            Value::Object(map)
        }
        loro::LoroValue::List(arr) => {
            Value::Array(arr.iter().map(loro_value_to_ndb_value).collect())
        }
        _ => Value::Null,
    }
}

pub(super) fn msgpack_bytes_to_crdt_fields(
    bytes: &[u8],
) -> Result<Vec<(String, loro::LoroValue)>, LiteError> {
    let val: Value = zerompk::from_msgpack(bytes).map_err(|e| LiteError::Serialization {
        detail: format!("decode document bytes: {e}"),
    })?;
    match val {
        Value::Object(map) => Ok(map
            .into_iter()
            .map(|(k, v)| (k, ndb_value_to_loro(v)))
            .collect()),
        _ => Err(LiteError::BadRequest {
            detail: "document payload must be a msgpack-encoded object".into(),
        }),
    }
}

pub(super) fn ndb_value_to_loro(v: Value) -> loro::LoroValue {
    match v {
        Value::Null => loro::LoroValue::Null,
        Value::Bool(b) => loro::LoroValue::Bool(b),
        Value::Integer(n) => loro::LoroValue::I64(n),
        Value::Float(f) => loro::LoroValue::Double(f),
        Value::String(s) => loro::LoroValue::String(s.into()),
        Value::Bytes(b) => loro::LoroValue::Binary(b.into()),
        Value::Object(map) => {
            let loro_map: HashMap<String, loro::LoroValue> = map
                .into_iter()
                .map(|(k, v)| (k, ndb_value_to_loro(v)))
                .collect();
            loro::LoroValue::Map(loro_map.into())
        }
        Value::Array(arr) => {
            let list: Vec<loro::LoroValue> = arr.into_iter().map(ndb_value_to_loro).collect();
            loro::LoroValue::List(list.into())
        }
        _ => loro::LoroValue::Null,
    }
}
