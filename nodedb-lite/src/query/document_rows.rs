// SPDX-License-Identifier: Apache-2.0

//! The row shape a SQL read of a schemaless collection answers with: the
//! document id and the document as one JSON-string `document` column. Scans,
//! point gets and index reads all build their rows here, so a query answers
//! in the same shape whichever path the planner picked.

use std::sync::Mutex;

use nodedb_types::value::Value;

use crate::engine::crdt::CrdtEngine;
use crate::error::LiteError;

/// Documents a scan reads per hold of the CRDT lock.
pub(crate) const SCAN_CHUNK: usize = 128;

/// The current documents of `ids` in `collection`, read under one hold of
/// the CRDT lock. An id with no live document is skipped.
///
/// Scans call this once per [`SCAN_CHUNK`] ids, so a writer or a point read
/// waits for one chunk rather than the whole scan. Row building and WHERE
/// evaluation run after the lock is released. The cost is snapshot
/// isolation: a scan sees each document as it was when its chunk was read,
/// so documents written while the scan runs can be seen in mixed states.
pub(crate) fn read_chunk(
    crdt: &Mutex<CrdtEngine>,
    collection: &str,
    ids: &[String],
) -> Result<Vec<(String, loro::LoroValue)>, LiteError> {
    let crdt = crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
    Ok(ids
        .iter()
        .filter_map(|id| crdt.read(collection, id).map(|doc| (id.clone(), doc)))
        .collect())
}

/// Column header of a schemaless document read.
pub(crate) fn columns() -> Vec<String> {
    vec!["id".into(), "document".into()]
}

/// One row from a document held in the CRDT store.
pub(crate) fn crdt_row(id: &str, doc: &loro::LoroValue) -> Vec<Value> {
    let json = sonic_rs::to_string(&loro_value_to_json(doc)).unwrap_or_default();
    vec![Value::String(id.to_string()), Value::String(json)]
}

/// One row from a document value.
pub(crate) fn value_row(id: &str, doc: Value) -> Vec<Value> {
    let json = match doc {
        Value::Object(_) => {
            sonic_rs::to_string(&value_to_json(doc)).unwrap_or_else(|_| "{}".to_owned())
        }
        _ => "{}".to_owned(),
    };
    vec![Value::String(id.to_string()), Value::String(json)]
}

/// One row from the MessagePack body of a bitemporal version. A body that is
/// empty or not a map reads as the empty document.
pub(crate) fn body_row(id: &str, body: &[u8]) -> Vec<Value> {
    let doc = if body.is_empty() {
        Value::Null
    } else {
        nodedb_types::json_msgpack::value_from_msgpack(body).unwrap_or(Value::Null)
    };
    value_row(id, doc)
}

fn value_to_json(v: Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(b),
        Value::Integer(n) => serde_json::json!(n),
        Value::Float(f) => serde_json::json!(f),
        Value::String(s) => serde_json::Value::String(s),
        Value::Array(arr) => serde_json::Value::Array(arr.into_iter().map(value_to_json).collect()),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, val) in map {
                out.insert(k, value_to_json(val));
            }
            serde_json::Value::Object(out)
        }
        _ => serde_json::Value::Null,
    }
}

fn loro_value_to_json(v: &loro::LoroValue) -> serde_json::Value {
    match v {
        loro::LoroValue::Null => serde_json::Value::Null,
        loro::LoroValue::Bool(b) => serde_json::Value::Bool(*b),
        loro::LoroValue::I64(n) => serde_json::json!(*n),
        loro::LoroValue::Double(f) => serde_json::json!(*f),
        loro::LoroValue::String(s) => serde_json::Value::String(s.to_string()),
        loro::LoroValue::Map(m) => {
            let mut obj = serde_json::Map::new();
            for (k, val) in m.iter() {
                obj.insert(k.to_string(), loro_value_to_json(val));
            }
            serde_json::Value::Object(obj)
        }
        loro::LoroValue::List(arr) => {
            serde_json::Value::Array(arr.iter().map(loro_value_to_json).collect())
        }
        _ => serde_json::Value::Null,
    }
}
