// SPDX-License-Identifier: Apache-2.0
//! Shared helpers for the Document engine write handlers: result shapes,
//! strict-schema lookup, and payload decoding.

use std::collections::HashMap;

use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::writes::UpdateValue;

pub(super) fn affected(n: u64, command: &'static str) -> QueryResult {
    QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: n,
        command: Some(command.into()),
    }
}

pub(super) fn strict_schema<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<nodedb_types::columnar::StrictSchema, LiteError> {
    engine
        .strict
        .schema(collection)
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("strict collection '{collection}' does not exist"),
        })
}

/// Decode msgpack document bytes into `(field_name, Value)` pairs.
pub(super) fn decode_strict_fields(value_bytes: &[u8]) -> Result<Vec<(String, Value)>, LiteError> {
    let val: Value = zerompk::from_msgpack(value_bytes).map_err(|e| LiteError::Serialization {
        detail: format!("decode strict document: {e}"),
    })?;
    match val {
        Value::Object(map) => Ok(map.into_iter().collect()),
        _ => Err(LiteError::BadRequest {
            detail: "strict document payload must be a msgpack-encoded object".into(),
        }),
    }
}

/// Build a `Vec<Value>` in schema column order from a field map.
pub(super) fn fields_to_values(
    fields: &[(String, Value)],
    columns: &[nodedb_types::columnar::ColumnDef],
) -> Vec<Value> {
    let map: HashMap<&str, &Value> = fields.iter().map(|(k, v)| (k.as_str(), v)).collect();
    columns
        .iter()
        .map(|c| {
            map.get(c.name.as_str())
                .copied()
                .cloned()
                .unwrap_or(Value::Null)
        })
        .collect()
}

/// Decode literal-only update values; non-literal `UpdateValue::Expr` arms
/// are ignored because the Lite executor has no expression evaluator.
pub(super) fn decode_literal_updates(
    updates: &[(String, UpdateValue)],
) -> Result<HashMap<String, Value>, LiteError> {
    let mut field_updates: HashMap<String, Value> = HashMap::new();
    for (field, update_val) in updates {
        if let UpdateValue::Literal(bytes) = update_val {
            let val: Value =
                zerompk::from_msgpack(bytes).map_err(|e| LiteError::Serialization {
                    detail: format!("decode update literal for '{field}': {e}"),
                })?;
            field_updates.insert(field.clone(), val);
        }
    }
    Ok(field_updates)
}
