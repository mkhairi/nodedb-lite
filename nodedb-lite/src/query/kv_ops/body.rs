// SPDX-License-Identifier: Apache-2.0
//! The body a Lite KV entry stores, in the one encoding Origin stores.
//!
//! A body takes one of two shapes, picked the way Origin's SQL lowering picks
//! them:
//! - a lone `value` column: raw scalar bytes, as `scalar_to_raw_bytes` writes
//!   them;
//! - any other column set: a standard MessagePack map of
//!   `nodedb_types::Value`, fields in key order, as `row_to_kv_body` writes
//!   it.
//!
//! Every Lite KV write encodes through this module, and every read that
//! needs typed columns decodes through it. The shared atomics
//! (`nodedb_physical::kv_atomic::compute`) read and write the same map
//! encoding, so a row stored by any path is readable by every other path,
//! and a row synced from Origin stores the bytes Origin stores.

use std::collections::HashMap;

use nodedb_query::msgpack_scan::{
    KvBodyError, KvBodyShape, kv_body_shape, kv_body_to_row, kv_row_msgpack, kv_row_to_body_fields,
    row_to_kv_body,
};
use nodedb_types::value::Value;

use crate::error::LiteError;

/// Encode `fields` as a body of `shape`.
pub(crate) fn encode_kv_body(
    fields: HashMap<String, Value>,
    shape: KvBodyShape,
) -> Result<Vec<u8>, LiteError> {
    row_to_kv_body(&Value::Object(fields), shape).map_err(kv_body_error)
}

/// The columns of `body` as a merge sees them, with the body's shape.
///
/// A map body yields its columns. A raw body yields `{"value": <text>}`, the
/// row every KV read presents for it.
pub(crate) fn kv_body_columns(
    body: &[u8],
) -> Result<(HashMap<String, Value>, KvBodyShape), LiteError> {
    let (row, shape) = kv_body_to_row(body).map_err(|e| kv_body_error(KvBodyError::from(e)))?;
    match row {
        Value::Object(columns) => Ok((columns, shape)),
        other => Err(kv_body_error(KvBodyError::RowNotObject {
            kind: other.type_name(),
        })),
    }
}

/// The `LiteError` for a body that cannot take its shape, as Origin
/// classifies it:
/// - a row that does not fit the body's shape is `BadRequest`;
/// - a body that does not decode or encode is `Serialization`.
pub(crate) fn kv_body_error(e: KvBodyError) -> LiteError {
    match e {
        KvBodyError::RowNotObject { .. }
        | KvBodyError::RawMissingValue
        | KvBodyError::RawExtraKeys { .. }
        | KvBodyError::RawNotScalar(_) => LiteError::BadRequest {
            detail: e.to_string(),
        },
        KvBodyError::Decode(_) | KvBodyError::Encode(_) => LiteError::Serialization {
            detail: e.to_string(),
        },
    }
}

/// Encode `fields` as a map body.
pub(crate) fn encode_kv_map(fields: HashMap<String, Value>) -> Result<Vec<u8>, LiteError> {
    encode_kv_body(fields, KvBodyShape::Map)
}

/// The typed columns of the map body `body`, or `None` for a raw body.
///
/// A map-shaped body that does not decode to a map is a `Serialization`
/// error.
pub(crate) fn decode_kv_map(body: &[u8]) -> Result<Option<HashMap<String, Value>>, LiteError> {
    if kv_body_shape(body) == KvBodyShape::Raw {
        return Ok(None);
    }
    match nodedb_types::value_from_msgpack(body) {
        Ok(Value::Object(columns)) => Ok(Some(columns)),
        Ok(other) => Err(LiteError::Serialization {
            detail: format!("KV body is a {}, not a row map", other.type_name()),
        }),
        Err(e) => Err(LiteError::Serialization {
            detail: format!("KV body does not decode: {e}"),
        }),
    }
}

/// The `{key, value…}` row every KV read presents for the entry at `key`.
///
/// This is Origin's read shaping, `kv_row_msgpack`, decoded:
/// - `key` is the key as text;
/// - a raw body is a `value` column holding its text;
/// - a map body contributes its typed columns.
pub(crate) fn kv_read_row(key: &[u8], body: &[u8]) -> Result<HashMap<String, Value>, LiteError> {
    let row = kv_row_msgpack(&String::from_utf8_lossy(key), body);
    match nodedb_types::value_from_msgpack(&row) {
        Ok(Value::Object(columns)) => Ok(columns),
        Ok(other) => Err(LiteError::Serialization {
            detail: format!("KV read row is a {}, not a row map", other.type_name()),
        }),
        Err(e) => Err(LiteError::Serialization {
            detail: format!("KV read row does not decode: {e}"),
        }),
    }
}

/// The body a local write stores for the `{key, value…}` row Origin sends
/// for the entry at `key`.
///
/// A payload that is not a msgpack row map is a `Serialization` error.
pub(crate) fn kv_body_from_row(key: &str, row: &[u8]) -> Result<Vec<u8>, LiteError> {
    let (fields, shape) =
        kv_row_to_body_fields(key, row).map_err(|e| LiteError::Serialization {
            detail: format!("KV row for key '{key}' is not a {{key, value}} row map: {e}"),
        })?;
    encode_kv_body(fields, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_raw_row_stores_the_raw_value() {
        let row = kv_row_msgpack("k1", b"v1");
        assert_eq!(kv_body_from_row("k1", &row).expect("body"), b"v1".to_vec());
    }

    #[test]
    fn a_typed_row_stores_the_body_origin_stores() {
        let mut fields = HashMap::new();
        fields.insert("n".to_string(), Value::Integer(7));
        let origin_body =
            row_to_kv_body(&Value::Object(fields.clone()), KvBodyShape::Map).expect("origin body");
        let row = kv_row_msgpack("k2", &origin_body);

        let stored = kv_body_from_row("k2", &row).expect("body");
        assert_eq!(stored, origin_body, "Lite stores Origin's exact bytes");
        assert_eq!(decode_kv_map(&stored).expect("decode"), Some(fields));
    }

    #[test]
    fn a_raw_body_decodes_as_no_columns() {
        assert_eq!(decode_kv_map(b"12").expect("decode"), None);
    }

    #[test]
    fn a_payload_that_is_not_a_row_map_is_a_serialization_error() {
        let scalar = nodedb_types::value_to_msgpack(&Value::Integer(118)).expect("encode");
        assert!(matches!(
            kv_body_from_row("k", &scalar),
            Err(LiteError::Serialization { .. })
        ));
        assert!(matches!(
            kv_body_from_row("k", b"v1"),
            Err(LiteError::Serialization { .. })
        ));
    }
}
