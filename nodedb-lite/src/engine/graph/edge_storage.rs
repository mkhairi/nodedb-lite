// SPDX-License-Identifier: Apache-2.0

use crate::error::LiteError;
use nodedb_types::value::Value;
use std::collections::HashMap;

/// Upsert edge properties into the Namespace::Graph storage table.
///
/// Key layout: `{collection}\x00{src}\x00{label}\x00{dst}`
pub(crate) fn edge_store_key(collection: &str, src: &str, label: &str, dst: &str) -> Vec<u8> {
    let mut k = collection.as_bytes().to_vec();
    k.push(0);
    k.extend_from_slice(src.as_bytes());
    k.push(0);
    k.extend_from_slice(label.as_bytes());
    k.push(0);
    k.extend_from_slice(dst.as_bytes());
    k
}

pub(crate) fn edge_to_value(
    collection: &str,
    src: &str,
    label: &str,
    dst: &str,
    props: &[u8],
) -> Result<Vec<u8>, LiteError> {
    let mut m = HashMap::new();
    m.insert(
        "collection".to_string(),
        Value::String(collection.to_string()),
    );
    m.insert("src".to_string(), Value::String(src.to_string()));
    m.insert("label".to_string(), Value::String(label.to_string()));
    m.insert("dst".to_string(), Value::String(dst.to_string()));
    if !props.is_empty() {
        // Properties are already msgpack bytes from the caller — store raw.
        m.insert("props".to_string(), Value::Bytes(props.to_vec()));
    }
    zerompk::to_msgpack_vec(&Value::Object(m)).map_err(|e| LiteError::Serialization {
        detail: e.to_string(),
    })
}
