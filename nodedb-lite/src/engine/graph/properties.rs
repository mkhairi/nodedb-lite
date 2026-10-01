// SPDX-License-Identifier: Apache-2.0

//! Bounded SQL edge properties and physical edge identities.

use std::collections::HashMap;

use nodedb_types::{
    Namespace,
    error::{NodeDbError, NodeDbResult},
    value::Value,
};

use crate::engine::graph::traversal::DEFAULT_MAX_VISITED;
use crate::storage::engine::{KvPair, PrefixScanLimit, StorageEngine};

pub(crate) type EdgeKey = (String, String, String);
pub(crate) type Properties = HashMap<String, Value>;
pub(crate) type PropertyMap = HashMap<EdgeKey, Properties>;

/// Serialized SQL edge records are limited to 64 MiB before decoding.
pub(crate) const MAX_PROPERTY_BYTES: usize = 64 * 1024 * 1024;

/// Reject SQL property collections exceeding either storage budget before decoding.
pub(crate) async fn load_sql_properties<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> NodeDbResult<PropertyMap> {
    let mut prefix = collection.as_bytes().to_vec();
    prefix.push(0);
    let scan = storage
        .scan_prefix_budgeted(
            Namespace::Graph,
            &prefix,
            DEFAULT_MAX_VISITED,
            MAX_PROPERTY_BYTES,
        )
        .await
        .map_err(NodeDbError::storage)?;
    if let Some(limit) = scan.limit {
        let budget = match limit {
            PrefixScanLimit::Records => format!("{DEFAULT_MAX_VISITED} records"),
            PrefixScanLimit::Bytes => format!("{MAX_PROPERTY_BYTES} serialized bytes"),
        };
        return Err(NodeDbError::program_limit_exceeded(format!(
            "SQL edge properties exceed {budget} in '{collection}': reduce the collection"
        )));
    }
    decode_properties(collection, scan.entries)
}

fn decode_properties(collection: &str, rows: Vec<KvPair>) -> NodeDbResult<PropertyMap> {
    if rows.len() > DEFAULT_MAX_VISITED {
        return Err(NodeDbError::program_limit_exceeded(format!(
            "SQL edge properties exceed {DEFAULT_MAX_VISITED} records in '{collection}': reduce the collection"
        )));
    }
    let mut bytes = 0usize;
    for (key, value) in &rows {
        bytes = bytes.saturating_add(key.len()).saturating_add(value.len());
        if bytes > MAX_PROPERTY_BYTES {
            return Err(NodeDbError::program_limit_exceeded(format!(
                "SQL edge properties exceed {MAX_PROPERTY_BYTES} serialized bytes in '{collection}': reduce the collection"
            )));
        }
    }
    let mut properties = HashMap::with_capacity(rows.len());
    for (key, value) in rows {
        let parts: Vec<&[u8]> = key.split(|b| *b == 0).collect();
        if parts.len() != 4 || parts[0] != collection.as_bytes() {
            return Err(invalid_record(collection, &key, "invalid edge key"));
        }
        let mut identity = Vec::with_capacity(3);
        for part in &parts[1..] {
            let text = std::str::from_utf8(part)
                .map_err(|_| invalid_record(collection, &key, "non-UTF-8 edge identity"))?;
            identity.push(text.to_owned());
        }
        let outer: Value = zerompk::from_msgpack(&value)
            .map_err(|e| invalid_record(collection, &key, &e.to_string()))?;
        let Value::Object(outer) = outer else {
            return Err(invalid_record(collection, &key, "expected edge object"));
        };
        for (name, expected) in [
            ("collection", collection),
            ("src", identity[0].as_str()),
            ("label", identity[1].as_str()),
            ("dst", identity[2].as_str()),
        ] {
            if outer.get(name).and_then(Value::as_str) != Some(expected) {
                return Err(invalid_record(
                    collection,
                    &key,
                    "edge identity differs from key",
                ));
            }
        }
        let props = match outer.get("props") {
            None => HashMap::new(),
            Some(Value::Bytes(bytes)) => {
                let inner: Value = zerompk::from_msgpack(bytes)
                    .map_err(|e| invalid_record(collection, &key, &e.to_string()))?;
                let Value::Object(props) = inner else {
                    return Err(invalid_record(
                        collection,
                        &key,
                        "expected properties object",
                    ));
                };
                props
            }
            Some(_) => {
                return Err(invalid_record(
                    collection,
                    &key,
                    "expected properties bytes",
                ));
            }
        };
        let mut identity = identity.into_iter();
        let (Some(src), Some(label), Some(dst)) =
            (identity.next(), identity.next(), identity.next())
        else {
            return Err(invalid_record(collection, &key, "invalid edge identity"));
        };
        properties.insert((src, label, dst), props);
    }
    Ok(properties)
}

fn invalid_record(collection: &str, key: &[u8], detail: &str) -> NodeDbError {
    NodeDbError::storage(format!(
        "invalid SQL edge record in '{collection}' at {key:?}: {detail}: recreate the edge"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::edge_storage::{edge_store_key, edge_to_value};

    fn row(src: &str, properties: Value) -> KvPair {
        let props = zerompk::to_msgpack_vec(&properties).unwrap();
        (
            edge_store_key("graph", src, "LINK", "b"),
            edge_to_value("graph", src, "LINK", "b", &props).unwrap(),
        )
    }

    #[test]
    fn properties_decode_objects_and_empty_rows() {
        let properties = HashMap::from([("score".to_owned(), Value::Integer(9))]);
        let rows = vec![
            row("a", Value::Object(properties.clone())),
            (
                edge_store_key("graph", "empty", "LINK", "b"),
                edge_to_value("graph", "empty", "LINK", "b", &[]).unwrap(),
            ),
        ];
        let decoded = decode_properties("graph", rows).unwrap();
        assert_eq!(
            decoded.get(&("a".into(), "LINK".into(), "b".into())),
            Some(&properties)
        );
        assert!(
            decoded
                .get(&("empty".into(), "LINK".into(), "b".into()))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn malformed_records_never_return_partial_properties() {
        let valid = row("a", Value::Object(HashMap::new()));
        for invalid in [
            (b"graph\0a".to_vec(), valid.1.clone()),
            (b"other\0a\0LINK\0b".to_vec(), valid.1.clone()),
            (valid.0.clone(), vec![0xc1]),
            row("scalar", Value::Integer(3)),
            (
                edge_store_key("graph", "different", "LINK", "b"),
                valid.1.clone(),
            ),
        ] {
            assert!(decode_properties("graph", vec![valid.clone(), invalid]).is_err());
        }
    }

    #[test]
    fn serialized_bounds_reject_rows_before_decoding() {
        let rows = vec![(Vec::new(), Vec::new()); DEFAULT_MAX_VISITED + 1];
        let count_error = decode_properties("graph", rows).unwrap_err();
        assert!(count_error.to_string().contains("records"));
        let byte_error =
            decode_properties("graph", vec![(vec![0], vec![0; MAX_PROPERTY_BYTES])]).unwrap_err();
        assert!(byte_error.to_string().contains("serialized bytes"));
    }
}
