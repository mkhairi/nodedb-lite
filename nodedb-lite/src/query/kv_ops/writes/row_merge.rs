// SPDX-License-Identifier: Apache-2.0
//! Pure row computations for KV field set and transfer.
//!
//! Both follow Origin's KV handlers rule for rule, so Lite and Origin store
//! the same bytes and refuse the same writes:
//! - A raw body (the single-`value` form) is not a set of typed columns. A
//!   field set or transfer against it is a `TypeMismatch`, as Redis refuses
//!   `HSET` on a string key. It is never replaced by a map.
//! - Field values arrive as standard MessagePack. Empty bytes set `NULL`.
//! - The result is a map body in the canonical encoding.

use std::collections::HashMap;

use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::kv_ops::body::{decode_kv_map, encode_kv_map};

/// The typed columns of `body`, refusing a raw body with `TypeMismatch`.
fn typed_columns(
    collection: &str,
    body: &[u8],
    what: &str,
) -> Result<HashMap<String, Value>, LiteError> {
    decode_kv_map(body)?.ok_or_else(|| LiteError::TypeMismatch {
        collection: collection.to_owned(),
        detail: format!("{what} holds a bare value, not a hash; it requires typed columns"),
    })
}

/// Merge `updates` into `current` (an absent key starts from no columns)
/// and encode the merged row.
pub(crate) fn merge_field_updates(
    collection: &str,
    current: Option<&[u8]>,
    updates: &[(String, Vec<u8>)],
) -> Result<Vec<u8>, LiteError> {
    let mut row = match current {
        None => HashMap::new(),
        Some(body) => typed_columns(collection, body, "key")?,
    };
    for (field, bytes) in updates {
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            nodedb_types::value_from_msgpack(bytes).map_err(|e| LiteError::Serialization {
                detail: format!("field set '{field}': msgpack decode: {e}"),
            })?
        };
        row.insert(field.clone(), value);
    }
    encode_kv_map(row)
}

/// The two rows a fungible transfer writes.
#[derive(Debug)]
pub(crate) struct TransferRows {
    pub source: Vec<u8>,
    pub dest: Vec<u8>,
}

/// Move `amount` of the numeric `field` from `source` to `dest`.
///
/// An absent `dest` becomes a row holding only `field`. A `dest` without
/// `field` starts at 0. A source without a numeric `field`, a non-numeric
/// `field` on either side, or a raw body on either side is a
/// `TypeMismatch`. A source balance below `amount` is refused.
pub(crate) fn compute_transfer(
    collection: &str,
    source: &[u8],
    dest: Option<&[u8]>,
    field: &str,
    amount: f64,
) -> Result<TransferRows, LiteError> {
    let mut source_row = typed_columns(collection, source, "source")?;
    let have =
        numeric_field(collection, &source_row, field)?.ok_or_else(|| LiteError::TypeMismatch {
            collection: collection.to_owned(),
            detail: format!("field '{field}' is not numeric or missing"),
        })?;
    if have < amount {
        return Err(LiteError::BadRequest {
            detail: format!(
                "Transfer: insufficient balance in '{collection}': source has {have}, need {amount}"
            ),
        });
    }
    let mut dest_row = match dest.filter(|b| !b.is_empty()) {
        None => HashMap::with_capacity(1),
        Some(body) => typed_columns(collection, body, "destination")?,
    };
    let dest_balance = numeric_field(collection, &dest_row, field)?.unwrap_or(0.0);

    source_row.insert(field.to_owned(), numeric_value(have - amount));
    dest_row.insert(field.to_owned(), numeric_value(dest_balance + amount));
    Ok(TransferRows {
        source: encode_kv_map(source_row)?,
        dest: encode_kv_map(dest_row)?,
    })
}

/// `field` as f64: `None` when absent, `TypeMismatch` when not numeric.
fn numeric_field(
    collection: &str,
    row: &HashMap<String, Value>,
    field: &str,
) -> Result<Option<f64>, LiteError> {
    match row.get(field) {
        None => Ok(None),
        Some(Value::Float(f)) => Ok(Some(*f)),
        Some(Value::Integer(i)) => Ok(Some(*i as f64)),
        Some(other) => Err(LiteError::TypeMismatch {
            collection: collection.to_owned(),
            detail: format!("field '{field}' is {}, not numeric", other.type_name()),
        }),
    }
}

/// A whole-number balance stays an integer. Anything else is a float.
fn numeric_value(v: f64) -> Value {
    if v.fract() == 0.0 && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
        Value::Integer(v as i64)
    } else {
        Value::Float(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(v: i64) -> Vec<u8> {
        nodedb_types::value_to_msgpack(&Value::Integer(v)).expect("encode")
    }

    fn row(fields: &[(&str, i64)]) -> Vec<u8> {
        encode_kv_map(
            fields
                .iter()
                .map(|(k, v)| (k.to_string(), Value::Integer(*v)))
                .collect(),
        )
        .expect("encode row")
    }

    fn column(body: &[u8], name: &str) -> Option<Value> {
        decode_kv_map(body)
            .expect("decode")
            .expect("map body")
            .get(name)
            .cloned()
    }

    #[test]
    fn a_field_set_merges_into_a_typed_row() {
        let merged = merge_field_updates("c", Some(&row(&[("a", 1)])), &[("b".into(), int(2))])
            .expect("merge");
        assert_eq!(column(&merged, "a"), Some(Value::Integer(1)));
        assert_eq!(column(&merged, "b"), Some(Value::Integer(2)));
    }

    #[test]
    fn a_field_set_on_a_raw_body_is_a_type_mismatch() {
        for body in [b"first".as_slice(), b"1".as_slice()] {
            let err = merge_field_updates("c", Some(body), &[("f".into(), int(1))])
                .expect_err("a raw body is not a hash");
            assert!(matches!(err, LiteError::TypeMismatch { .. }), "{err:?}");
        }
    }

    #[test]
    fn empty_update_bytes_set_null() {
        let merged = merge_field_updates("c", None, &[("f".into(), Vec::new())]).expect("merge");
        assert_eq!(column(&merged, "f"), Some(Value::Null));
    }

    #[test]
    fn a_transfer_moves_an_integer_balance_and_keeps_it_integral() {
        let rows = compute_transfer("c", &row(&[("bal", 10)]), None, "bal", 4.0).expect("move");
        assert_eq!(column(&rows.source, "bal"), Some(Value::Integer(6)));
        assert_eq!(column(&rows.dest, "bal"), Some(Value::Integer(4)));
    }

    #[test]
    fn a_transfer_refuses_a_raw_side() {
        let err = compute_transfer("c", b"10", None, "bal", 1.0).expect_err("raw source");
        assert!(matches!(err, LiteError::TypeMismatch { .. }));
        let err = compute_transfer("c", &row(&[("bal", 10)]), Some(b"5"), "bal", 1.0)
            .expect_err("raw destination");
        assert!(matches!(err, LiteError::TypeMismatch { .. }));
    }

    #[test]
    fn a_transfer_refuses_an_insufficient_balance() {
        let err =
            compute_transfer("c", &row(&[("bal", 1)]), None, "bal", 5.0).expect_err("too little");
        assert!(matches!(err, LiteError::BadRequest { .. }));
    }
}
