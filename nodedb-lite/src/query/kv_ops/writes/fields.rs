// SPDX-License-Identifier: Apache-2.0
//! Field-level and cross-key writes for the KV engine: field_set, transfer,
//! transfer_item.

use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::super::reads::{decode_value, encode_value, is_expired, kv_key};
use super::row_merge::{TransferRows, compute_transfer, merge_field_updates};

/// FieldSet: read-modify-write on named fields of a typed row.
///
/// The merge follows Origin's field-set rules: a raw single-`value` body is
/// a `TypeMismatch`, and update values are standard MessagePack.
///
/// `if_present` is the SQL `UPDATE` contract: an absent or expired key is
/// `UPDATE 0` and no row is created. `false` is the RESP hash-set contract,
/// which creates the row from nothing.
pub async fn kv_field_set<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    field_updates: &[(String, Vec<u8>)],
    if_present: bool,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let stored = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    let live = match stored {
        None => None,
        Some(raw) => {
            let (deadline, user_bytes) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
                detail: "corrupt KV entry".into(),
            })?;
            if is_expired(deadline) {
                None
            } else {
                Some((deadline, user_bytes.to_vec()))
            }
        }
    };
    if if_present && live.is_none() {
        return Ok(QueryResult {
            columns: vec![],
            rows: vec![],
            rows_affected: 0,
            command: Some("UPDATE".into()),
        });
    }
    let old_deadline = live.as_ref().map_or(0, |(deadline, _)| *deadline);
    let new_user_bytes = merge_field_updates(
        collection,
        live.as_ref().map(|(_, body)| body.as_slice()),
        field_updates,
    )?;
    let encoded = encode_value(old_deadline, &new_user_bytes);
    engine
        .storage
        .put(Namespace::Kv, &rkey, &encoded)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: Some("UPDATE".into()),
    })
}

/// Transfer: atomic fungible transfer between two keys in the same collection.
///
/// Reads source and dest, validates source.field >= amount, writes both back.
pub async fn kv_transfer<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    source_key: &[u8],
    dest_key: &[u8],
    field: &str,
    amount: f64,
) -> Result<QueryResult, LiteError> {
    let src_rkey = kv_key(collection, source_key);
    let dst_rkey = kv_key(collection, dest_key);

    let src_raw = engine
        .storage
        .get(Namespace::Kv, &src_rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("Transfer: source key not found in '{collection}'"),
        })?;

    let (src_deadline, src_user_bytes) =
        decode_value(&src_raw).ok_or_else(|| LiteError::Storage {
            detail: "corrupt KV entry: source".into(),
        })?;
    if is_expired(src_deadline) {
        return Err(LiteError::BadRequest {
            detail: "Transfer: source key is expired".into(),
        });
    }

    let dst_raw = engine
        .storage
        .get(Namespace::Kv, &dst_rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
    let dst_live = match dst_raw {
        None => None,
        Some(raw) => {
            let (deadline, user_bytes) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
                detail: "corrupt KV entry: dest".into(),
            })?;
            if is_expired(deadline) {
                None
            } else {
                Some((deadline, user_bytes.to_vec()))
            }
        }
    };
    let dst_deadline = dst_live.as_ref().map_or(0, |(deadline, _)| *deadline);
    let TransferRows {
        source: src_bytes,
        dest: dst_bytes,
    } = compute_transfer(
        collection,
        src_user_bytes,
        dst_live.as_ref().map(|(_, body)| body.as_slice()),
        field,
        amount,
    )?;

    let ops = vec![
        WriteOp::Put {
            ns: Namespace::Kv,
            key: src_rkey,
            value: encode_value(src_deadline, &src_bytes),
        },
        WriteOp::Put {
            ns: Namespace::Kv,
            key: dst_rkey,
            value: encode_value(dst_deadline, &dst_bytes),
        },
    ];
    engine
        .storage
        .batch_write(&ops)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 2,
        command: None,
    })
}

/// TransferItem: atomic non-fungible item transfer between two collections.
///
/// Deletes item from source collection and inserts at dest collection key.
pub async fn kv_transfer_item<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    source_collection: &str,
    dest_collection: &str,
    item_key: &[u8],
    dest_key: &[u8],
) -> Result<QueryResult, LiteError> {
    let src_rkey = kv_key(source_collection, item_key);
    let src_raw = engine
        .storage
        .get(Namespace::Kv, &src_rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!(
                "TransferItem: item not found in source collection '{source_collection}'"
            ),
        })?;

    let (src_deadline, src_user_bytes) =
        decode_value(&src_raw).ok_or_else(|| LiteError::Storage {
            detail: "corrupt KV entry: source item".into(),
        })?;
    if is_expired(src_deadline) {
        return Err(LiteError::BadRequest {
            detail: "TransferItem: source item is expired".into(),
        });
    }
    let item_bytes = src_user_bytes.to_vec();

    let dst_rkey = kv_key(dest_collection, dest_key);
    let ops = vec![
        WriteOp::Delete {
            ns: Namespace::Kv,
            key: src_rkey,
        },
        WriteOp::Put {
            ns: Namespace::Kv,
            key: dst_rkey,
            value: encode_value(0, &item_bytes),
        },
    ];
    engine
        .storage
        .batch_write(&ops)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::engine::test_engine;
    use crate::query::kv_ops::reads::kv_get;

    fn update(field: &str, value: i64) -> (String, Vec<u8>) {
        let bytes = nodedb_types::value_to_msgpack(&nodedb_types::value::Value::Integer(value))
            .expect("encode field");
        (field.to_string(), bytes)
    }

    #[tokio::test]
    async fn field_set_if_present_on_absent_key_is_a_no_op() {
        let engine = test_engine().await;
        let r = kv_field_set(&engine, "kvfs", b"missing", &[update("n", 1)], true)
            .await
            .expect("field set");
        assert_eq!(r.rows_affected, 0);
        let stored = kv_get(&engine, "kvfs", b"missing", None)
            .await
            .expect("get");
        assert!(stored.rows.is_empty(), "no row is created");
    }

    #[tokio::test]
    async fn field_set_without_if_present_creates_the_row() {
        let engine = test_engine().await;
        let r = kv_field_set(&engine, "kvfs", b"fresh", &[update("n", 1)], false)
            .await
            .expect("field set");
        assert_eq!(r.rows_affected, 1);
        let stored = kv_get(&engine, "kvfs", b"fresh", None).await.expect("get");
        assert_eq!(stored.rows.len(), 1);
    }

    /// SQL `UPDATE` on a raw single-`value` row is Origin's `TypeMismatch`,
    /// and the row keeps its value.
    #[tokio::test]
    async fn field_set_on_a_raw_row_is_a_type_mismatch() {
        let engine = test_engine().await;
        crate::query::kv_ops::writes::kv_put(&engine, "kvfs", b"raw", b"first", 0)
            .await
            .expect("seed");
        let err = kv_field_set(&engine, "kvfs", b"raw", &[update("value", 1)], true)
            .await
            .expect_err("a raw row is not a hash");
        assert!(matches!(err, LiteError::TypeMismatch { .. }), "{err:?}");
        let stored = kv_get(&engine, "kvfs", b"raw", None).await.expect("get");
        assert_eq!(
            stored.rows[0][1],
            nodedb_types::value::Value::Bytes(b"first".to_vec())
        );
    }

    #[tokio::test]
    async fn field_set_if_present_merges_into_an_existing_row() {
        let engine = test_engine().await;
        kv_field_set(&engine, "kvfs", b"k", &[update("a", 1)], false)
            .await
            .expect("seed");
        let r = kv_field_set(&engine, "kvfs", b"k", &[update("b", 2)], true)
            .await
            .expect("field set");
        assert_eq!(r.rows_affected, 1);
    }
}
