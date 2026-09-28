// SPDX-License-Identifier: Apache-2.0
//! SQL reads of a KV collection, shaped as Origin shapes them.
//!
//! Every row is the `{key, value…}` row of `body::kv_read_row`:
//! - `key` is the entry's key as text;
//! - a raw body is a `value` column;
//! - a typed body contributes its columns.
//!
//! The result's columns are `key`, then every other column any row carries,
//! in name order. A row without a column holds `NULL` there. Expired
//! entries are invisible, because `kv_get` and `kv_scan` skip them.
//!
//! A read first commits the public KV API's buffered writes, so SQL sees
//! every write the public API has accepted.

use std::collections::{BTreeSet, HashMap};

use nodedb_sql::types_expr::SqlValue;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use super::body::kv_read_row;
use super::reads::{kv_get, kv_scan};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

/// The key bytes a SQL value names: strings and bytes as they are, integers
/// as decimal text, anything else as its debug text. A SQL insert stores
/// its key by the same rule.
pub(crate) fn kv_key_bytes(v: &SqlValue) -> Vec<u8> {
    match v {
        SqlValue::String(s) => s.as_bytes().to_vec(),
        SqlValue::Bytes(b) => b.clone(),
        SqlValue::Int(i) => i.to_string().into_bytes(),
        SqlValue::Float(_)
        | SqlValue::Decimal(_)
        | SqlValue::Bool(_)
        | SqlValue::Null
        | SqlValue::Timestamp(_)
        | SqlValue::Timestamptz(_)
        | SqlValue::Array(_) => format!("{v:?}").into_bytes(),
    }
}

/// Every live row of `collection`.
pub(crate) async fn kv_select_all<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    Ok(kv_select_entries(engine, collection).await?.0)
}

/// Every live row of `collection`, with each row's key bytes in row order.
pub(crate) async fn kv_select_entries<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<(QueryResult, Vec<Vec<u8>>), LiteError> {
    engine.kv_local.flush_to(&*engine.storage).await?;
    let scanned = kv_scan(engine, collection, b"", usize::MAX, None, None).await?;
    let entries = entries_of(scanned)?;
    let keys = entries.iter().map(|(key, _)| key.clone()).collect();
    Ok((shape_rows(&entries)?, keys))
}

/// The live row at `key` in `collection`: one row, or none.
pub(crate) async fn kv_select_key<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
) -> Result<QueryResult, LiteError> {
    engine.kv_local.flush_to(&*engine.storage).await?;
    let found = kv_get(engine, collection, key, None).await?;
    shape_rows(&entries_of(found)?)
}

/// One stored entry: its key bytes and its body bytes.
type KvEntry = (Vec<u8>, Vec<u8>);

/// The `(key, body)` pairs of a `kv_get` / `kv_scan` result.
fn entries_of(result: QueryResult) -> Result<Vec<KvEntry>, LiteError> {
    result
        .rows
        .into_iter()
        .map(|row| match <[Value; 2]>::try_from(row) {
            Ok([Value::Bytes(key), Value::Bytes(body)]) => Ok((key, body)),
            Ok(other) => Err(LiteError::Storage {
                detail: format!("KV read returned a non-bytes (key, value) pair: {other:?}"),
            }),
            Err(row) => Err(LiteError::Storage {
                detail: format!(
                    "KV read returned a {}-column row, not (key, value)",
                    row.len()
                ),
            }),
        })
        .collect()
}

/// Shape `(key, body)` pairs as Origin's KV read rows.
fn shape_rows(entries: &[KvEntry]) -> Result<QueryResult, LiteError> {
    let rows: Vec<HashMap<String, Value>> = entries
        .iter()
        .map(|(key, body)| kv_read_row(key, body))
        .collect::<Result<_, _>>()?;
    let others: BTreeSet<&String> = rows
        .iter()
        .flat_map(|row| row.keys())
        .filter(|name| name.as_str() != "key")
        .collect();
    let mut columns = Vec::with_capacity(others.len() + 1);
    columns.push("key".to_string());
    columns.extend(others.into_iter().cloned());
    let rows = rows
        .into_iter()
        .map(|mut row| {
            columns
                .iter()
                .map(|name| row.remove(name).unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    Ok(QueryResult {
        columns,
        rows,
        rows_affected: 0,
        command: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::engine::test_engine;
    use crate::query::kv_ops::body::encode_kv_map;
    use crate::query::kv_ops::writes::kv_put;

    #[tokio::test]
    async fn raw_and_typed_rows_share_one_rectangular_shape() {
        let engine = test_engine().await;
        kv_put(&engine, "kv", b"a", b"alpha", 0).await.expect("raw");
        let typed = encode_kv_map(HashMap::from([("n".to_string(), Value::Integer(3))]))
            .expect("typed body");
        kv_put(&engine, "kv", b"b", &typed, 0).await.expect("typed");
        kv_put(&engine, "other", b"c", b"x", 0)
            .await
            .expect("other");

        let result = kv_select_all(&engine, "kv").await.expect("select");
        assert_eq!(result.columns, vec!["key", "n", "value"]);
        assert_eq!(
            result.rows,
            vec![
                vec![
                    Value::String("a".into()),
                    Value::Null,
                    Value::String("alpha".into())
                ],
                vec![Value::String("b".into()), Value::Integer(3), Value::Null],
            ]
        );
    }

    #[tokio::test]
    async fn an_expired_row_is_invisible() {
        let engine = test_engine().await;
        kv_put(&engine, "kv", b"live", b"1", 0).await.expect("live");
        kv_put(&engine, "kv", b"gone", b"2", 1)
            .await
            .expect("short ttl");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        let all = kv_select_all(&engine, "kv").await.expect("select");
        assert_eq!(all.rows.len(), 1);
        assert_eq!(all.rows[0][0], Value::String("live".into()));
        let one = kv_select_key(&engine, "kv", b"gone").await.expect("get");
        assert!(one.rows.is_empty());
    }
}
