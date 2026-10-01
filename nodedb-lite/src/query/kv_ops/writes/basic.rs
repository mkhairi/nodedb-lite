// SPDX-License-Identifier: Apache-2.0
//! Point writes for the KV engine: put and the insert variants.

use nodedb_physical::physical_plan::document::UpdateValue;
use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::kv_ops::body::{encode_kv_body, kv_body_columns};
use crate::query::on_conflict::apply_patch;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::super::reads::{decode_value, encode_value, is_expired, kv_key};

/// Store one encoded value, with the index entries it implies.
pub(super) async fn put_encoded<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    rkey: Vec<u8>,
    encoded: Vec<u8>,
) -> Result<(), LiteError> {
    engine
        .kv_local
        .commit(
            &*engine.storage,
            vec![WriteOp::Put {
                ns: Namespace::Kv,
                key: rkey,
                value: encoded,
            }],
        )
        .await
}

/// `kv_put`, tagged `INSERT` instead of `UPSERT`. Backs every insert variant
/// that falls through to an unconditional put once absence is confirmed.
async fn insert_via_put<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let mut result = kv_put(engine, collection, key, value, ttl_ms).await?;
    result.command = Some("INSERT".into());
    Ok(result)
}

/// Put: unconditional upsert.
pub async fn kv_put<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let deadline = if ttl_ms > 0 {
        crate::runtime::now_millis().saturating_add(ttl_ms)
    } else {
        0
    };
    put_encoded(
        engine,
        kv_key(collection, key),
        encode_value(deadline, value),
    )
    .await?;
    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: Some("UPSERT".into()),
    })
}

/// Insert: write only if key absent; error on duplicate.
pub async fn kv_insert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let existing = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
    if let Some(raw) = existing
        && let Some((deadline, _)) = decode_value(&raw)
        && !is_expired(deadline)
    {
        return Err(LiteError::BadRequest {
            detail: format!("unique_violation: key already exists in collection '{collection}'"),
        });
    }
    insert_via_put(engine, collection, key, value, ttl_ms).await
}

/// InsertIfAbsent: write if absent, silently no-op on duplicate.
pub async fn kv_insert_if_absent<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let existing = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
    if let Some(raw) = existing
        && let Some((deadline, _)) = decode_value(&raw)
        && !is_expired(deadline)
    {
        return Ok(QueryResult {
            columns: vec![],
            rows: vec![],
            rows_affected: 0,
            command: Some("INSERT".into()),
        });
    }
    insert_via_put(engine, collection, key, value, ttl_ms).await
}

/// InsertOnConflictUpdate: write if absent; on conflict apply field updates.
///
/// `updates` carries `UpdateValue`: a `Literal` decodes and overwrites the
/// field directly; an `Expr` (`n + 1`, `EXCLUDED.n`, ...) evaluates via
/// `query::on_conflict::apply_patch` against the existing stored row, with
/// `EXCLUDED.col` bound to `value` (the row that would have been inserted).
/// The merge follows Origin's: a raw single-`value` row reads as
/// `{"value": <text>}` and stays raw, so `SET value = EXCLUDED.value`
/// overwrites it, and assigning any other column to it is a `BadRequest`.
pub async fn kv_insert_on_conflict_update<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let existing = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    let raw = match existing {
        None => return insert_via_put(engine, collection, key, value, ttl_ms).await,
        Some(raw) => match decode_value(&raw) {
            None => return insert_via_put(engine, collection, key, value, ttl_ms).await,
            Some((deadline, _)) if is_expired(deadline) => {
                return insert_via_put(engine, collection, key, value, ttl_ms).await;
            }
            Some(_) => raw,
        },
    };

    let (old_deadline, old_user_bytes) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
        detail: "corrupt KV entry".into(),
    })?;

    // Origin's merge: both sides decode to rows (a raw body reads as
    // `{"value": <text>}`), and the merged row re-encodes in the stored
    // body's shape, so a raw row stays raw.
    let (mut map, shape) = kv_body_columns(old_user_bytes)?;
    let (excluded, _) = kv_body_columns(value)?;

    apply_patch(&mut map, updates, &excluded)?;

    let new_user_bytes = encode_kv_body(map, shape)?;

    let keep_deadline = if ttl_ms > 0 {
        crate::runtime::now_millis().saturating_add(ttl_ms)
    } else {
        old_deadline
    };

    put_encoded(engine, rkey, encode_value(keep_deadline, &new_user_bytes)).await?;

    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: Some("UPDATE".into()),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nodedb_query::expr::types::{BinaryOp, SqlExpr as QExpr};
    use nodedb_types::value::Value;

    use nodedb_physical::physical_plan::document::UpdateValue;

    use super::super::bulk::kv_truncate;
    use super::*;
    use crate::error::LiteError;
    use crate::query::engine::LiteQueryEngine;
    use crate::query::engine::test_engine;
    use crate::query::kv_ops::reads::kv_get;
    use crate::storage::engine::StorageEngine;

    /// Encode a `{field: value}` map body, the body a multi-column KV row
    /// stores.
    fn row_bytes(fields: &[(&str, i64)]) -> Vec<u8> {
        let map: HashMap<String, Value> = fields
            .iter()
            .map(|(k, v)| (k.to_string(), Value::Integer(*v)))
            .collect();
        crate::query::kv_ops::body::encode_kv_map(map).expect("encode row")
    }

    fn literal(n: i64) -> UpdateValue {
        UpdateValue::Literal(zerompk::to_msgpack_vec(&Value::Integer(n)).expect("encode literal"))
    }

    /// Read back a stored KV row's `n` field.
    async fn stored_n<S: StorageEngine>(engine: &LiteQueryEngine<S>, key: &[u8]) -> i64 {
        let r = kv_get(engine, "kvoc", key, None).await.expect("get");
        let Value::Bytes(bytes) = &r.rows[0][1] else {
            panic!("kv_get value column is not bytes");
        };
        let map = crate::query::kv_ops::body::decode_kv_map(bytes)
            .expect("decode row")
            .expect("map body");
        match map.get("n") {
            Some(Value::Integer(n)) => *n,
            other => panic!("expected integer 'n', got {other:?}"),
        }
    }

    #[tokio::test]
    async fn literal_assignment_overwrites_field() {
        let engine = test_engine().await;
        kv_put(&engine, "kvoc", b"k", &row_bytes(&[("n", 1)]), 0)
            .await
            .expect("seed");
        let updates = vec![("n".to_string(), literal(5))];
        let r = kv_insert_on_conflict_update(
            &engine,
            "kvoc",
            b"k",
            &row_bytes(&[("n", 99)]),
            0,
            &updates,
        )
        .await
        .expect("on conflict update");
        assert_eq!(r.rows_affected, 1);
        assert_eq!(stored_n(&engine, b"k").await, 5);
    }

    #[tokio::test]
    async fn expr_assignment_evaluates_against_existing_row() {
        let engine = test_engine().await;
        kv_put(&engine, "kvoc", b"k2", &row_bytes(&[("n", 1)]), 0)
            .await
            .expect("seed");
        let expr = QExpr::BinaryOp {
            left: Box::new(QExpr::Column("n".to_string())),
            op: BinaryOp::Add,
            right: Box::new(QExpr::Literal(Value::Integer(1))),
        };
        let updates = vec![("n".to_string(), UpdateValue::Expr(expr))];
        let r = kv_insert_on_conflict_update(
            &engine,
            "kvoc",
            b"k2",
            &row_bytes(&[("n", 99)]),
            0,
            &updates,
        )
        .await
        .expect("on conflict update");
        assert_eq!(r.rows_affected, 1);
        // `n + 1` against the existing row (1), not the incoming row (99).
        assert_eq!(stored_n(&engine, b"k2").await, 2);
    }

    #[tokio::test]
    async fn excluded_assignment_resolves_to_incoming_row() {
        let engine = test_engine().await;
        kv_put(&engine, "kvoc", b"k3", &row_bytes(&[("n", 1)]), 0)
            .await
            .expect("seed");
        let expr = QExpr::ExcludedColumn("n".to_string());
        let updates = vec![("n".to_string(), UpdateValue::Expr(expr))];
        let r = kv_insert_on_conflict_update(
            &engine,
            "kvoc",
            b"k3",
            &row_bytes(&[("n", 42)]),
            0,
            &updates,
        )
        .await
        .expect("on conflict update");
        assert_eq!(r.rows_affected, 1);
        assert_eq!(stored_n(&engine, b"k3").await, 42);
    }

    #[tokio::test]
    async fn absent_key_writes_the_incoming_row_unmerged() {
        let engine = test_engine().await;
        let updates = vec![("n".to_string(), literal(5))];
        let r = kv_insert_on_conflict_update(
            &engine,
            "kvoc",
            b"missing",
            &row_bytes(&[("n", 7)]),
            0,
            &updates,
        )
        .await
        .expect("insert");
        assert_eq!(r.rows_affected, 1);
        assert_eq!(stored_n(&engine, b"missing").await, 7);
    }

    #[tokio::test]
    async fn a_raw_row_overwritten_from_excluded_stays_raw() {
        let engine = test_engine().await;
        kv_put(&engine, "kvoc", b"r", b"first", 0)
            .await
            .expect("seed");
        let updates = vec![(
            "value".to_string(),
            UpdateValue::Expr(QExpr::ExcludedColumn("value".to_string())),
        )];
        kv_insert_on_conflict_update(&engine, "kvoc", b"r", b"second", 0, &updates)
            .await
            .expect("on conflict update");
        let r = kv_get(&engine, "kvoc", b"r", None).await.expect("get");
        assert_eq!(r.rows[0][1], Value::Bytes(b"second".to_vec()));
    }

    #[tokio::test]
    async fn a_raw_row_refuses_a_typed_column_assignment() {
        let engine = test_engine().await;
        kv_put(&engine, "kvoc", b"r2", b"first", 0)
            .await
            .expect("seed");
        let updates = vec![("n".to_string(), literal(1))];
        let err = kv_insert_on_conflict_update(&engine, "kvoc", b"r2", b"second", 0, &updates)
            .await
            .expect_err("a raw row cannot grow a typed column");
        assert!(matches!(err, LiteError::BadRequest { .. }), "{err:?}");
    }

    #[tokio::test]
    async fn truncate_removes_rows_and_index_entries() {
        use crate::query::kv_ops::indexes::kv_register_index;
        let engine = test_engine().await;
        for (k, n) in [("a", 1), ("b", 2), ("c", 3)] {
            kv_put(&engine, "kvt", k.as_bytes(), &row_bytes(&[("n", n)]), 0)
                .await
                .expect("seed");
        }
        kv_put(&engine, "kvoc", b"z", &row_bytes(&[("n", 9)]), 0)
            .await
            .expect("seed other");
        kv_register_index(&engine, "kvt", "n", true)
            .await
            .expect("register index");
        let def = engine
            .indexes
            .def_named(&crate::index::default_index_name("kvt", "n"))
            .expect("index declared");
        assert_eq!(
            engine.indexes.lookup_eq(&def, &Value::Integer(2)),
            vec![crate::index::key::doc_id_of_key(b"b")],
            "the build indexed the stored rows"
        );

        let r = kv_truncate(&engine, "kvt").await.expect("truncate");
        assert_eq!(r.rows_affected, 0);
        assert_eq!(r.command.as_deref(), Some("TRUNCATE"));
        for k in ["a", "b", "c"] {
            let got = kv_get(&engine, "kvt", k.as_bytes(), None)
                .await
                .expect("get");
            assert!(got.rows.is_empty(), "{k} survives truncate");
        }
        for n in 1..=3 {
            assert!(
                engine
                    .indexes
                    .lookup_eq(&def, &Value::Integer(n))
                    .is_empty(),
                "entry {n} survives truncate"
            );
        }
        assert!(
            engine.indexes.def_named(&def.name).is_some(),
            "truncate keeps the index"
        );
        assert_eq!(
            stored_n(&engine, b"z").await,
            9,
            "other collection untouched"
        );
    }
}
