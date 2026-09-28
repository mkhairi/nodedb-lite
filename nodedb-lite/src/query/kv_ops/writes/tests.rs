// SPDX-License-Identifier: Apache-2.0
//! KV point, multi-key and TTL writes.

use std::collections::HashMap;

use nodedb_query::expr::types::{BinaryOp, SqlExpr as QExpr};
use nodedb_types::value::Value;

use nodedb_physical::physical_plan::document::UpdateValue;

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
    let r =
        kv_insert_on_conflict_update(&engine, "kvoc", b"k", &row_bytes(&[("n", 99)]), 0, &updates)
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
