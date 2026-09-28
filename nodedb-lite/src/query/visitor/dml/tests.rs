// SPDX-License-Identifier: Apache-2.0

//! DML lowerings: INSERT SELECT, UPDATE FROM, MERGE.

use nodedb_sql::types::SqlValue;
use nodedb_sql::types_expr::BinaryOp;
use nodedb_types::columnar::{ColumnDef, ColumnType, StrictSchema};

use nodedb_sql::types::SqlPlan;
use nodedb_sql::types::query::EngineType;
use nodedb_sql::types_expr::SqlExpr;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;

use super::insert_select::lower_insert_select;
use crate::query::engine::test_engine;

fn schema() -> StrictSchema {
    StrictSchema {
        columns: vec![
            ColumnDef::required("id", ColumnType::String).with_primary_key(),
            ColumnDef::nullable("n", ColumnType::Int64),
        ],
        version: 1,
        dropped_columns: Vec::new(),
        bitemporal: false,
    }
}

fn scan(collection: &str) -> SqlPlan {
    SqlPlan::Scan {
        collection: collection.to_string(),
        alias: None,
        engine: EngineType::DocumentStrict,
        filters: Vec::new(),
        projection: Vec::new(),
        sort_keys: Vec::new(),
        limit: None,
        offset: 0,
        distinct: false,
        window_functions: Vec::new(),
        temporal: nodedb_sql::temporal::TemporalScope::default(),
    }
}

fn col(name: &str) -> SqlExpr {
    SqlExpr::Column {
        table: None,
        name: name.to_string(),
    }
}

async fn seed(engine: &LiteQueryEngine<crate::PagedbStorageMem>) {
    engine
        .strict
        .create_collection("src", schema())
        .await
        .expect("src");
    engine
        .strict
        .create_collection("dst", schema())
        .await
        .expect("dst");
    engine
        .strict
        .insert("src", &[Value::String("a".into()), Value::Integer(5)])
        .await
        .expect("row");
}

#[tokio::test]
async fn insert_select_column_map_shapes_each_target_row() {
    let engine = test_engine().await;
    seed(&engine).await;
    let column_map = vec![
        (
            "id".to_string(),
            SqlExpr::BinaryOp {
                left: Box::new(col("id")),
                op: BinaryOp::Concat,
                right: Box::new(SqlExpr::Literal(SqlValue::String("-copy".into()))),
            },
        ),
        (
            "n".to_string(),
            SqlExpr::BinaryOp {
                left: Box::new(col("n")),
                op: BinaryOp::Mul,
                right: Box::new(SqlExpr::Literal(SqlValue::Int(2))),
            },
        ),
    ];
    let r = lower_insert_select(&engine, "dst", &scan("src"), 0, &column_map)
        .expect("lower")
        .await
        .expect("run");
    assert_eq!(r.rows_affected, 1);
    let row = engine
        .strict
        .get("dst", &Value::String("a-copy".into()))
        .await
        .expect("get")
        .expect("row present");
    assert_eq!(row[1], Value::Integer(10));
}

#[tokio::test]
async fn insert_select_literal_null_into_pk_is_rejected_before_scanning() {
    let engine = test_engine().await;
    seed(&engine).await;
    let column_map = vec![
        ("id".to_string(), SqlExpr::Literal(SqlValue::Null)),
        ("n".to_string(), col("n")),
    ];
    let err = match lower_insert_select(&engine, "dst", &scan("src"), 0, &column_map) {
        Ok(_) => panic!("a literal NULL into the pk must be rejected"),
        Err(e) => e,
    };
    assert!(matches!(err, LiteError::NotNullViolation { .. }), "{err}");
}

#[tokio::test]
async fn insert_select_empty_map_copies_rows_unchanged() {
    let engine = test_engine().await;
    seed(&engine).await;
    let r = lower_insert_select(&engine, "dst", &scan("src"), 0, &[])
        .expect("lower")
        .await
        .expect("run");
    assert_eq!(r.rows_affected, 1);
    let row = engine
        .strict
        .get("dst", &Value::String("a".into()))
        .await
        .expect("get")
        .expect("row present");
    assert_eq!(row[1], Value::Integer(5));
}
