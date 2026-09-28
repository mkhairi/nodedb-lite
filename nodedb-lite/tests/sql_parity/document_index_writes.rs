// SPDX-License-Identifier: Apache-2.0

//! Document writes and secondary indexes: structured values, whole-statement
//! uniqueness, and bitemporal deletes. Lite-only checks; Origin is not
//! started.

use std::collections::HashMap;
use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::value::Value;

use crate::common::sql::open_lite;

type Db = Arc<NodeDbLite<PagedbStorageMem>>;

async fn exec(db: &Db, sql: &str) {
    db.execute_sql(sql, &[])
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
}

async fn exec_err(db: &Db, sql: &str) -> String {
    match db.execute_sql(sql, &[]).await {
        Ok(_) => panic!("expected an error for {sql}"),
        Err(e) => e.to_string(),
    }
}

async fn ids(db: &Db, sql: &str) -> Vec<String> {
    let result = db
        .execute_sql(sql, &[])
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut ids: Vec<String> = result
        .rows
        .iter()
        .map(|row| match row.first() {
            Some(Value::String(id)) => id.clone(),
            other => panic!("{sql}: id column is {other:?}"),
        })
        .collect();
    ids.sort();
    ids
}

fn doc(id: &str, fields: &[(&str, Value)]) -> Document {
    let mut d = Document::new(id);
    for (k, v) in fields {
        d.set(*k, v.clone());
    }
    d
}

fn text(s: &str) -> Value {
    Value::String(s.to_string())
}

fn list(items: &[&str]) -> Value {
    Value::Array(items.iter().map(|s| text(s)).collect())
}

async fn put(db: &Db, collection: &str, d: Document) {
    db.document_put(collection, d)
        .await
        .unwrap_or_else(|e| panic!("put into {collection}: {e}"));
}

#[tokio::test]
async fn array_index_matches_elements_written_by_document_put() {
    let db = open_lite().await;
    put(&db, "posts", doc("p1", &[("tags", list(&["rust", "db"]))])).await;
    exec(&db, "CREATE UNIQUE INDEX idx_tags ON posts (\"tags[]\")").await;

    let err = db
        .document_put("posts", doc("p2", &[("tags", list(&["go", "db"]))]))
        .await
        .expect_err("element 'db' is taken");
    assert!(err.to_string().contains("unique"), "{err}");
    put(&db, "posts", doc("p2", &[("tags", list(&["go", "web"]))])).await;
}

#[tokio::test]
async fn document_get_round_trips_nested_arrays_and_objects() {
    let db = open_lite().await;
    let address = Value::Object(HashMap::from([
        ("city".to_string(), text("Oslo")),
        (
            "geo".to_string(),
            Value::Array(vec![Value::Float(59.9), Value::Float(10.7)]),
        ),
    ]));
    let history = Value::Array(vec![
        Value::Object(HashMap::from([("n".to_string(), Value::Integer(1))])),
        Value::Array(vec![Value::Bool(true), Value::Null]),
    ]);
    let written = doc(
        "d1",
        &[
            ("address", address),
            ("history", history),
            // Text that looks like JSON is still text.
            ("raw", text("[1,2]")),
        ],
    );
    put(&db, "people", written.clone()).await;

    let read = db
        .document_get("people", "d1")
        .await
        .expect("get")
        .expect("row");
    assert_eq!(read.fields, written.fields);
}

/// `users` holding `u0` with a unique index on `email`.
async fn users_with_unique_email() -> Db {
    let db = open_lite().await;
    put(&db, "users", doc("u0", &[("email", text("zero@x"))])).await;
    exec(&db, "CREATE UNIQUE INDEX idx_email ON users (email)").await;
    db
}

#[tokio::test]
async fn multi_row_insert_with_duplicate_writes_nothing() {
    let db = users_with_unique_email().await;

    let err = exec_err(
        &db,
        "INSERT INTO users (id, email) VALUES ('u1', 'a@x'), ('u2', 'a@x')",
    )
    .await;
    assert!(err.contains("unique"), "{err}");
    let err = exec_err(
        &db,
        "INSERT INTO users (id, email) VALUES ('u3', 'b@x'), ('u4', 'zero@x')",
    )
    .await;
    assert!(err.contains("unique"), "{err}");
    let err = exec_err(
        &db,
        "INSERT INTO users (id, email) VALUES ('u5', 'c@x'), ('u5', 'd@x')",
    )
    .await;
    assert!(err.contains("unique"), "{err}");

    assert_eq!(ids(&db, "SELECT id FROM users").await, vec!["u0"]);
}

#[tokio::test]
async fn multi_row_update_violating_unique_writes_nothing() {
    let db = users_with_unique_email().await;
    exec(
        &db,
        "INSERT INTO users (id, email) VALUES ('u1', 'a@x'), ('u2', 'b@x')",
    )
    .await;

    let err = exec_err(&db, "UPDATE users SET email = 'same@x'").await;
    assert!(err.contains("unique"), "{err}");

    let q = |email: &str| format!("SELECT id FROM users WHERE email = '{email}'");
    assert!(ids(&db, &q("same@x")).await.is_empty());
    assert_eq!(ids(&db, &q("a@x")).await, vec!["u1"]);
    assert_eq!(ids(&db, &q("b@x")).await, vec!["u2"]);
    assert_eq!(ids(&db, &q("zero@x")).await, vec!["u0"]);
}

#[tokio::test]
async fn a_deleted_bitemporal_row_stays_out_of_its_indexes() {
    let db = open_lite().await;
    exec(&db, "CREATE COLLECTION bt WITH (bitemporal=true)").await;
    put(&db, "bt", doc("d1", &[("tag", text("x"))])).await;
    exec(&db, "CREATE UNIQUE INDEX idx_tag ON bt (tag)").await;
    db.document_delete("bt", "d1").await.expect("delete");

    // A vector merge changes the CRDT copy the delete left behind.
    db.vector_insert(
        "bt",
        "d1",
        &[1.0, 0.0],
        Some(doc("d1", &[("tag", text("x"))])),
    )
    .await
    .expect("vector merge");

    assert!(
        ids(&db, "SELECT id FROM bt WHERE tag = 'x'")
            .await
            .is_empty()
    );
    // The deleted row holds no value a unique check could count.
    put(&db, "bt", doc("d2", &[("tag", text("x"))])).await;
    assert_eq!(
        ids(&db, "SELECT id FROM bt WHERE tag = 'x'").await,
        vec!["d2"]
    );

    // Writing the row again makes it live, and indexed, again.
    put(&db, "bt", doc("d1", &[("tag", text("y"))])).await;
    assert_eq!(
        ids(&db, "SELECT id FROM bt WHERE tag = 'y'").await,
        vec!["d1"]
    );
}
