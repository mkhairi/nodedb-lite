//! Per-row SQL functions and window output on the Lite engine.
//!
//! - A window column is computed by the window pass only. ORDER BY and the
//!   SELECT list read it as a column.
//! - `doc_get`, `doc_exists`, `doc_array_contains` and the vector distances
//!   evaluate per row in a projection and in WHERE.
//! - A vector dimension mismatch, a non-vector operand and a malformed
//!   JSONPath fail the statement, never a silent `NULL`.
//! - An index-owned search function in a row filter is refused at plan time.
//!
//! Run with:
//!   cargo nextest run -p nodedb-lite --test sql_row_functions

use std::collections::HashMap;
use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::value::Value;

async fn open_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("open_in_memory");
    NodeDbLite::open(storage).await.expect("NodeDbLite::open")
}

fn text(s: &str) -> Value {
    Value::String(s.into())
}

fn object(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(
        pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<_, _>>(),
    )
}

async fn rows(db: &Arc<NodeDbLite<PagedbStorageMem>>, sql: &str) -> Vec<Vec<Value>> {
    db.execute_sql(sql, &[])
        .await
        .unwrap_or_else(|e| panic!("expected Ok for SQL: {sql:?}\n  got: {e}"))
        .rows
}

async fn seed_ids(db: &Arc<NodeDbLite<PagedbStorageMem>>, collection: &str, ids: &[&str]) {
    for id in ids {
        let mut doc = Document::new(*id);
        doc.set("_seed", Value::Bool(true));
        db.document_put(collection, doc)
            .await
            .unwrap_or_else(|e| panic!("seed {collection}/{id}: {e}"));
    }
}

/// `ORDER BY` names the window alias: the sort runs after the window pass,
/// and the projection reads `rn` as a column.
#[tokio::test]
async fn order_by_a_window_alias_sorts_on_the_window_column() {
    let db = open_db().await;
    seed_ids(&db, "win_order", &["w1", "w2", "w3"]).await;

    let r = rows(
        &db,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM win_order ORDER BY rn",
    )
    .await;
    assert_eq!(
        r,
        vec![
            vec![text("w3"), Value::Integer(1)],
            vec![text("w2"), Value::Integer(2)],
            vec![text("w1"), Value::Integer(3)],
        ]
    );

    let r = rows(
        &db,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM win_order \
         ORDER BY rn DESC LIMIT 2",
    )
    .await;
    assert_eq!(
        r,
        vec![
            vec![text("w3"), Value::Integer(3)],
            vec![text("w2"), Value::Integer(2)],
        ]
    );
}

async fn seed_events(db: &Arc<NodeDbLite<PagedbStorageMem>>) {
    let events = [
        (
            "e1",
            object(vec![
                (
                    "user",
                    object(vec![("name", text("ada")), ("email", text("a@x"))]),
                ),
                ("tags", Value::Array(vec![text("important"), text("ops")])),
            ]),
        ),
        (
            "e2",
            object(vec![
                ("user", object(vec![("name", text("bob"))])),
                ("tags", Value::Array(vec![text("ops")])),
            ]),
        ),
    ];
    for (id, payload) in events {
        let mut doc = Document::new(id);
        doc.set("payload", payload);
        db.document_put("events", doc)
            .await
            .unwrap_or_else(|e| panic!("seed events/{id}: {e}"));
    }
}

#[tokio::test]
async fn doc_get_projects_a_nested_field() {
    let db = open_db().await;
    seed_events(&db).await;
    let r = rows(
        &db,
        "SELECT id, doc_get(payload, '$.user.name') AS name, \
         doc_get(payload, '$.user.email', 'none') AS email FROM events ORDER BY id",
    )
    .await;
    assert_eq!(
        r,
        vec![
            vec![text("e1"), text("ada"), text("a@x")],
            vec![text("e2"), text("bob"), text("none")],
        ]
    );
}

#[tokio::test]
async fn doc_exists_and_doc_array_contains_filter_rows() {
    let db = open_db().await;
    seed_events(&db).await;

    let r = rows(
        &db,
        "SELECT id FROM events WHERE doc_exists(payload, '$.user.email') ORDER BY id",
    )
    .await;
    assert_eq!(r, vec![vec![text("e1")]]);

    let r = rows(
        &db,
        "SELECT id FROM events WHERE doc_array_contains(payload, '$.tags', 'ops') ORDER BY id",
    )
    .await;
    assert_eq!(r, vec![vec![text("e1")], vec![text("e2")]]);

    let r = rows(
        &db,
        "SELECT id FROM events \
         WHERE doc_array_contains(payload, '$.tags', 'important') ORDER BY id",
    )
    .await;
    assert_eq!(r, vec![vec![text("e1")]]);
}

/// Without a vector-search ORDER BY, `vector_distance` evaluates per row.
#[tokio::test]
async fn vector_distance_in_a_projection_evaluates_per_row() {
    let db = open_db().await;
    let mut doc = Document::new("v1");
    doc.set(
        "emb",
        Value::Array(vec![Value::Float(3.0), Value::Float(4.0)]),
    );
    db.document_put("vecs", doc).await.expect("seed vecs/v1");

    let r = rows(
        &db,
        "SELECT id, vector_distance(emb, ARRAY[0.0, 0.0]) AS d FROM vecs",
    )
    .await;
    assert_eq!(r, vec![vec![text("v1"), Value::Float(25.0)]]);
}

async fn query_error(db: &Arc<NodeDbLite<PagedbStorageMem>>, sql: &str) -> String {
    db.execute_sql(sql, &[])
        .await
        .err()
        .unwrap_or_else(|| panic!("expected an error for SQL: {sql:?}"))
        .to_string()
}

#[tokio::test]
async fn vector_argument_faults_fail_the_statement() {
    let db = open_db().await;
    let mut good = Document::new("v1");
    good.set(
        "emb",
        Value::Array(vec![Value::Float(3.0), Value::Float(4.0)]),
    );
    db.document_put("vec_faults", good).await.expect("seed v1");

    let message = query_error(
        &db,
        "SELECT id, vector_distance(emb, ARRAY[0.0, 0.0, 0.0]) AS d FROM vec_faults",
    )
    .await;
    assert!(
        message.contains("vector dimension mismatch: expected 2, got 3"),
        "expected both dimensions, got: {message}"
    );

    let message = query_error(
        &db,
        "SELECT id, vector_distance(emb, 'not a vector') AS d FROM vec_faults",
    )
    .await;
    assert!(
        message.contains("argument 2") && message.contains("got string"),
        "expected argument 2 and its type, got: {message}"
    );
}

#[tokio::test]
async fn a_malformed_json_path_fails_the_statement() {
    let db = open_db().await;
    seed_events(&db).await;
    let message = query_error(
        &db,
        "SELECT id FROM events WHERE doc_array_contains(payload, '$.tags[', 'ops')",
    )
    .await;
    assert!(
        message.contains("invalid JSONPath"),
        "expected the JSONPath error, got: {message}"
    );
}

/// `bm25_score` reads the full-text index and has no per-row value, so a
/// comparison on it in WHERE is refused at plan time, rows or not.
#[tokio::test]
async fn a_search_score_in_a_row_filter_is_refused() {
    let db = open_db().await;
    seed_ids(&db, "fts_refuse", &["f1"]).await;
    let err = db
        .execute_sql(
            "SELECT id FROM fts_refuse WHERE bm25_score(body, 'rust') > 1.0",
            &[],
        )
        .await
        .expect_err("bm25_score in a row filter must be refused");
    let message = err.to_string();
    assert!(
        message.contains("bm25_score") && message.contains("search index"),
        "expected the search-function refusal, got: {message}"
    );
}
