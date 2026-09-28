// SPDX-License-Identifier: Apache-2.0

//! SQL writes keep full-text search current.
//!
//! - A row written by SQL `INSERT` is text-searchable, per field and whole.
//! - SQL `UPDATE` replaces the updated field's terms and keeps the others.
//! - SQL `DELETE` leaves no terms behind.
//! - SQL `UPDATE` and `DELETE` with a non-key WHERE reach the rows it matches.
//! - A strict row is searchable with an empty field, across its text columns.
//! - A known collection with no text returns no rows, by trait and by SQL.
//! - A columnar row written by SQL is searchable until SQL deletes it.
//!
//! Run with:
//!   cargo nextest run -p nodedb-lite --test fts_sql_writes

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::text_search::TextSearchParams;
use nodedb_types::value::Value;

async fn open_test_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("open_in_memory");
    NodeDbLite::open(storage).await.expect("NodeDbLite::open")
}

async fn sql(db: &NodeDbLite<PagedbStorageMem>, statement: &str) {
    db.execute_sql(statement, &[])
        .await
        .unwrap_or_else(|e| panic!("SQL {statement:?}: {e}"));
}

/// Ids `text_search` returns for `query` on `field` of `collection`, sorted.
async fn search_ids(
    db: &NodeDbLite<PagedbStorageMem>,
    collection: &str,
    field: &str,
    query: &str,
) -> Vec<String> {
    let mut ids: Vec<String> = db
        .text_search(
            collection,
            field,
            query,
            10,
            TextSearchParams::default(),
            None,
        )
        .await
        .unwrap_or_else(|e| panic!("text_search {collection}/{field:?} for {query:?}: {e}"))
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    ids
}

/// Ids a SQL query returns in its `id` column, sorted.
async fn sql_ids(db: &NodeDbLite<PagedbStorageMem>, statement: &str) -> Vec<String> {
    let result = db
        .execute_sql(statement, &[])
        .await
        .unwrap_or_else(|e| panic!("SQL {statement:?}: {e}"));
    let Some(id_col) = result.columns.iter().position(|c| c == "id") else {
        assert!(
            result.rows.is_empty(),
            "rows without an id column: {result:?}"
        );
        return Vec::new();
    };
    let mut ids: Vec<String> = result
        .rows
        .iter()
        .map(|row| match row.get(id_col) {
            Some(Value::String(id)) => id.clone(),
            other => panic!("id column holds {other:?}"),
        })
        .collect();
    ids.sort();
    ids
}

async fn seed_articles(db: &NodeDbLite<PagedbStorageMem>) {
    sql(db, "CREATE COLLECTION articles").await;
    sql(
        db,
        "INSERT INTO articles (id, title, body) VALUES ('d1', 'rust guide', 'python tips')",
    )
    .await;
}

#[tokio::test]
async fn sql_insert_row_is_text_searchable() {
    let db = open_test_db().await;
    seed_articles(&db).await;

    assert_eq!(
        search_ids(&db, "articles", "title", "rust").await,
        vec!["d1"]
    );
    assert!(
        search_ids(&db, "articles", "title", "python")
            .await
            .is_empty()
    );
    assert_eq!(search_ids(&db, "articles", "", "python").await, vec!["d1"]);
    assert_eq!(
        sql_ids(
            &db,
            "SELECT id FROM articles WHERE text_match(body, 'python')"
        )
        .await,
        vec!["d1"]
    );
}

#[tokio::test]
async fn sql_update_replaces_text_terms() {
    let db = open_test_db().await;
    seed_articles(&db).await;
    sql(
        &db,
        "UPDATE articles SET title = 'go handbook' WHERE id = 'd1'",
    )
    .await;

    assert!(
        search_ids(&db, "articles", "title", "rust")
            .await
            .is_empty(),
        "the old title must stop matching"
    );
    assert!(search_ids(&db, "articles", "", "rust").await.is_empty());
    assert_eq!(
        search_ids(&db, "articles", "title", "handbook").await,
        vec!["d1"]
    );
    assert_eq!(
        search_ids(&db, "articles", "body", "python").await,
        vec!["d1"],
        "a field the UPDATE did not assign keeps its terms"
    );
}

#[tokio::test]
async fn sql_delete_removes_text_terms() {
    let db = open_test_db().await;
    seed_articles(&db).await;
    sql(&db, "DELETE FROM articles WHERE id = 'd1'").await;

    assert!(
        search_ids(&db, "articles", "title", "rust")
            .await
            .is_empty()
    );
    assert!(
        search_ids(&db, "articles", "body", "python")
            .await
            .is_empty()
    );
    assert!(search_ids(&db, "articles", "", "rust").await.is_empty());
}

#[tokio::test]
async fn strict_empty_field_search_matches_any_column() {
    let db = open_test_db().await;
    sql(
        &db,
        "CREATE COLLECTION shelf (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT,
            body TEXT
        ) WITH storage = 'strict'",
    )
    .await;
    sql(
        &db,
        "INSERT INTO shelf (id, title, body) VALUES ('r1', 'rust guide', 'python tips')",
    )
    .await;
    sql(
        &db,
        "INSERT INTO shelf (id, title, body) VALUES ('r2', 'haskell notes', 'ocaml tricks')",
    )
    .await;

    assert_eq!(search_ids(&db, "shelf", "", "rust").await, vec!["r1"]);
    assert_eq!(search_ids(&db, "shelf", "", "python").await, vec!["r1"]);
    assert_eq!(search_ids(&db, "shelf", "body", "ocaml").await, vec!["r2"]);

    sql(&db, "UPDATE shelf SET body = 'erlang' WHERE id = 'r1'").await;
    assert!(
        search_ids(&db, "shelf", "", "python").await.is_empty(),
        "an updated strict column must stop matching its old terms"
    );
    assert_eq!(search_ids(&db, "shelf", "", "erlang").await, vec!["r1"]);

    sql(&db, "DELETE FROM shelf WHERE id = 'r2'").await;
    assert!(search_ids(&db, "shelf", "", "haskell").await.is_empty());
}

#[tokio::test]
async fn text_search_on_empty_collection_returns_no_rows() {
    let db = open_test_db().await;
    sql(&db, "CREATE COLLECTION empty_docs").await;

    assert!(search_ids(&db, "empty_docs", "", "rust").await.is_empty());
    assert!(
        search_ids(&db, "empty_docs", "title", "rust")
            .await
            .is_empty()
    );
    assert!(
        sql_ids(
            &db,
            "SELECT id FROM empty_docs WHERE text_match(title, 'rust')"
        )
        .await
        .is_empty()
    );
}

#[tokio::test]
async fn sql_update_and_delete_by_non_key_predicate_reach_matching_rows() {
    let db = open_test_db().await;
    seed_articles(&db).await;
    sql(
        &db,
        "INSERT INTO articles (id, title, body) VALUES ('d2', 'haskell notes', 'ocaml tricks')",
    )
    .await;

    sql(
        &db,
        "UPDATE articles SET title = 'go handbook' WHERE body = 'python tips'",
    )
    .await;
    assert!(
        search_ids(&db, "articles", "title", "rust")
            .await
            .is_empty()
    );
    assert_eq!(
        search_ids(&db, "articles", "title", "handbook").await,
        vec!["d1"]
    );
    assert_eq!(
        search_ids(&db, "articles", "title", "haskell").await,
        vec!["d2"],
        "a row the WHERE does not match is untouched"
    );

    sql(&db, "DELETE FROM articles WHERE title = 'go handbook'").await;
    assert!(search_ids(&db, "articles", "", "python").await.is_empty());
    assert_eq!(search_ids(&db, "articles", "", "ocaml").await, vec!["d2"]);
}

#[tokio::test]
async fn columnar_sql_rows_are_text_searchable_until_deleted() {
    let db = open_test_db().await;
    sql(
        &db,
        "CREATE COLLECTION events (
            id TEXT NOT NULL PRIMARY KEY,
            message TEXT
        ) WITH storage = 'columnar'",
    )
    .await;
    sql(
        &db,
        "INSERT INTO events (id, message) VALUES ('e1', 'disk almost full')",
    )
    .await;

    assert_eq!(
        search_ids(&db, "events", "message", "disk").await,
        vec!["e1"]
    );
    assert_eq!(search_ids(&db, "events", "", "full").await, vec!["e1"]);

    sql(&db, "DELETE FROM events WHERE id = 'e1'").await;
    assert!(search_ids(&db, "events", "", "disk").await.is_empty());
}
