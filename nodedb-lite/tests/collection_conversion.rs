// SPDX-License-Identifier: Apache-2.0

//! `CONVERT COLLECTION` is all or nothing.
//!
//! - A document the target schema refuses fails the conversion, and every
//!   document stays where it was.
//! - A conversion that succeeds moves every document, and the rows stay
//!   text-searchable.
//!
//! Run with:
//!   cargo nextest run -p nodedb-lite --test collection_conversion

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::text_search::TextSearchParams;

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

async fn seed(db: &NodeDbLite<PagedbStorageMem>) {
    sql(db, "CREATE COLLECTION notes").await;
    sql(
        db,
        "INSERT INTO notes (id, title) VALUES ('n1', 'rust guide')",
    )
    .await;
}

#[tokio::test]
async fn a_refused_document_leaves_the_collection_unconverted() {
    let db = open_test_db().await;
    seed(&db).await;
    sql(
        &db,
        "INSERT INTO notes (id, body) VALUES ('n2', 'no title here')",
    )
    .await;

    db.execute_sql(
        "CONVERT COLLECTION notes TO strict (id TEXT NOT NULL PRIMARY KEY, title TEXT NOT NULL)",
        &[],
    )
    .await
    .expect_err("a document without the NOT NULL title must refuse the conversion");

    for id in ["n1", "n2"] {
        assert!(
            db.document_get("notes", id)
                .await
                .expect("document_get")
                .is_some(),
            "document {id} must still be a document"
        );
    }
}

#[tokio::test]
async fn a_converted_collection_stays_text_searchable() {
    let db = open_test_db().await;
    seed(&db).await;

    sql(
        &db,
        "CONVERT COLLECTION notes TO strict (id TEXT NOT NULL PRIMARY KEY, title TEXT)",
    )
    .await;

    let hits = db
        .text_search("notes", "", "rust", 10, TextSearchParams::default(), None)
        .await
        .expect("text_search");
    let ids: Vec<String> = hits.into_iter().map(|r| r.id).collect();
    assert_eq!(ids, vec!["n1"]);
}
