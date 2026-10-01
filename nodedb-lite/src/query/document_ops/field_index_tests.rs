// SPDX-License-Identifier: Apache-2.0
//! Document indexes driven through SQL DDL and the document API, read back
//! through the index lookup.

use nodedb_client::NodeDb;
use nodedb_types::document::Document;
use nodedb_types::value::Value;

use crate::storage::engine::StorageEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::{Encryption, PagedbStorageDefault};
use crate::{NodeDbLite, PagedbStorageMem};

async fn put_scoped<S: StorageEngine>(db: &NodeDbLite<S>, id: &str, scope: &str) {
    let mut doc = Document::new(id);
    doc.set("scope", Value::String(scope.into()));
    db.document_put("notes", doc).await.unwrap();
}

/// Ids the index lookup returns for `scope = <scope>`, sorted.
async fn lookup<S: StorageEngine>(db: &NodeDbLite<S>, scope: &str) -> Vec<String> {
    let result = super::index_reads::index_lookup(
        &db.query_engine,
        "notes",
        "scope",
        &Value::String(scope.into()),
    )
    .await
    .unwrap();
    let mut ids: Vec<String> = result
        .rows
        .into_iter()
        .filter_map(|row| match row.into_iter().next() {
            Some(Value::String(id)) => Some(id),
            _ => None,
        })
        .collect();
    ids.sort();
    ids
}

fn has_index<S: StorageEngine>(db: &NodeDbLite<S>) -> bool {
    !db.query_engine.indexes.planner_specs("notes").is_empty()
}

#[tokio::test]
async fn document_writes_maintain_index_and_drop_index_removes_it() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    put_scoped(&db, "d1", "a").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();
    // CREATE INDEX builds the entries from the documents already present.
    assert_eq!(lookup(&db, "a").await, ["d1"]);

    put_scoped(&db, "d2", "a").await;
    put_scoped(&db, "d1", "b").await;
    assert_eq!(lookup(&db, "a").await, ["d2"]);
    assert_eq!(lookup(&db, "b").await, ["d1"]);
    db.document_delete("notes", "d2").await.unwrap();
    assert!(lookup(&db, "a").await.is_empty());

    db.execute_sql("DROP INDEX idx_notes_scope ON notes", &[])
        .await
        .unwrap();
    assert!(!has_index(&db));
}

#[tokio::test]
async fn drop_collection_removes_the_index() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    db.execute_sql("CREATE COLLECTION notes", &[])
        .await
        .unwrap();
    put_scoped(&db, "d1", "a").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();
    assert!(has_index(&db));

    db.execute_sql("DROP COLLECTION notes", &[]).await.unwrap();
    assert!(!has_index(&db));
}

/// A RESTORE moves the restored row in the collection's document index, as
/// every other write does.
#[tokio::test]
async fn restore_to_version_moves_the_row_in_the_index() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    put_scoped(&db, "d1", "a").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();
    let earlier: Vec<(u64, i32)> = {
        let crdt = db.crdt.lock().unwrap();
        let vv = crdt.state("notes").expect("written").oplog_version_vector();
        vv.iter().map(|(peer, counter)| (*peer, *counter)).collect()
    };
    let earlier_json = format!(
        "{{{}}}",
        earlier
            .iter()
            .map(|(peer, counter)| format!("\"{peer:x}\": {counter}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    put_scoped(&db, "d1", "b").await;
    assert_eq!(lookup(&db, "b").await, ["d1"]);

    crate::query::crdt_ops::version::handle_restore_to_version(
        &db.query_engine,
        "notes",
        "d1",
        &earlier_json,
    )
    .await
    .unwrap();
    assert_eq!(lookup(&db, "a").await, ["d1"], "the row is back at a");
    assert!(
        lookup(&db, "b").await.is_empty(),
        "and no longer indexed at b"
    );
}

/// The definition is durable. A reopened store finds the documents written
/// before and after `CREATE INDEX` through the index.
#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn index_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("field_index.db");

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        db.execute_sql("CREATE COLLECTION notes", &[])
            .await
            .unwrap();
        put_scoped(&db, "d1", "a").await;
        db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
            .await
            .unwrap();
        put_scoped(&db, "d2", "a").await;
        put_scoped(&db, "d3", "b").await;
        put_scoped(&db, "d1", "b").await;
        assert_eq!(lookup(&db, "a").await, ["d2"]);
        assert_eq!(lookup(&db, "b").await, ["d1", "d3"]);
        db.flush().await.unwrap();
        db.shutdown().await;
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    assert_eq!(lookup(&db, "a").await, ["d2"]);
    assert_eq!(lookup(&db, "b").await, ["d1", "d3"]);
}
