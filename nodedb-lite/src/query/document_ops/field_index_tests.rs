// SPDX-License-Identifier: Apache-2.0
//! In-memory field indexes driven through SQL DDL and the document API, read back through the index lookup.

use nodedb_client::NodeDb;
use nodedb_types::document::Document;
use nodedb_types::value::Value;

use super::reads::index_lookup_ids;
use crate::storage::engine::StorageEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::{Encryption, PagedbStorageDefault};
use crate::{NodeDbLite, PagedbStorageMem};

async fn put_scoped<S: StorageEngine>(db: &NodeDbLite<S>, id: &str, scope: &str) {
    let mut doc = Document::new(id);
    doc.set("scope", Value::String(scope.into()));
    db.document_put("notes", doc).await.unwrap();
}

/// Ids the index lookup returns for `scope = <scope>`, with the field in the
/// canonical form the planner passes.
async fn lookup<S: StorageEngine>(db: &NodeDbLite<S>, scope: &str) -> Vec<String> {
    index_lookup_ids(&db.query_engine, "notes", "$.scope", scope)
        .await
        .unwrap()
}

fn assert_consistent<S: StorageEngine>(db: &NodeDbLite<S>) {
    db.crdt.lock().unwrap().assert_field_indexes_consistent();
}

fn has_resident_index<S: StorageEngine>(db: &NodeDbLite<S>) -> bool {
    db.crdt
        .lock()
        .unwrap()
        .field_index_lookup("notes", "$.scope", "a")
        .is_some()
}

#[tokio::test]
async fn document_writes_maintain_index_and_drop_index_removes_it() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    put_scoped(&db, "d1", "a").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();
    // CREATE INDEX builds the postings from the documents already present.
    assert_eq!(lookup(&db, "a").await, ["d1"]);

    put_scoped(&db, "d2", "a").await;
    put_scoped(&db, "d1", "b").await;
    assert_eq!(lookup(&db, "a").await, ["d2"]);
    assert_eq!(lookup(&db, "b").await, ["d1"]);
    db.document_delete("notes", "d2").await.unwrap();
    assert!(lookup(&db, "a").await.is_empty());
    assert_consistent(&db);

    db.execute_sql("DROP INDEX idx_notes_scope ON notes", &[])
        .await
        .unwrap();
    assert!(!has_resident_index(&db));
}

#[tokio::test]
async fn drop_collection_removes_in_memory_index() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    db.execute_sql("CREATE COLLECTION notes", &[])
        .await
        .unwrap();
    put_scoped(&db, "d1", "a").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();
    assert!(has_resident_index(&db));

    db.execute_sql("DROP COLLECTION notes", &[]).await.unwrap();
    assert!(!has_resident_index(&db));
}

/// Only the spec is durable. A reopened store derives the postings from its
/// restored documents, including writes made after `CREATE INDEX`.
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
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    assert_eq!(lookup(&db, "a").await, ["d2"]);
    assert_eq!(lookup(&db, "b").await, ["d1", "d3"]);
    assert_consistent(&db);
}
