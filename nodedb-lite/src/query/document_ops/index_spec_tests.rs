// SPDX-License-Identifier: Apache-2.0
//! Index specs of schemaless document collections: the SQL catalog lists
//! them, a duplicate is refused, they leave with their collection, and they
//! survive a reopen.

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_sql::SqlCatalog;
use nodedb_sql::types::{CollectionInfo, EngineType, IndexState};
use nodedb_types::document::Document;
use nodedb_types::id::DatabaseId;
use nodedb_types::value::Value;

use crate::query::catalog::LiteCatalog;
use crate::storage::engine::StorageEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::{Encryption, PagedbStorageDefault};
use crate::{NodeDbLite, PagedbStorageMem};

async fn make_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    NodeDbLite::open(storage).await.unwrap()
}

async fn put_scoped_doc<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) {
    let mut doc = Document::new("d1");
    doc.set("scope", Value::String("a".into()));
    db.document_put(collection, doc).await.unwrap();
}

/// The catalog the SQL planner sees, with the engine's index catalog.
async fn catalog_for<S: StorageEngine>(db: &NodeDbLite<S>) -> LiteCatalog<S> {
    let metas =
        crate::nodedb::collection::ddl::load_persisted_collection_metas(db.storage.as_ref())
            .await
            .unwrap();
    LiteCatalog::new(
        Arc::clone(&db.crdt),
        Arc::clone(&db.strict),
        Arc::clone(&db.columnar),
        metas,
    )
    .with_indexes(Arc::clone(&db.query_engine.indexes))
}

/// The indexes the catalog lists on `collection`; none when it is absent.
async fn listed<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> Vec<String> {
    catalog_for(db)
        .await
        .get_collection(DatabaseId::new(1), collection)
        .unwrap()
        .map(|info| info.indexes.into_iter().map(|idx| idx.name).collect())
        .unwrap_or_default()
}

fn assert_lists_scope_index(info: &CollectionInfo) {
    assert_eq!(info.engine, EngineType::DocumentSchemaless);
    assert_eq!(info.indexes.len(), 1, "one index spec listed");
    let idx = &info.indexes[0];
    assert_eq!(idx.name, "idx_notes_scope");
    assert_eq!(idx.field, "$.scope", "field in canonical JSON-path form");
    assert!(!idx.unique);
    assert!(!idx.case_insensitive);
    assert!(idx.predicate.is_none());
    // Upstream maintains document index entries on every write, so the
    // spec is Ready and drives query rewrites.
    assert_eq!(idx.state, IndexState::Ready);
}

/// `CREATE INDEX` on a schemaless collection lists the index spec on the
/// collection, and `DROP INDEX` removes it.
#[tokio::test]
async fn catalog_lists_index_after_create_index() {
    let db = make_db().await;
    put_scoped_doc(&db, "notes").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();

    let info = catalog_for(&db)
        .await
        .get_collection(DatabaseId::new(1), "notes")
        .unwrap()
        .expect("collection surfaced");
    assert_lists_scope_index(&info);

    db.execute_sql("DROP INDEX idx_notes_scope ON notes", &[])
        .await
        .unwrap();
    assert!(
        listed(&db, "notes").await.is_empty(),
        "DROP INDEX removes the spec"
    );
}

/// A duplicate `CREATE INDEX` on the same field is refused before any write,
/// whatever its name, and `IF NOT EXISTS` keeps the existing spec untouched.
#[tokio::test]
async fn duplicate_create_index_is_refused() {
    let db = make_db().await;
    put_scoped_doc(&db, "notes").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();

    let err = db
        .execute_sql("CREATE INDEX other_name ON notes (scope)", &[])
        .await
        .expect_err("duplicate refused");
    assert!(
        err.to_string().contains("idx_notes_scope"),
        "the refusal names the existing index: {err}"
    );
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .expect_err("a reused name is refused too");

    db.execute_sql(
        "CREATE INDEX IF NOT EXISTS other_name ON notes (scope)",
        &[],
    )
    .await
    .unwrap();
    let info = catalog_for(&db)
        .await
        .get_collection(DatabaseId::new(1), "notes")
        .unwrap()
        .expect("collection surfaced");
    assert_lists_scope_index(&info);
}

/// `DROP COLLECTION` removes the collection's index specs.
#[tokio::test]
async fn drop_collection_removes_index_specs() {
    let db = make_db().await;
    db.execute_sql("CREATE COLLECTION notes", &[])
        .await
        .unwrap();
    put_scoped_doc(&db, "notes").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();
    assert_eq!(listed(&db, "notes").await, ["idx_notes_scope"]);

    db.execute_sql("DROP COLLECTION notes", &[]).await.unwrap();

    assert!(
        db.query_engine.indexes.planner_specs("notes").is_empty(),
        "specs removed with the collection"
    );
    assert!(listed(&db, "notes").await.is_empty());
}

/// `NodeDbLite::drop_collection` removes the collection's index specs.
#[tokio::test]
async fn api_drop_collection_removes_index_specs() {
    let db = make_db().await;
    put_scoped_doc(&db, "notes").await;
    db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
        .await
        .unwrap();

    db.drop_collection("notes").await.unwrap();

    assert!(db.query_engine.indexes.planner_specs("notes").is_empty());
}

/// The index spec is durable: a reopened database lists it again.
#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn index_spec_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index_spec.db");

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        db.execute_sql("CREATE COLLECTION notes", &[])
            .await
            .unwrap();
        put_scoped_doc(&db, "notes").await;
        db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
            .await
            .unwrap();
        db.flush().await.unwrap();
        db.shutdown().await;
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    let info = catalog_for(&db)
        .await
        .get_collection(DatabaseId::new(1), "notes")
        .unwrap()
        .expect("collection surfaced after reopen");
    assert_lists_scope_index(&info);
}
