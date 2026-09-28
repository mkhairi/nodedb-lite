// SPDX-License-Identifier: Apache-2.0

//! Imported CRDT deltas keep full-text search current on the receiving
//! device: rows a delta creates become searchable, rows it rewrites lose
//! their old terms, and rows it deletes leave no terms behind.
//!
//! Two Lite instances exchange deltas directly through
//! `pending_crdt_deltas` / `import_remote_deltas`; no Origin is involved.
//!
//! Run with:
//!   cargo nextest run -p nodedb-lite --test fts_sync_import

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::text_search::TextSearchParams;
use nodedb_types::value::Value;

const COLLECTION: &str = "notes";

async fn open_test_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("open_in_memory");
    NodeDbLite::open(storage).await.expect("NodeDbLite::open")
}

fn make_doc(id: &str, title: &str) -> Document {
    let mut doc = Document::new(id);
    doc.set("title", Value::String(title.to_owned()));
    doc
}

/// Import every delta `from` has not had acknowledged into `to`.
fn ship(from: &NodeDbLite<PagedbStorageMem>, to: &NodeDbLite<PagedbStorageMem>) {
    for delta in from.pending_crdt_deltas().expect("pending deltas") {
        to.import_remote_deltas(&delta.collection, &delta.delta_bytes)
            .expect("import delta");
    }
}

async fn search_ids(db: &NodeDbLite<PagedbStorageMem>, field: &str, query: &str) -> Vec<String> {
    let mut ids: Vec<String> = db
        .text_search(
            COLLECTION,
            field,
            query,
            10,
            TextSearchParams::default(),
            None,
        )
        .await
        .unwrap_or_else(|e| panic!("text_search {field:?} for {query:?}: {e}"))
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    ids
}

#[tokio::test]
async fn imported_deltas_reindex_the_rows_they_change() {
    let writer = open_test_db().await;
    let reader = open_test_db().await;

    writer
        .document_put(COLLECTION, make_doc("n1", "rust guide"))
        .await
        .expect("put n1");
    writer
        .document_put(COLLECTION, make_doc("n2", "python tips"))
        .await
        .expect("put n2");
    ship(&writer, &reader);

    assert_eq!(search_ids(&reader, "title", "rust").await, vec!["n1"]);
    assert_eq!(search_ids(&reader, "", "python").await, vec!["n2"]);

    writer
        .document_put(COLLECTION, make_doc("n1", "go handbook"))
        .await
        .expect("overwrite n1");
    writer
        .document_delete(COLLECTION, "n2")
        .await
        .expect("delete n2");
    ship(&writer, &reader);

    assert!(
        search_ids(&reader, "title", "rust").await.is_empty(),
        "an imported overwrite must retract the old terms"
    );
    assert_eq!(search_ids(&reader, "title", "handbook").await, vec!["n1"]);
    assert!(
        search_ids(&reader, "", "python").await.is_empty(),
        "an imported delete must retract the row"
    );
}
