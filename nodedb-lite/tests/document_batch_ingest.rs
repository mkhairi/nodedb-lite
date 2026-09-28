// SPDX-License-Identifier: Apache-2.0

//! Batch document ingest tests.
//!
//! Verifies that `document_put_with_vector_batch_impl` correctly writes all
//! documents (queryable via `document_get`), indexes their vectors (queryable
//! via vector search), and advances the CRDT version vector by producing one
//! pending delta per CRDT row it touches — never one coalesced delta for the
//! whole batch, which no receiver could apply row by row.

use nodedb_client::NodeDb;
use nodedb_lite::storage::pagedb_storage::PagedbStorageMem;
use nodedb_lite::{BatchItem, NodeDbLite};
use nodedb_types::document::Document;

async fn open_db() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("open in-memory storage");
    NodeDbLite::open(storage).await.expect("open NodeDbLite")
}

fn make_doc(id: &str, content: &str) -> Document {
    let mut doc = Document::new(id);
    doc.set(
        "content",
        nodedb_types::value::Value::String(content.to_owned()),
    );
    doc
}

fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| seed + i as f32 * 0.01).collect()
}

#[tokio::test]
async fn batch_100_docs_all_queryable() {
    let db = open_db().await;

    let docs: Vec<Document> = (0..100)
        .map(|i| make_doc(&format!("doc{i:03}"), &format!("content {i}")))
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..100).map(|i| make_embedding(8, i as f32)).collect();

    let items: Vec<BatchItem<'_>> = docs
        .iter()
        .zip(embeddings.iter())
        .map(|(doc, emb)| BatchItem {
            doc_collection: "docs",
            doc: doc.clone(),
            vector_collection: "vecs",
            id: doc.id.as_str(),
            embedding: Some(emb.as_slice()),
        })
        .collect();

    let ids = db
        .document_put_with_vector_batch_impl(&items)
        .await
        .expect("batch put");

    assert_eq!(ids.len(), 100, "should return one ID per item");

    // All documents must be readable.
    for i in 0..100usize {
        let id = format!("doc{i:03}");
        let doc = db
            .document_get("docs", &id)
            .await
            .expect("document_get")
            .unwrap_or_else(|| panic!("doc {id} not found after batch insert"));
        assert_eq!(doc.id, id);
    }
}

/// A batch emits one delta per CRDT row, not one per batch: each document row
/// and each vector-metadata row is exported on its own so the receiver — which
/// commits per row and stores documents per collection — can apply it
/// independently. Coalescing them back into a single delta would break sync.
#[tokio::test]
async fn batch_produces_one_crdt_delta_per_row() {
    let db = open_db().await;

    let docs: Vec<Document> = (0..50)
        .map(|i| make_doc(&format!("d{i}"), &format!("text {i}")))
        .collect();
    let embeddings: Vec<Vec<f32>> = (0..50).map(|i| make_embedding(4, i as f32)).collect();

    let items: Vec<BatchItem<'_>> = docs
        .iter()
        .zip(embeddings.iter())
        .map(|(doc, emb)| BatchItem {
            doc_collection: "col",
            doc: doc.clone(),
            vector_collection: "col_vec",
            id: doc.id.as_str(),
            embedding: Some(emb.as_slice()),
        })
        .collect();

    let deltas_before = db
        .pending_crdt_deltas()
        .expect("pending_crdt_deltas before")
        .len();

    db.document_put_with_vector_batch_impl(&items)
        .await
        .expect("batch put");

    let deltas_after = db
        .pending_crdt_deltas()
        .expect("pending_crdt_deltas after")
        .len();

    // One delta per CRDT row: every item writes a document row, and every item
    // carrying a non-empty embedding also writes a vector-metadata row.
    let vector_rows = items
        .iter()
        .filter(|it| it.embedding.is_some_and(|e| !e.is_empty()))
        .count();
    let expected_rows = items.len() + vector_rows;

    assert_eq!(
        deltas_after,
        deltas_before + expected_rows,
        "batch of {} items ({} with embeddings) must produce {expected_rows} CRDT \
         deltas — one per row, not one per batch — because a delta spanning rows \
         or collections is not independently applicable by a receiver that commits \
         per row and stores documents per collection; got {deltas_after} (was \
         {deltas_before})",
        items.len(),
        vector_rows
    );
}

#[tokio::test]
async fn batch_vectors_searchable() {
    let db = open_db().await;

    let docs: Vec<Document> = (0..10)
        .map(|i| make_doc(&format!("e{i}"), &format!("entry {i}")))
        .collect();

    // Make the first embedding a unit vector so it ranks first.
    let mut embeddings: Vec<Vec<f32>> = (0..10).map(|i| make_embedding(4, i as f32)).collect();
    embeddings[0] = vec![1.0, 0.0, 0.0, 0.0];

    let items: Vec<BatchItem<'_>> = docs
        .iter()
        .zip(embeddings.iter())
        .map(|(doc, emb)| BatchItem {
            doc_collection: "vec_entries",
            doc: doc.clone(),
            vector_collection: "vec_entries",
            id: doc.id.as_str(),
            embedding: Some(emb.as_slice()),
        })
        .collect();

    db.document_put_with_vector_batch_impl(&items)
        .await
        .expect("batch put");

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let results = db
        .vector_search("vec_entries", &query, 3, None, None)
        .await
        .expect("vector_search");

    assert!(
        !results.is_empty(),
        "vector search should return results after batch insert"
    );
    assert_eq!(
        results[0].id, "e0",
        "closest vector should be e0 (exact match)"
    );
}

#[tokio::test]
async fn batch_without_embeddings() {
    let db = open_db().await;

    let docs: Vec<Document> = (0..20)
        .map(|i| make_doc(&format!("p{i}"), &format!("plain {i}")))
        .collect();

    let items: Vec<BatchItem<'_>> = docs
        .iter()
        .map(|doc| BatchItem {
            doc_collection: "plain",
            doc: doc.clone(),
            vector_collection: "plain",
            id: doc.id.as_str(),
            embedding: None,
        })
        .collect();

    let ids = db
        .document_put_with_vector_batch_impl(&items)
        .await
        .expect("batch put without embeddings");

    assert_eq!(ids.len(), 20);

    for i in 0..20usize {
        let id = format!("p{i}");
        assert!(
            db.document_get("plain", &id).await.expect("get").is_some(),
            "doc {id} should exist"
        );
    }
}

/// An id repeated within one batch keeps one HNSW node: the latest vector.
#[tokio::test]
async fn batch_with_duplicate_ids_keeps_one_node_per_id() {
    let db = open_db().await;

    let first = make_doc("d1", "first");
    let second = make_doc("d1", "second");
    let other = make_doc("d2", "other");
    let items = vec![
        BatchItem {
            doc_collection: "docs",
            doc: first,
            vector_collection: "vecs",
            id: "d1",
            embedding: Some(&[1.0, 0.0, 0.0]),
        },
        BatchItem {
            doc_collection: "docs",
            doc: second,
            vector_collection: "vecs",
            id: "d1",
            embedding: Some(&[0.0, 1.0, 0.0]),
        },
        BatchItem {
            doc_collection: "docs",
            doc: other,
            vector_collection: "vecs",
            id: "d2",
            embedding: Some(&[0.0, 0.0, 1.0]),
        },
    ];
    db.document_put_with_vector_batch_impl(&items)
        .await
        .expect("batch put");

    let hits = db
        .vector_search("vecs", &[0.0, 1.0, 0.0], 10, None, None)
        .await
        .expect("vector_search");
    let d1: Vec<_> = hits.iter().filter(|h| h.id == "d1").collect();
    assert_eq!(d1.len(), 1, "one result per id, got {hits:?}");
    assert!(
        d1[0].distance.abs() < 1e-5,
        "scored against the latest vector"
    );
    assert_eq!(hits.len(), 2, "two ids, two results");
}

/// A batch whose documents and vectors share a collection and id keeps
/// every document field on the row.
#[tokio::test]
async fn batch_same_collection_keeps_document_fields() {
    let db = open_db().await;

    let items = vec![BatchItem {
        doc_collection: "notes",
        doc: make_doc("n1", "body text"),
        vector_collection: "notes",
        id: "n1",
        embedding: Some(&[1.0, 0.0]),
    }];
    db.document_put_with_vector_batch_impl(&items)
        .await
        .expect("batch put");

    let row = db
        .document_get("notes", "n1")
        .await
        .expect("document_get")
        .expect("row exists");
    assert_eq!(
        row.fields.get("content"),
        Some(&nodedb_types::value::Value::String("body text".to_owned()))
    );
}
