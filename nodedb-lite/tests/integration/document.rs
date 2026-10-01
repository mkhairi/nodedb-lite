// SPDX-License-Identifier: Apache-2.0

use super::fixtures::open_test_db;
use nodedb_client::NodeDb;
use nodedb_types::document::Document;
use nodedb_types::value::Value;

// ─── Document CRUD ───────────────────────────────────────────────────

#[tokio::test]
async fn document_crud_100() {
    let db = open_test_db().await;

    for i in 0..100 {
        let mut doc = Document::new(format!("doc-{i}"));
        doc.set("title", Value::String(format!("Document {i}")));
        doc.set("score", Value::Float(i as f64 * 0.1));
        db.document_put("notes", doc).await.unwrap();
    }

    let doc = db.document_get("notes", "doc-50").await.unwrap().unwrap();
    assert_eq!(doc.id, "doc-50");
    assert_eq!(doc.get_str("title"), Some("Document 50"));

    // Update.
    let mut updated = Document::new("doc-50");
    updated.set("title", Value::String("Updated 50".into()));
    db.document_put("notes", updated).await.unwrap();
    let doc = db.document_get("notes", "doc-50").await.unwrap().unwrap();
    assert_eq!(doc.get_str("title"), Some("Updated 50"));

    // Delete.
    db.document_delete("notes", "doc-50").await.unwrap();
    assert!(db.document_get("notes", "doc-50").await.unwrap().is_none());
    assert!(db.document_get("notes", "doc-49").await.unwrap().is_some());
}
