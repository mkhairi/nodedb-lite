// SPDX-License-Identifier: Apache-2.0

use super::fixtures::open_test_db;
use nodedb_client::NodeDb;
use nodedb_types::document::Document;
use nodedb_types::id::NodeId;
use nodedb_types::value::Value;

// ─── Multi-Modal Query ───────────────────────────────────────────────

#[tokio::test]
async fn multi_modal_vector_graph_document() {
    let db = open_test_db().await;

    db.batch_vector_insert(
        "kb",
        &[
            ("concept-ai", &[1.0, 0.0, 0.0][..]),
            ("concept-ml", &[0.9, 0.1, 0.0]),
            ("concept-db", &[0.0, 0.0, 1.0]),
        ],
    )
    .await
    .unwrap();

    db.batch_graph_insert_edges(
        "kb",
        &[
            (
                NodeId::from_validated("concept-ai".to_string()),
                NodeId::from_validated("concept-ml".to_string()),
                "RELATES_TO",
                None,
            ),
            (
                NodeId::from_validated("concept-ml".to_string()),
                NodeId::from_validated("concept-db".to_string()),
                "USES",
                None,
            ),
        ],
    )
    .await
    .unwrap();

    let mut doc = Document::new("note-1");
    doc.set("body", Value::String("AI and ML are related".into()));
    db.document_put("notes", doc).await.unwrap();

    // Vector search → graph traverse → document read.
    let results = db
        .vector_search("kb", &[1.0, 0.0, 0.0], 2, None, None)
        .await
        .unwrap();
    assert!(!results.is_empty());

    let start = NodeId::from_validated(results[0].id.clone());
    let subgraph = db
        .graph_traverse("kb", &start, 2, nodedb_types::graph::Direction::Out, None)
        .await
        .unwrap();
    assert!(subgraph.node_count() >= 1);

    let note = db.document_get("notes", "note-1").await.unwrap().unwrap();
    assert!(note.get_str("body").unwrap().contains("AI"));
}
