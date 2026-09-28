//! Integration tests for NodeDB-Lite.
//!
//! Tests the full stack: StorageEngine → Engines → NodeDbLite → NodeDb trait.
//! Performance/scale workloads live in `nodedb-bench/benches/`.

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{Encryption, NodeDbLite, PagedbStorageDefault, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::id::{EdgeId, NodeId};
use nodedb_types::value::Value;

async fn open_test_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    NodeDbLite::open(storage).await.unwrap()
}

// ─── 1K Vector Insert + Search ───────────────────────────────────────

/// Correctness-only counterpart of the 1k×32d batch workload.
/// Scale benchmark: `nodedb-bench/benches/micro/lite_vector.rs`.
#[tokio::test]
async fn vector_batch_insert_and_search_correctness() {
    let db = open_test_db().await;
    let n = 50;
    let dim = 32;

    let vectors: Vec<(String, Vec<f32>)> = (0..n)
        .map(|i| {
            let emb: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32) * 0.001).collect();
            (format!("v{i}"), emb)
        })
        .collect();

    let refs: Vec<(&str, &[f32])> = vectors
        .iter()
        .map(|(id, emb)| (id.as_str(), emb.as_slice()))
        .collect();

    db.batch_vector_insert("vecs", &refs).await.unwrap();

    let query: Vec<f32> = (0..dim).map(|d| ((25 * dim + d) as f32) * 0.001).collect();
    let results = db
        .vector_search("vecs", &query, 10, None, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 10);
    for w in results.windows(2) {
        assert!(
            w[0].distance <= w[1].distance,
            "results not sorted by distance"
        );
    }
    assert!(
        results[0].distance < 0.5,
        "top result distance {} is too large",
        results[0].distance
    );
}

// ─── 10K Graph Edges + Traverse ──────────────────────────────────────

/// Correctness-only counterpart of the 10k-edge graph workload.
/// Scale benchmark: `nodedb-bench/benches/micro/lite_graph.rs`.
#[tokio::test]
async fn graph_batch_and_traverse_correctness() {
    let db = open_test_db().await;

    // 50 nodes × 4 edges each = 200 edges. 2-hop BFS visits 10+ nodes.
    let mut edges: Vec<(String, String, &str)> = Vec::with_capacity(200);
    for i in 0..50 {
        for j in 1..=4 {
            let dst = (i * 4 + j) % 50;
            edges.push((format!("n{i}"), format!("n{dst}"), "LINK"));
        }
    }
    let refs: Vec<(NodeId, NodeId, &str, Option<Document>)> = edges
        .iter()
        .map(|(s, d, l)| {
            (
                NodeId::from_validated(s.clone()),
                NodeId::from_validated(d.clone()),
                *l,
                None,
            )
        })
        .collect();

    db.batch_graph_insert_edges("graph", &refs).await.unwrap();
    db.compact_graph().unwrap();

    let subgraph = db
        .graph_traverse("graph", &NodeId::from_validated("n0".to_string()), 2, None)
        .await
        .unwrap();

    assert!(subgraph.node_count() > 5);
    assert!(subgraph.edge_count() > 0);
}

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
    let subgraph = db.graph_traverse("kb", &start, 2, None).await.unwrap();
    assert!(subgraph.node_count() >= 1);

    let note = db.document_get("notes", "note-1").await.unwrap().unwrap();
    assert!(note.get_str("body").unwrap().contains("AI"));
}

// ─── Batch Graph Edge Insert: EdgeId Keying ───────────────────────────

/// A batch-inserted edge's returned `EdgeId` must be the same key
/// `graph_delete_edge` removes, and the deletion must survive a reopen
/// (regression guard: the pre-`EdgeId` batch path stored edges under
/// `"{src}--{label}-->{dst}"`, a key `graph_delete_edge` never matched, so
/// the edge resurrected on every CSR rebuild).
#[tokio::test]
async fn batch_inserted_edge_is_removed_by_delete_and_stays_removed_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batch_delete.db");

    let edge_id = {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();

        let ids = db
            .batch_graph_insert_edges(
                "social",
                &[(
                    NodeId::from_validated("a".to_string()),
                    NodeId::from_validated("b".to_string()),
                    "KNOWS",
                    None,
                )],
            )
            .await
            .unwrap();
        assert_eq!(ids.len(), 1);
        let edge_id = ids[0].clone();

        db.graph_delete_edge("social", &edge_id).await.unwrap();
        db.flush().await.unwrap();
        edge_id
    };

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();

    let subgraph = db
        .graph_traverse("social", &NodeId::from_validated("a".to_string()), 1, None)
        .await
        .unwrap();

    assert!(
        subgraph.edges.iter().all(|e| e.id != edge_id),
        "a batch-inserted edge deleted before flush must stay deleted after reopen"
    );
}

/// Properties passed to a batch edge insert must be readable from traversal,
/// the same way a single `graph_insert_edge` call's properties are.
#[tokio::test]
async fn batch_inserted_edge_properties_are_visible_in_traversal() {
    let db = open_test_db().await;

    let mut props = Document::new("");
    props.set("weight", Value::Integer(7));

    db.batch_graph_insert_edges(
        "social",
        &[(
            NodeId::from_validated("a".to_string()),
            NodeId::from_validated("b".to_string()),
            "KNOWS",
            Some(props),
        )],
    )
    .await
    .unwrap();

    let subgraph = db
        .graph_traverse("social", &NodeId::from_validated("a".to_string()), 1, None)
        .await
        .unwrap();

    assert_eq!(subgraph.edges.len(), 1);
    assert_eq!(
        subgraph.edges[0].properties.get("weight"),
        Some(&Value::Integer(7))
    );
}

/// Each `EdgeId` a batch insert returns must identify exactly the edge it
/// was allocated for, so deleting one by its returned id leaves the rest.
#[tokio::test]
async fn batch_insert_returns_edge_ids_usable_for_delete() {
    let db = open_test_db().await;

    let ids = db
        .batch_graph_insert_edges(
            "social",
            &[
                (
                    NodeId::from_validated("a".to_string()),
                    NodeId::from_validated("b".to_string()),
                    "KNOWS",
                    None,
                ),
                (
                    NodeId::from_validated("a".to_string()),
                    NodeId::from_validated("c".to_string()),
                    "KNOWS",
                    None,
                ),
            ],
        )
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);

    db.graph_delete_edge("social", &ids[0]).await.unwrap();

    let subgraph = db
        .graph_traverse("social", &NodeId::from_validated("a".to_string()), 1, None)
        .await
        .unwrap();

    assert_eq!(subgraph.edges.len(), 1);
    assert_eq!(subgraph.edges[0].id, ids[1]);
}

/// An edge written under the legacy `"{src}--{label}-->{dst}"` CRDT key
/// (the pre-`EdgeId` batch-insert format) must be rewritten to its `EdgeId`
/// key on open: its properties become visible to traversal, and deleting it
/// by its `EdgeId` is durable across a further reopen.
#[tokio::test]
async fn legacy_batch_edge_keys_are_migrated_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy_migrate.db");

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();

        // Write a legacy-format edge document directly through the CRDT
        // document API, mirroring what the pre-EdgeId batch insert path
        // used to store: "{src}--{label}-->{dst}" as the CRDT doc id under
        // "__edges__{collection}", with src/dst/label as its fields.
        let mut legacy = Document::new("a--KNOWS-->b");
        legacy.set("src", Value::String("a".into()));
        legacy.set("dst", Value::String("b".into()));
        legacy.set("label", Value::String("KNOWS".into()));
        legacy.set("weight", Value::Integer(3));
        db.document_put("__edges__social", legacy).await.unwrap();

        db.flush().await.unwrap();
    }

    // Reopen: migration must rewrite the legacy key to its EdgeId form.
    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();

    let subgraph = db
        .graph_traverse("social", &NodeId::from_validated("a".to_string()), 1, None)
        .await
        .unwrap();
    assert_eq!(
        subgraph.edges.len(),
        1,
        "migrated legacy edge must appear in traversal"
    );
    assert_eq!(
        subgraph.edges[0].properties.get("weight"),
        Some(&Value::Integer(3)),
        "migrated legacy edge properties must survive the key rewrite"
    );

    let edge_id = EdgeId::try_first(
        NodeId::from_validated("a".to_string()),
        NodeId::from_validated("b".to_string()),
        "KNOWS",
    )
    .unwrap();
    db.graph_delete_edge("social", &edge_id).await.unwrap();

    // Keep "social" non-empty across the next reopen so the assertion below
    // exercises a CSR-checkpoint restore, not a from-scratch rebuild.
    db.graph_insert_edge(
        "social",
        &NodeId::from_validated("a".to_string()),
        &NodeId::from_validated("c".to_string()),
        "KNOWS",
        None,
    )
    .await
    .unwrap();
    db.flush().await.unwrap();
    drop(db);

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    let subgraph = db
        .graph_traverse("social", &NodeId::from_validated("a".to_string()), 1, None)
        .await
        .unwrap();
    assert_eq!(
        subgraph.edges.len(),
        1,
        "the migrated edge deleted by its EdgeId must stay deleted after reopen"
    );
    assert_eq!(subgraph.edges[0].to.as_str(), "c");
}

// ─── Persistence: Flush and Reopen ───────────────────────────────────

#[tokio::test]
async fn flush_and_reopen_persists_all() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("persist.db");

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();

        db.batch_vector_insert("vecs", &[("v1", &[1.0, 2.0, 3.0][..])])
            .await
            .unwrap();
        db.batch_graph_insert_edges(
            "vecs",
            &[(
                NodeId::from_validated("a".to_string()),
                NodeId::from_validated("b".to_string()),
                "KNOWS",
                None,
            )],
        )
        .await
        .unwrap();
        let mut doc = Document::new("d1");
        doc.set("key", Value::String("persistent".into()));
        db.document_put("docs", doc).await.unwrap();

        db.flush().await.unwrap();
    }

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();

        let doc = db.document_get("docs", "d1").await.unwrap();
        assert!(doc.is_some(), "document should persist across restart");

        let results = db
            .vector_search("vecs", &[1.0, 2.0, 3.0], 1, None, None)
            .await
            .unwrap();
        assert!(!results.is_empty(), "vector should persist across restart");

        let sg = db
            .graph_traverse("vecs", &NodeId::from_validated("a".to_string()), 1, None)
            .await
            .unwrap();
        assert!(sg.node_count() >= 2, "graph should persist across restart");
    }
}

// ─── CRDT Deltas ─────────────────────────────────────────────────────

#[tokio::test]
async fn all_operations_generate_deltas() {
    let db = open_test_db().await;

    db.vector_insert("v", "v1", &[1.0], None).await.unwrap();
    db.graph_insert_edge(
        "test",
        &NodeId::from_validated("a".to_string()),
        &NodeId::from_validated("b".to_string()),
        "L",
        None,
    )
    .await
    .unwrap();
    db.document_put("d", Document::new("d1")).await.unwrap();
    db.document_delete("d", "d1").await.unwrap();

    let deltas = db.pending_crdt_deltas().unwrap();
    assert!(
        deltas.len() >= 4,
        "expected >= 4 deltas, got {}",
        deltas.len()
    );
}

// ─── Arc<dyn NodeDb> Pattern ─────────────────────────────────────────

#[tokio::test]
async fn arc_dyn_nodedb_pattern() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    // `open` already yields an `Arc`, which coerces straight to the trait object.
    let db: Arc<dyn NodeDb> = NodeDbLite::open(storage).await.unwrap();

    db.vector_insert("coll", "v1", &[1.0, 0.0], None)
        .await
        .unwrap();
    let results = db
        .vector_search("coll", &[1.0, 0.0], 1, None, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    db.document_put("docs", Document::new("d1")).await.unwrap();
    assert!(db.document_get("docs", "d1").await.unwrap().is_some());
}

// `benchmark_vector_search_1k` and `benchmark_graph_bfs_10k_edges` were
// migrated to fluxbench benchmarks — they asserted only wall-clock budgets
// and don't belong in the test suite. See:
//   nodedb-bench/benches/micro/lite_vector.rs (lite_vector_search_1k_32d)
//   nodedb-bench/benches/micro/lite_graph.rs  (lite_graph_bfs_2hop_10k)
