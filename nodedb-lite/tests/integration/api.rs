// SPDX-License-Identifier: Apache-2.0

use super::fixtures::open_test_db;
use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::id::NodeId;
use std::sync::Arc;

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
