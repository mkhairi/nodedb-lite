// SPDX-License-Identifier: Apache-2.0

use nodedb_client::NodeDb;
use nodedb_lite::{GraphRagParams, NodeDbLite, PagedbStorageMem};
use nodedb_types::{Document, Value, id::NodeId};
use std::collections::HashSet;

fn node(id: &str) -> NodeId {
    NodeId::try_new(id).unwrap()
}

#[tokio::test]
async fn caller_seeds_expand_without_vectors_and_include_depth_zero() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    db.graph_insert_edge("docs", &node("a"), &node("b"), "LINK", None)
        .await
        .unwrap();
    db.graph_insert_edge("other", &node("a"), &node("outside"), "LINK", None)
        .await
        .unwrap();
    let mut doc = Document::new("b");
    doc.set("title", Value::String("context".into()));
    db.document_put("docs", doc).await.unwrap();
    let seeds = [node("a")];
    let params = GraphRagParams {
        collection: "docs",
        vector_k: 0,
        seed_nodes: Some(&seeds),
        ..Default::default()
    };
    let results = db.graph_rag_search(&params).await.unwrap();
    assert_eq!(
        results.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(
        results[1].metadata.get("title"),
        Some(&Value::String("context".into()))
    );
    assert_eq!(
        db.graph_rag_search(&GraphRagParams {
            graph_depth: 0,
            ..params
        })
        .await
        .unwrap()
        .len(),
        1
    );
}

#[tokio::test]
async fn vector_and_caller_seeds_union_and_deduplicate() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    db.vector_insert("docs", "a", &[1.0, 0.0], None)
        .await
        .unwrap();
    db.graph_insert_edge("docs", &node("a"), &node("b"), "LINK", None)
        .await
        .unwrap();
    db.graph_insert_edge("docs", &node("c"), &node("d"), "LINK", None)
        .await
        .unwrap();
    let params = GraphRagParams {
        collection: "docs",
        query: &[1.0, 0.0],
        vector_k: 1,
        ..Default::default()
    };
    let vector_only = db.graph_rag_search(&params).await.unwrap();
    assert_eq!(
        vector_only
            .iter()
            .map(|r| r.id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    let seeds = [node("c"), node("a"), node("c")];
    let mixed = db
        .graph_rag_search(&GraphRagParams {
            seed_nodes: Some(&seeds),
            ..params
        })
        .await
        .unwrap();
    let ids: HashSet<_> = mixed.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, HashSet::from(["a", "b", "c", "d"]));
    assert_eq!(mixed.len(), 4);
}

#[tokio::test]
async fn allowed_nodes_bound_seeds_and_every_traversal_hop() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    db.graph_insert_edge("docs", &node("a"), &node("blocked"), "LINK", None)
        .await
        .unwrap();
    db.graph_insert_edge("docs", &node("blocked"), &node("b"), "LINK", None)
        .await
        .unwrap();
    let seeds = [node("a"), node("blocked")];
    let allowed = HashSet::from(["a".to_owned(), "b".to_owned()]);
    let params = GraphRagParams {
        collection: "docs",
        vector_k: 0,
        seed_nodes: Some(&seeds),
        allowed_ids: Some(&allowed),
        ..Default::default()
    };
    assert_eq!(
        db.graph_rag_search(&params)
            .await
            .unwrap()
            .iter()
            .map(|r| r.id.as_str())
            .collect::<Vec<_>>(),
        ["a"]
    );
    let empty = HashSet::new();
    assert!(
        db.graph_rag_search(&GraphRagParams {
            allowed_ids: Some(&empty),
            ..params
        })
        .await
        .unwrap()
        .is_empty()
    );
    assert!(
        db.graph_rag_search(&GraphRagParams { top_k: 0, ..params })
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn invalid_fusion_constants_return_typed_errors() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    for rrf_k in [-1.0, f64::INFINITY, f64::NAN] {
        let error = db
            .graph_rag_search(&GraphRagParams {
                vector_k: 0,
                rrf_k,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(error.code(), nodedb_types::error::ErrorCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn multi_source_expansion_keeps_minimum_depth_and_stable_order() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    for (from, to) in [("a", "b"), ("b", "c"), ("z", "c"), ("z", "d")] {
        db.graph_insert_edge("docs", &node(from), &node(to), "LINK", None)
            .await
            .unwrap();
    }
    let seeds = [node("z"), node("a")];
    let params = GraphRagParams {
        collection: "docs",
        vector_k: 0,
        seed_nodes: Some(&seeds),
        graph_depth: 2,
        ..Default::default()
    };
    let results = db.graph_rag_search(&params).await.unwrap();
    assert_eq!(
        results.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["a", "z", "b", "c", "d"]
    );
    let reversed = [node("a"), node("z")];
    assert_eq!(
        db.graph_rag_search(&GraphRagParams {
            seed_nodes: Some(&reversed),
            ..params
        })
        .await
        .unwrap()
        .iter()
        .map(|r| r.id.as_str())
        .collect::<Vec<_>>(),
        ["a", "z", "b", "c", "d"]
    );
}

#[tokio::test]
async fn vector_only_results_keep_similarity_order_without_graph_evidence() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    db.vector_insert("docs", "z", &[1.0, 0.0], None)
        .await
        .unwrap();
    db.vector_insert("docs", "a", &[0.0, 1.0], None)
        .await
        .unwrap();
    let results = db
        .graph_rag_search(&GraphRagParams {
            collection: "docs",
            query: &[1.0, 0.0],
            vector_k: 2,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        results.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert!(results[0].distance < results[1].distance);
}

#[tokio::test]
async fn zero_rank_smoothing_keeps_finite_scores() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    let seeds = [node("a")];
    let results = db
        .graph_rag_search(&GraphRagParams {
            vector_k: 0,
            seed_nodes: Some(&seeds),
            rrf_k: 0.0,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(results[0].distance, 1.0);
}

#[tokio::test]
async fn seed_limit_counts_unique_allowed_nodes() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    let limit = nodedb_types::config::tuning::DEFAULT_MAX_VISITED;
    let duplicates = vec![node("a"); limit + 1];
    let params = GraphRagParams {
        collection: "docs",
        vector_k: 0,
        graph_depth: 0,
        seed_nodes: Some(&duplicates),
        ..Default::default()
    };
    let results = db.graph_rag_search(&params).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "a");

    let unique: Vec<_> = (0..=limit).map(|id| node(&format!("seed-{id}"))).collect();
    let params = GraphRagParams {
        seed_nodes: Some(&unique),
        ..params
    };
    let error = db.graph_rag_search(&params).await.unwrap_err();
    assert_eq!(
        error.code(),
        nodedb_types::error::ErrorCode::PROGRAM_LIMIT_EXCEEDED
    );

    let allowed = HashSet::from(["seed-0".to_owned()]);
    let results = db
        .graph_rag_search(&GraphRagParams {
            allowed_ids: Some(&allowed),
            ..params
        })
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "seed-0");
}
