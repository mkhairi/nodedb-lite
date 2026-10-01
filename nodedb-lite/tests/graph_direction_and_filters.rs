// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::{
    Document, Value,
    filter::{EdgeFilter, MetadataFilter},
    graph::Direction,
    id::NodeId,
};

async fn open() -> Arc<NodeDbLite<PagedbStorageMem>> {
    NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap()
}

fn node(id: &str) -> NodeId {
    NodeId::try_new(id).unwrap()
}
fn props(score: i64) -> Document {
    let mut doc = Document::new("properties");
    doc.set("score", Value::Integer(score));
    doc.set("active", Value::Bool(true));
    doc
}
fn score_filter() -> EdgeFilter {
    EdgeFilter {
        labels: vec!["LINK".into(), "NEXT".into()],
        property_filters: vec![
            MetadataFilter::Gt {
                field: "score".into(),
                value: Value::Float(5.0),
            },
            MetadataFilter::and(vec![
                MetadataFilter::eq("active", true),
                MetadataFilter::Not(Box::new(MetadataFilter::eq("missing", "present"))),
            ]),
        ],
    }
}

#[tokio::test]
async fn incoming_and_both_preserve_orientation_and_unique_edges() {
    let db = open().await;
    for (src, dst) in [("a", "b"), ("b", "a"), ("b", "b"), ("c", "b")] {
        db.graph_insert_edge("graph", &node(src), &node(dst), "LINK", None)
            .await
            .unwrap();
    }
    db.graph_insert_edge("other", &node("outside"), &node("b"), "LINK", None)
        .await
        .unwrap();
    let incoming = db
        .graph_traverse("graph", &node("b"), 1, Direction::In, None)
        .await
        .unwrap();
    let pairs: HashSet<_> = incoming
        .edges
        .iter()
        .map(|e| (e.from.as_str(), e.to.as_str()))
        .collect();
    assert!(pairs.contains(&("a", "b")));
    assert!(pairs.contains(&("c", "b")));
    assert!(pairs.contains(&("b", "b")));
    assert!(!pairs.contains(&("outside", "b")));
    let both = db
        .graph_traverse("graph", &node("b"), 1, Direction::Both, None)
        .await
        .unwrap();
    assert_eq!(both.edges.len(), 4);
    assert_eq!(both.nodes.len(), 3);
    assert_eq!(
        both.edges
            .iter()
            .map(|e| e.id.to_string())
            .collect::<HashSet<_>>()
            .len(),
        4
    );
    let zero = db
        .graph_traverse("graph", &node("b"), 0, Direction::Both, None)
        .await
        .unwrap();
    assert_eq!(zero.nodes.len(), 1);
    assert_eq!(zero.edges.len(), 1);
}

#[tokio::test]
async fn predicates_reject_intermediate_edges_before_discovery() {
    let db = open().await;
    for (src, dst, label, score) in [
        ("a", "b", "LINK", 9),
        ("b", "c", "NEXT", 1),
        ("c", "d", "LINK", 9),
    ] {
        db.graph_insert_edge("graph", &node(src), &node(dst), label, Some(props(score)))
            .await
            .unwrap();
    }
    let filter = score_filter();
    let subgraph = db
        .graph_traverse("graph", &node("a"), 3, Direction::Out, Some(&filter))
        .await
        .unwrap();
    assert_eq!(
        subgraph
            .nodes
            .iter()
            .map(|n| n.id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(subgraph.edges.len(), 1);
    assert_eq!(
        subgraph.edges[0].properties.get("score"),
        Some(&Value::Integer(9))
    );
    assert!(
        db.graph_shortest_path("graph", &node("a"), &node("d"), 3, Some(&filter))
            .await
            .unwrap()
            .is_none()
    );
    let incoming = db
        .graph_traverse("graph", &node("b"), 1, Direction::In, Some(&filter))
        .await
        .unwrap();
    assert_eq!(incoming.edges[0].from.as_str(), "a");
}

#[tokio::test]
async fn paths_accept_multiple_labels_and_bound_edge_count() {
    let db = open().await;
    for (src, dst, label) in [("a", "b", "LINK"), ("b", "c", "NEXT")] {
        db.graph_insert_edge("graph", &node(src), &node(dst), label, Some(props(9)))
            .await
            .unwrap();
    }
    for filter in [EdgeFilter::labels(["LINK", "NEXT"]), score_filter()] {
        assert!(
            db.graph_shortest_path("graph", &node("a"), &node("c"), 1, Some(&filter))
                .await
                .unwrap()
                .is_none()
        );
        let path = db
            .graph_shortest_path("graph", &node("a"), &node("c"), 2, Some(&filter))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            path.iter().map(NodeId::as_str).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(
            db.graph_shortest_path("graph", &node("a"), &node("a"), 0, Some(&filter))
                .await
                .unwrap()
                .unwrap(),
            vec![node("a")]
        );
        assert!(
            db.graph_shortest_path("graph", &node("a"), &node("b"), 0, Some(&filter))
                .await
                .unwrap()
                .is_none()
        );
    }
    let unknown = EdgeFilter::labels(["UNKNOWN"]);
    assert!(
        db.graph_shortest_path("graph", &node("a"), &node("c"), 3, Some(&unknown))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        db.graph_traverse("graph", &node("a"), 3, Direction::Both, Some(&unknown))
            .await
            .unwrap()
            .edges
            .is_empty()
    );
}

#[tokio::test]
async fn batch_properties_and_missing_fields_use_shared_predicates() {
    let db = open().await;
    db.batch_graph_insert_edges(
        "graph",
        &[
            (node("a"), node("b"), "LINK", Some(props(9))),
            (node("a"), node("c"), "LINK", None),
        ],
    )
    .await
    .unwrap();
    let filter = score_filter();
    let result = db
        .graph_traverse("graph", &node("a"), 1, Direction::Out, Some(&filter))
        .await
        .unwrap();
    assert_eq!(result.edges.len(), 1);
    assert_eq!(result.edges[0].to.as_str(), "b");
    let missing = EdgeFilter {
        labels: Vec::new(),
        property_filters: vec![MetadataFilter::eq("missing", Value::Null)],
    };
    assert_eq!(
        db.graph_traverse("graph", &node("a"), 1, Direction::Out, Some(&missing))
            .await
            .unwrap()
            .edges
            .len(),
        2
    );
}

#[tokio::test]
async fn empty_and_zero_hop_queries_skip_irrelevant_sql_properties() {
    use nodedb_lite::storage::engine::StorageEngine;
    use nodedb_types::Namespace;

    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage.clone()).await.unwrap();
    let filter = score_filter();
    storage
        .put(Namespace::Graph, b"empty\0a\0LINK\0b", b"malformed")
        .await
        .unwrap();
    assert!(
        db.graph_traverse("empty", &node("a"), 1, Direction::Out, Some(&filter))
            .await
            .unwrap()
            .nodes
            .is_empty()
    );
    assert!(
        db.graph_shortest_path("empty", &node("a"), &node("a"), 0, Some(&filter))
            .await
            .unwrap()
            .is_none()
    );

    db.graph_insert_edge("graph", &node("a"), &node("b"), "LINK", Some(props(9)))
        .await
        .unwrap();
    storage
        .put(Namespace::Graph, b"graph\0a\0LINK\0b", b"malformed")
        .await
        .unwrap();
    assert!(
        db.graph_traverse("graph", &node("unknown"), 1, Direction::Both, Some(&filter))
            .await
            .unwrap()
            .nodes
            .is_empty()
    );
    assert!(
        db.graph_shortest_path("graph", &node("unknown"), &node("b"), 1, Some(&filter))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        db.graph_shortest_path("graph", &node("a"), &node("unknown"), 1, Some(&filter))
            .await
            .unwrap()
            .is_none()
    );
    for depth in [0, 2] {
        assert_eq!(
            db.graph_shortest_path("graph", &node("a"), &node("a"), depth, Some(&filter))
                .await
                .unwrap(),
            Some(vec![node("a")])
        );
    }
    assert!(
        db.graph_shortest_path("graph", &node("a"), &node("b"), 0, Some(&filter))
            .await
            .unwrap()
            .is_none()
    );
    let zero = db
        .graph_traverse("graph", &node("a"), 0, Direction::Both, Some(&filter))
        .await
        .unwrap();
    assert_eq!(zero.nodes.len(), 1);
    assert!(zero.edges.is_empty());
    assert!(
        db.graph_shortest_path("graph", &node("a"), &node("b"), 1, Some(&filter))
            .await
            .is_err()
    );
    assert!(
        db.graph_traverse("graph", &node("a"), 1, Direction::Out, Some(&filter))
            .await
            .is_err()
    );
}
