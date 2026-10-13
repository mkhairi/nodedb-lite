// SPDX-License-Identifier: Apache-2.0

//! Edges stored as `Namespace::Graph` rows survive a reopen, and edges moved
//! out of the CRDT engine do not come back.

use nodedb_client::NodeDb;
use nodedb_lite::{Encryption, NodeDbLite, PagedbStorageDefault};
use nodedb_types::filter::EdgeFilter;
use nodedb_types::graph::Direction;
use nodedb_types::id::NodeId;

fn node(id: &str) -> NodeId {
    NodeId::from_validated(id.to_string())
}

#[tokio::test]
async fn moved_and_stored_edges_survive_reopen() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();
    let (a, b, c) = (node("a"), node("b"), node("c"));
    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        db.graph_insert_edge("g", &a, &b, "E", None).await.unwrap();
        assert_eq!(db.graph_move_crdt_edges_to_store("g").await.unwrap(), 1);
        db.graph_put_stored_edges("g", &[(&a, "E", &c)])
            .await
            .unwrap();
        db.flush().await.unwrap();
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    assert!(db.graph_crdt_edge_collections().is_empty());
    let mut ids: Vec<String> = db
        .graph_traverse("g", &a, 1, Direction::Out, Some(&EdgeFilter::default()))
        .await
        .unwrap()
        .nodes
        .into_iter()
        .map(|n| n.id.as_str().to_string())
        .filter(|id| id != "a")
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["b", "c"]);
}
