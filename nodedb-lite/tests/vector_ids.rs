// SPDX-License-Identifier: Apache-2.0
//! `NodeDbLite::vector_ids` reports exact durable membership.

use nodedb_client::NodeDb;
use nodedb_lite::{Encryption, NodeDbLite};

fn v(seed: usize) -> Vec<f32> {
    (0..8)
        .map(|i| (((seed * 31 + i * 17 + 7) % 101) as f32) / 10.0)
        .collect()
}

fn strs(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| (*s).to_string()).collect()
}

#[tokio::test]
async fn vector_ids_is_exact_and_tracks_delete_and_reinsert() {
    let dir = tempfile::tempdir().unwrap();
    let db = NodeDbLite::open_at_path(dir.path(), Encryption::Plaintext)
        .await
        .unwrap();

    for (i, id) in ["c", "a", "b"].iter().enumerate() {
        db.vector_insert("c1", id, &v(i), None).await.unwrap();
    }
    // Siblings sharing a name prefix with `c1`.
    db.vector_insert("c10", "x", &v(9), None).await.unwrap();
    db.vector_insert("c1_sub", "y", &v(10), None).await.unwrap();

    assert_eq!(db.vector_ids("c1").await.unwrap(), strs(&["a", "b", "c"]));
    assert_eq!(db.vector_ids("c10").await.unwrap(), strs(&["x"]));

    db.vector_delete("c1", "b").await.unwrap();
    assert_eq!(db.vector_ids("c1").await.unwrap(), strs(&["a", "c"]));

    db.vector_insert("c1", "b", &v(1), None).await.unwrap();
    assert_eq!(db.vector_ids("c1").await.unwrap(), strs(&["a", "b", "c"]));
}

#[tokio::test]
async fn vector_ids_keeps_ids_containing_colons() {
    let dir = tempfile::tempdir().unwrap();
    let db = NodeDbLite::open_at_path(dir.path(), Encryption::Plaintext)
        .await
        .unwrap();
    db.vector_insert("c1", "u:1", &v(1), None).await.unwrap();
    db.vector_insert("c1", "u:2", &v(2), None).await.unwrap();
    assert_eq!(db.vector_ids("c1").await.unwrap(), strs(&["u:1", "u:2"]));
}

#[tokio::test]
async fn vector_ids_survives_flush_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = NodeDbLite::open_at_path(dir.path(), Encryption::Plaintext)
            .await
            .unwrap();
        for (i, id) in ["a", "b", "c"].iter().enumerate() {
            db.vector_insert("c1", id, &v(i), None).await.unwrap();
        }
        db.flush().await.unwrap();
        db.shutdown().await;
    }
    let db = NodeDbLite::open_at_path(dir.path(), Encryption::Plaintext)
        .await
        .unwrap();
    assert_eq!(db.vector_ids("c1").await.unwrap(), strs(&["a", "b", "c"]));
}

#[tokio::test]
async fn vector_ids_empty_for_unknown_collection() {
    let dir = tempfile::tempdir().unwrap();
    let db = NodeDbLite::open_at_path(dir.path(), Encryption::Plaintext)
        .await
        .unwrap();
    assert!(db.vector_ids("nope").await.unwrap().is_empty());
    db.vector_insert("c1", "a", &v(0), None).await.unwrap();
    assert!(db.vector_ids("nope").await.unwrap().is_empty());
}

/// `vector_search(k = n)` is approximate and may return fewer than n hits;
/// `vector_ids` must still list every stored id.
#[tokio::test]
async fn vector_ids_lists_all_when_search_recall_is_partial() {
    const N: usize = 2_000;
    let dir = tempfile::tempdir().unwrap();
    let db = NodeDbLite::open_at_path(dir.path(), Encryption::Plaintext)
        .await
        .unwrap();

    let mut expected: Vec<String> = Vec::with_capacity(N);
    for i in 0..N {
        let id = format!("d{i:05}");
        db.vector_insert("big", &id, &v(i), None).await.unwrap();
        expected.push(id);
    }

    let hits = db.vector_search("big", &v(0), N, None, None).await.unwrap();
    assert!(hits.len() <= N);

    assert_eq!(db.vector_ids("big").await.unwrap(), expected);
}
