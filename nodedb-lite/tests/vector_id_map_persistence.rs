// SPDX-License-Identifier: Apache-2.0

//! Integration tests: `vector_id_map` survives flush → close → reopen.
//!
//! Before this fix, the `vector_id_map` (which maps HNSW integer IDs back to
//! user-supplied doc_ids) was never persisted. After any restart, vector_search
//! would fall back to returning HNSW integer strings ("0", "1", ...) instead of
//! real doc_ids. These tests verify that the fix holds.
//!
//! Vectors require an explicit `flush()` to persist (HNSW is a checkpoint-only
//! index with no per-insert durability path). The id_map follows the same contract.

use nodedb_client::NodeDb;
use nodedb_lite::{Encryption, NodeDbLite, PagedbStorageDefault};

fn make_embedding(seed: f32, dim: usize) -> Vec<f32> {
    (0..dim).map(|i| seed + i as f32 * 0.001).collect()
}

#[tokio::test]
async fn vector_search_returns_real_doc_id_after_flush_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_path_buf();

    // ── Write + flush ──────────────────────────────────────────────────────────
    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();

        let embedding = make_embedding(0.1, 384);
        db.vector_insert("embeds", "my-real-doc-id", &embedding, None)
            .await
            .unwrap();

        // Explicit flush: HNSW checkpoint + id_map land on disk.
        db.flush().await.unwrap();
    }

    // ── Reopen + search ────────────────────────────────────────────────────────
    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();

    let query = make_embedding(0.1, 384);
    let results = db
        .vector_search("embeds", &query, 5, None, None)
        .await
        .unwrap();

    assert!(
        !results.is_empty(),
        "vector_search must return the indexed embedding after reopen"
    );
    assert_eq!(
        results[0].id, "my-real-doc-id",
        "vector_search must return the REAL doc_id after reopen, \
         not an HNSW integer like \"0\" — got {:?}",
        results[0].id
    );
}

#[tokio::test]
async fn vector_search_multiple_collections_preserve_ids_after_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_path_buf();

    // ── Write two collections with two docs each, flush ────────────────────────
    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();

        // alpha: doc-a0 and doc-a1
        for (i, id) in ["doc-a0", "doc-a1"].iter().enumerate() {
            let emb = make_embedding(1.0 + i as f32 * 10.0, 64);
            db.vector_insert("alpha", id, &emb, None).await.unwrap();
        }

        // beta: doc-b0 and doc-b1
        for (i, id) in ["doc-b0", "doc-b1"].iter().enumerate() {
            let emb = make_embedding(100.0 + i as f32 * 10.0, 64);
            db.vector_insert("beta", id, &emb, None).await.unwrap();
        }

        db.flush().await.unwrap();
    }

    // ── Reopen + verify each collection independently ──────────────────────────
    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();

    // Query close to doc-a0's embedding.
    let query_a = make_embedding(1.0, 64);
    let results_a = db
        .vector_search("alpha", &query_a, 2, None, None)
        .await
        .unwrap();
    assert!(
        !results_a.is_empty(),
        "alpha search must return results after reopen"
    );
    let ids_a: Vec<&str> = results_a.iter().map(|r| r.id.as_str()).collect();
    for id in &ids_a {
        assert!(
            id.starts_with("doc-a"),
            "alpha results must have doc-a* ids, not HNSW integers or beta ids — got {id}"
        );
    }

    // Query close to doc-b0's embedding.
    let query_b = make_embedding(100.0, 64);
    let results_b = db
        .vector_search("beta", &query_b, 2, None, None)
        .await
        .unwrap();
    assert!(
        !results_b.is_empty(),
        "beta search must return results after reopen"
    );
    let ids_b: Vec<&str> = results_b.iter().map(|r| r.id.as_str()).collect();
    for id in &ids_b {
        assert!(
            id.starts_with("doc-b"),
            "beta results must have doc-b* ids, not HNSW integers or alpha ids — got {id}"
        );
    }
}

/// A re-inserted id comes back from a reopen as one node, scored against
/// its latest vector.
#[tokio::test]
async fn reinserted_id_appears_once_after_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_path_buf();

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        db.vector_insert("embeds", "bg", &[0.0, 0.0, 1.0], None)
            .await
            .unwrap();
        db.vector_insert("embeds", "x", &[1.0, 0.0, 0.0], None)
            .await
            .unwrap();
        db.vector_insert("embeds", "x", &[0.0, 1.0, 0.0], None)
            .await
            .unwrap();
        db.flush().await.unwrap();
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    let results = db
        .vector_search("embeds", &[0.0, 1.0, 0.0], 10, None, None)
        .await
        .unwrap();
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids.len(), 2, "one result per id, got {ids:?}");
    assert_eq!(ids[0], "x", "the latest vector of x is the nearest");
    assert!(results[0].distance.abs() < 1e-5);
}

/// Collections whose names share a prefix keep separate bindings across a
/// reopen: an `allowed_ids` search never resolves another collection's id.
#[tokio::test]
async fn prefix_colliding_collections_stay_separate_after_reopen() {
    use std::collections::HashSet;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_path_buf();

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        db.vector_insert("chat", "a", &[1.0, 0.0], None)
            .await
            .unwrap();
        db.vector_insert("chat2", "b", &[1.0, 0.0], None)
            .await
            .unwrap();
        db.flush().await.unwrap();
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();

    let allowed: HashSet<String> = std::iter::once("b".to_owned()).collect();
    let restricted = db
        .vector_search("chat", &[1.0, 0.0], 10, None, Some(&allowed))
        .await
        .unwrap();
    assert!(restricted.is_empty(), "got {restricted:?}");

    let all = db
        .vector_search("chat", &[1.0, 0.0], 10, None, None)
        .await
        .unwrap();
    let ids: Vec<&str> = all.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["a"]);
}

/// A segment backs its index positionally, so ids inserted out of key
/// order must each come back from a reopen scored against their own vector.
#[tokio::test]
async fn ids_inserted_out_of_key_order_keep_their_vectors_after_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_path_buf();
    let rows: [(&str, [f32; 3]); 3] = [
        ("c", [1.0, 0.0, 0.0]),
        ("a", [0.0, 1.0, 0.0]),
        ("b", [0.0, 0.0, 1.0]),
    ];

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        for (id, v) in &rows {
            db.vector_insert("ordered", id, v, None).await.unwrap();
        }
        db.flush().await.unwrap();
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    for (id, v) in &rows {
        let hits = db.vector_search("ordered", v, 1, None, None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, *id, "exact query for {id} returns {id}");
        assert!(
            hits[0].distance.abs() < 1e-5,
            "{id} is scored against its own vector, got {}",
            hits[0].distance
        );
    }
}

/// A base index rebuilt from durable rows on reopen holds only base
/// vectors; the named index is rebuilt from its own rows.
#[tokio::test]
async fn base_rebuild_after_reopen_excludes_named_field_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_path_buf();

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        db.vector_insert("chat", "a", &[1.0, 0.0], None)
            .await
            .unwrap();
        db.vector_insert_field("chat", "emb", "b", &[1.0, 0.0], None)
            .await
            .unwrap();
        // No flush: the reopen rebuilds both indexes from durable rows.
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    let base = db
        .vector_search("chat", &[1.0, 0.0], 10, None, None)
        .await
        .unwrap();
    let ids: Vec<&str> = base.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["a"]);
    let named = db
        .vector_search_field("chat", "emb", &[1.0, 0.0], 10, None)
        .await
        .unwrap();
    let ids: Vec<&str> = named.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["b"]);
}
