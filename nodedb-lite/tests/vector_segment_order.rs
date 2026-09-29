// SPDX-License-Identifier: Apache-2.0

//! A vector segment must attach each vector to the node it belongs to.
//!
//! The graph checkpoint numbers nodes in insertion order, and attaching a
//! segment maps its entry `i` to node `i`. The segment used to be built from
//! the durable rows, which come back in document-id key order. Whenever the
//! two orders differed, a reopened or lazily reloaded index scored every node
//! against another document's vector and returned wrong results, with no
//! error anywhere.
//!
//! Every test here inserts ids out of key order on purpose.

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::nodedb::FlushArtifact;
use nodedb_lite::{Encryption, LiteConfig, NodeDbLite, PagedbStorageDefault, StorageEngine};
use nodedb_vector::segment_backing::VectorSegmentBacking;

const COLL: &str = "docs";

/// Ids with distinct directions, inserted in this order. Key order is
/// `a, b, c, d, e`, which differs from it at every position but the last two.
const DOCS: [(&str, [f32; 3]); 5] = [
    ("b", [1.0, 0.0, 0.0]),
    ("a", [0.0, 1.0, 0.0]),
    ("e", [0.0, 0.0, 1.0]),
    ("c", [1.0, 1.0, 0.0]),
    ("d", [0.0, 1.0, 1.0]),
];

/// Upper bound on the distance of an exact match under the default metric.
const EXACT: f32 = 1e-4;

async fn open(path: &std::path::Path) -> Arc<NodeDbLite<PagedbStorageDefault>> {
    let storage = PagedbStorageDefault::open(path, Encryption::Plaintext)
        .await
        .expect("open storage");
    let config = LiteConfig {
        auto_flush_ms: 0,
        ..LiteConfig::default()
    };
    NodeDbLite::open_with_config(storage, config)
        .await
        .expect("open db")
}

async fn insert_docs(db: &NodeDbLite<PagedbStorageDefault>) {
    for (id, v) in DOCS {
        db.vector_insert(COLL, id, &v, None)
            .await
            .expect("vector_insert");
    }
}

/// `(id, distance)` of the nearest hit for `query`.
async fn top1(db: &NodeDbLite<PagedbStorageDefault>, query: &[f32]) -> (String, f32) {
    let hits = db
        .vector_search(COLL, query, 3, None, None)
        .await
        .expect("vector_search");
    let first = hits.first().expect("at least one hit");
    (first.id.clone(), first.distance)
}

/// Every document's own vector finds that document first, at distance ~0.
async fn assert_every_doc_finds_itself(db: &NodeDbLite<PagedbStorageDefault>, when: &str) {
    for (id, v) in DOCS {
        let (got, distance) = top1(db, &v).await;
        assert_eq!(
            got, id,
            "{when}: the vector of {id:?} found {got:?} first — a node is scored \
             against another document's vector"
        );
        assert!(
            distance < EXACT,
            "{when}: {id:?} matched its own vector at distance {distance}"
        );
    }
}

/// Whether the index was attached to its stored checkpoint and segment
/// rather than rebuilt. A rebuild leaves both artifacts dirty.
fn attached_clean(db: &NodeDbLite<PagedbStorageDefault>) -> bool {
    !db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, COLL)
        && !db.flush_artifact_is_dirty(FlushArtifact::VectorSegment, COLL)
}

/// Out-of-order ids survive a flush and reopen with their own vectors.
#[tokio::test]
async fn out_of_order_ids_keep_their_vectors_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reopen.pagedb");
    {
        let db = open(&path).await;
        insert_docs(&db).await;
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let db = open(&path).await;
    assert!(
        attached_clean(&db),
        "a segment written by this build must attach without a rebuild"
    );
    let (got, distance) = top1(&db, &[1.0, 0.0, 0.0]).await;
    assert_eq!(got, "b", "[1,0,0] belongs to \"b\"");
    assert!(distance < EXACT, "\"b\" matched at distance {distance}");
    assert_every_doc_finds_itself(&db, "after reopen").await;
}

/// Eviction writes the segment, and the lazy load attaches it.
#[tokio::test]
async fn out_of_order_ids_keep_their_vectors_across_eviction() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open(&dir.path().join("evict.pagedb")).await;
    insert_docs(&db).await;
    db.flush().await.expect("flush");

    assert_eq!(db.evict_collections(1).await.expect("evict"), 1);
    let (got, _) = top1(&db, &[1.0, 0.0, 0.0]).await;
    assert_eq!(got, "b", "[1,0,0] belongs to \"b\" after the lazy load");
    assert!(
        !db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, COLL),
        "the lazy load must attach the evicted segment, not rebuild"
    );
    assert_every_doc_finds_itself(&db, "after eviction").await;
}

/// A segment in the pre-fix format — key order, no stamps — is detected,
/// the index is rebuilt, and the next flush writes a stamped segment.
#[tokio::test]
async fn an_unstamped_segment_is_rebuilt_and_then_rewritten_with_stamps() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("old_format.pagedb");
    {
        let db = open(&path).await;
        insert_docs(&db).await;
        db.flush().await.expect("flush");

        // What every earlier build wrote: vectors in key order, zero stamps.
        let mut key_ordered: Vec<(&str, [f32; 3])> = DOCS.to_vec();
        key_ordered.sort_by_key(|(id, _)| *id);
        let vectors: Vec<Vec<f32>> = key_ordered.iter().map(|(_, v)| v.to_vec()).collect();
        db.storage()
            .as_vector_segment_ext()
            .expect("pagedb has vector segments")
            .write_vector_segment(COLL, 3, &vectors, &[])
            .await
            .expect("overwrite segment");
        db.shutdown().await;
    }

    {
        let db = open(&path).await;
        assert!(
            db.flush_artifact_is_dirty(FlushArtifact::VectorSegment, COLL),
            "an unstamped segment must be refused, leaving the segment dirty"
        );
        assert_every_doc_finds_itself(&db, "after the refused segment").await;

        db.flush().await.expect("flush after rebuild");
        let segment = db
            .storage()
            .as_vector_segment_ext()
            .expect("pagedb has vector segments")
            .open_vector_segment(COLL)
            .await
            .expect("open segment")
            .expect("segment exists");
        assert_eq!(segment.len(), DOCS.len());
        for slot in 0..segment.len() as u32 {
            assert_ne!(
                segment.get_surrogate(slot),
                Some(0),
                "slot {slot} must carry a stamp after the rewrite"
            );
        }
        db.shutdown().await;
    }

    let db = open(&path).await;
    assert!(
        attached_clean(&db),
        "the rewritten segment must attach without another rebuild"
    );
    assert_every_doc_finds_itself(&db, "after the rewrite").await;
}

/// Tombstoned slots keep their vector in the segment, so a segment with
/// deletes, upserts, and re-inserts still attaches and answers correctly.
#[tokio::test]
async fn tombstones_upserts_and_reinserts_attach_without_a_rebuild() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tombstones.pagedb");

    // Delete: slot 0 ("a") is tombstoned, slot 1 ("b") is live.
    {
        let db = open(&path).await;
        db.vector_insert(COLL, "a", &[1.0, 0.0, 0.0], None)
            .await
            .expect("insert a");
        db.vector_insert(COLL, "b", &[0.0, 1.0, 0.0], None)
            .await
            .expect("insert b");
        db.vector_delete(COLL, "a").await.expect("delete a");
        db.flush().await.expect("flush");
        db.shutdown().await;
    }
    {
        let db = open(&path).await;
        assert!(
            attached_clean(&db),
            "a segment with a tombstoned slot must attach without a rebuild"
        );
        let (got, distance) = top1(&db, &[0.0, 1.0, 0.0]).await;
        assert_eq!(got, "b");
        assert!(distance < EXACT);
        let hits = db
            .vector_search(COLL, &[1.0, 0.0, 0.0], 3, None, None)
            .await
            .expect("vector_search");
        assert!(
            hits.iter().all(|h| h.id != "a"),
            "a deleted document must stay deleted: {hits:?}"
        );

        // Upsert: "b" is tombstoned at slot 1 and appended at slot 2.
        db.vector_insert(COLL, "b", &[0.0, 0.0, 1.0], None)
            .await
            .expect("upsert b");
        db.flush().await.expect("flush");
        db.shutdown().await;
    }
    {
        let db = open(&path).await;
        assert!(attached_clean(&db), "an upserted segment must attach");
        let (got, distance) = top1(&db, &[0.0, 0.0, 1.0]).await;
        assert_eq!(got, "b", "the upsert's vector belongs to \"b\"");
        assert!(distance < EXACT);
        let hits = db
            .vector_search(COLL, &[0.0, 1.0, 0.0], 3, None, None)
            .await
            .expect("vector_search");
        assert!(
            hits.iter().all(|h| h.distance > 0.5),
            "the superseded vector of \"b\" must not match anymore: {hits:?}"
        );

        // Re-insert the deleted id: slot 0 is released, slot 3 holds "a".
        db.vector_insert(COLL, "a", &[1.0, 1.0, 0.0], None)
            .await
            .expect("re-insert a");
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let db = open(&path).await;
    assert!(
        attached_clean(&db),
        "a re-inserted id's segment must attach"
    );
    let (got, distance) = top1(&db, &[1.0, 1.0, 0.0]).await;
    assert_eq!(got, "a", "the re-inserted vector belongs to \"a\"");
    assert!(distance < EXACT);
    let (got, _) = top1(&db, &[0.0, 0.0, 1.0]).await;
    assert_eq!(got, "b");
}

/// `TRUNCATE` unlinks the stored segment, so the same ids inserted again
/// with new vectors cannot pick up the old ones.
#[tokio::test]
async fn truncate_unlinks_the_segment_and_reinserted_ids_get_new_vectors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("truncate.pagedb");
    {
        let db = open(&path).await;
        insert_docs(&db).await;
        db.flush().await.expect("flush");

        db.execute_sql(&format!("TRUNCATE {COLL}"), &[])
            .await
            .expect("truncate");
        let segment = db
            .storage()
            .as_vector_segment_ext()
            .expect("pagedb has vector segments")
            .open_vector_segment(COLL)
            .await
            .expect("open segment");
        assert!(
            segment.is_none(),
            "TRUNCATE must unlink the stored vector segment"
        );

        // The same ids in the same order, each with another document's
        // vector: every stamp of the old segment would match again.
        for (i, (id, _)) in DOCS.iter().enumerate() {
            let (_, v) = DOCS[(i + 1) % DOCS.len()];
            db.vector_insert(COLL, id, &v, None)
                .await
                .expect("re-insert");
        }
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let db = open(&path).await;
    for (i, (id, _)) in DOCS.iter().enumerate() {
        let (_, v) = DOCS[(i + 1) % DOCS.len()];
        let (got, distance) = top1(&db, &v).await;
        assert_eq!(
            got, *id,
            "the re-inserted vector of {id:?} must be the new one"
        );
        assert!(distance < EXACT);
    }
}
