// SPDX-License-Identifier: Apache-2.0

//! CRDT history compaction must not rewrite what is already on disk.
//!
//! `compact_crdt_history` runs on a timer. It used to drop each compacted
//! collection's persistence marks, so the next flush wrote every compacted
//! collection as a full snapshot in one commit. A store compacting every 30
//! minutes wrote 250-500 MB per run, more than pagedb reuses per commit, and
//! the file grew 400-600 MB an hour. One touched row forced a rewrite of its
//! whole collection.
//!
//! A collection already persisted at its current frontier now keeps its marks.
//! The flush after a compaction writes only the update since that frontier.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{Encryption, LiteConfig, NodeDbLite, PagedbStorageDefault};
use nodedb_types::document::Document;
use nodedb_types::value::Value;

const COLLECTION: &str = "blobs";
/// Rows in the working set.
const ROWS: u64 = 2_000;
/// Bytes of incompressible payload per row.
const BLOB_BYTES: usize = 1_024;
/// Compaction runs, each one write, a flush, a compaction and a flush.
const RUNS: u64 = 5;

/// Allowed store growth from run 1 to run 5: 64 pages of 4 KiB.
///
/// A run under the fix writes one update of about 1 KiB and the copy-on-write
/// path to it. That is a few pages per commit, and pagedb reuses pages freed by
/// earlier commits. Four runs without any reuse stay under 64 pages.
///
/// One full rewrite of the working set costs its whole base snapshot: at least
/// `ROWS * BLOB_BYTES` (2000 KiB), and the test asserts the base exceeds four
/// times this bound. Any run that rewrites the collection fails the bound.
const GROWTH_BOUND: u64 = 64 * 4096;

/// Open with auto-flush and pagedb auto-compaction off, so every write and
/// every reclaim in the test is one it asked for. Sync is off so the outbound
/// queue does not add bytes unrelated to CRDT state.
async fn open(path: &Path) -> Arc<NodeDbLite<PagedbStorageDefault>> {
    let storage = PagedbStorageDefault::open(path, Encryption::Plaintext)
        .await
        .expect("open storage");
    let config = LiteConfig {
        auto_flush_ms: 0,
        auto_compact_ms: 0,
        sync_enabled: false,
        ..LiteConfig::default()
    };
    NodeDbLite::open_with_config(storage, config)
        .await
        .expect("open db")
}

/// Total bytes of every file under `path`, so the measure covers whatever
/// files pagedb keeps.
fn store_bytes(path: &Path) -> u64 {
    let meta = std::fs::metadata(path).expect("stat store");
    if meta.is_file() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .expect("read store directory")
        .map(|entry| store_bytes(&entry.expect("directory entry").path()))
        .sum()
}

/// Deterministic incompressible bytes, so the snapshot size tracks the payload.
fn blob(seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..BLOB_BYTES)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

async fn put(db: &NodeDbLite<PagedbStorageDefault>, id: &str, n: i64, payload: Vec<u8>) {
    let mut doc = Document::new(id.to_string());
    doc.set("n", Value::Integer(n));
    doc.set("blob", Value::Bytes(payload));
    db.document_put(COLLECTION, doc)
        .await
        .expect("document_put");
}

#[tokio::test]
async fn repeated_compaction_does_not_grow_the_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("crdt_history_compaction.pagedb");
    let mut expected: BTreeMap<String, (i64, Vec<u8>)> = BTreeMap::new();

    {
        let db = open(&path).await;
        let empty = store_bytes(&path);

        for i in 0..ROWS {
            let id = format!("r{i}");
            let payload = blob(i);
            put(&db, &id, i as i64, payload.clone()).await;
            expected.insert(id, (i as i64, payload));
        }
        db.flush().await.expect("flush the working set");
        let base = store_bytes(&path);
        assert!(
            base.saturating_sub(empty) > 4 * GROWTH_BOUND,
            "the working set must dwarf the growth bound, or a full rewrite could pass it: \
             {} bytes written",
            base.saturating_sub(empty)
        );
        let exports = db.crdt_snapshot_export_count();

        let mut after_first_run = 0;
        for run in 1..=RUNS {
            // One touched row, as on a live store between two compactions.
            let id = format!("r{run}");
            let payload = blob(ROWS + run);
            put(&db, &id, -(run as i64), payload.clone()).await;
            expected.insert(id, (-(run as i64), payload));

            db.flush().await.expect("flush the write");
            db.compact_crdt_history().expect("compact CRDT history");
            db.flush().await.expect("flush after compaction");

            if run == 1 {
                after_first_run = store_bytes(&path);
            }
        }
        let after_last_run = store_bytes(&path);

        assert_eq!(
            db.crdt_snapshot_export_count(),
            exports,
            "{RUNS} compactions of a collection already on disk exported {} full snapshots; each \
             one rewrites the whole collection for a single touched row",
            db.crdt_snapshot_export_count() - exports
        );
        assert!(
            after_last_run <= after_first_run + GROWTH_BOUND,
            "the store grew {} bytes from run 1 to run {RUNS}, over the {GROWTH_BOUND}-byte \
             bound",
            after_last_run.saturating_sub(after_first_run)
        );
        assert!(
            after_last_run <= base + GROWTH_BOUND,
            "the store grew {} bytes over {RUNS} runs of one write each, over the \
             {GROWTH_BOUND}-byte bound",
            after_last_run.saturating_sub(base)
        );
    }

    // The base on disk predates every compaction, and four of the updates on
    // top of it were exported from a compacted document. Restore replays them
    // all and must land on the same state.
    let db = open(&path).await;
    for (id, (n, payload)) in &expected {
        let doc = db
            .document_get(COLLECTION, id)
            .await
            .expect("document_get after reopen")
            .unwrap_or_else(|| panic!("row {id} missing after reopen"));
        assert_eq!(doc.get("n"), Some(&Value::Integer(*n)), "row {id}: n");
        assert_eq!(
            doc.get("blob"),
            Some(&Value::Bytes(payload.clone())),
            "row {id}: blob"
        );
    }
}
