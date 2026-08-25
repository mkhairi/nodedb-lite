//! Regression test for NDB-AQL-37: a document indexed before a memtable spill
//! must still be searchable after flush → close → reopen.
//!
//! `FtsIndex::index_document` drains the whole memtable into an LSM segment
//! once it crosses its spill threshold, and Lite's checkpoint serializes the
//! memtable only. With the stock 100k-term threshold that combination silently
//! dropped every posting written before the spill: present in memory, absent
//! from the checkpoint, gone at the next open. Lite therefore configures its
//! indexes never to spill.
//!
//! The shape here is the one the live store hit: an ordinary document is
//! indexed first, and a later, wider one trips the threshold and drains the
//! earlier document's postings out of the memtable with it. Spilled postings
//! turn out to be invisible to search immediately, not merely after a reopen,
//! so both assertions below fail without the fix.

use nodedb_client::NodeDb;
use nodedb_lite::{Encryption, NodeDbLite, PagedbStorageDefault};
use nodedb_types::document::Document;
use nodedb_types::text_search::TextSearchParams;
use nodedb_types::value::Value;

const COLLECTION: &str = "spill";
/// Comfortably past the stock `DEFAULT_SPILL_TERMS` of 100_000.
const UNIQUE_TERMS: usize = 120_000;
/// A term unique to the ordinary document indexed before the spill.
const CANARY: &str = "singularterm";

fn canary_doc() -> Document {
    let mut doc = Document::new("canary");
    doc.set(
        "body",
        Value::String(format!("an ordinary document containing {CANARY}")),
    );
    doc
}

fn wide_doc() -> Document {
    let mut body = String::with_capacity(UNIQUE_TERMS * 8);
    for i in 0..UNIQUE_TERMS {
        body.push_str(&format!("zq{i} "));
    }
    let mut doc = Document::new("wide");
    doc.set("body", Value::String(body));
    doc
}

#[tokio::test]
async fn postings_written_before_a_spill_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spill.db");

    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .expect("open storage");
        let db = NodeDbLite::open(storage).await.expect("open NodeDbLite");

        db.document_put(COLLECTION, canary_doc())
            .await
            .expect("document_put canary");
        // Trips the stock 100k-term threshold, draining the canary's postings
        // along with its own.
        db.document_put(COLLECTION, wide_doc())
            .await
            .expect("document_put wide");

        let before = db
            .text_search(
                COLLECTION,
                "body",
                CANARY,
                10,
                TextSearchParams::default(),
                None,
            )
            .await
            .expect("text_search before flush");
        assert!(
            !before.is_empty(),
            "postings drained into a spill segment are no longer found by \
             search, before any flush or reopen (NDB-AQL-37)"
        );

        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
        .await
        .expect("reopen storage");
    let db = NodeDbLite::open(storage).await.expect("reopen NodeDbLite");

    let after = db
        .text_search(
            COLLECTION,
            "body",
            CANARY,
            10,
            TextSearchParams::default(),
            None,
        )
        .await
        .expect("text_search after restart");

    assert!(
        !after.is_empty(),
        "postings indexed before a memtable spill were lost at the checkpoint \
         — the canary document is unsearchable after reopen (NDB-AQL-37)"
    );
}
