// SPDX-License-Identifier: Apache-2.0

//! `SELECT ... WHERE` over a bitemporal document collection: the WHERE and the
//! `LIMIT` are applied as the scan produces rows, so these tests pin the
//! answers that streaming must not change.

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::value::Value;

/// A bitemporal collection holding `count` documents `d000..`, every fifth one
/// tiered `gold` and the rest `bronze`, each with an ascending `n`.
async fn seeded_db(collection: &str, count: usize) -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    db.execute_sql(
        &format!("CREATE COLLECTION {collection} WITH (bitemporal=true)"),
        &[],
    )
    .await
    .unwrap();

    for i in 0..count {
        let mut doc = Document::new(format!("d{i:03}"));
        doc.set(
            "tier",
            Value::String(if i % 5 == 0 { "gold" } else { "bronze" }.to_owned()),
        );
        doc.set("n", Value::Integer(i as i64));
        db.document_put(collection, doc).await.unwrap();
    }
    db
}

fn ids(result: &nodedb_types::result::QueryResult) -> Vec<String> {
    let idx = result
        .columns
        .iter()
        .position(|c| c == "id")
        .expect("id column");
    result
        .rows
        .iter()
        .map(|r| match &r[idx] {
            Value::String(s) => s.clone(),
            other => panic!("id must be a string, got {other:?}"),
        })
        .collect()
}

/// A WHERE returns only its matches, and a LIMIT on top of it takes matching
/// rows — not the first rows of the collection.
#[tokio::test]
async fn where_then_limit_takes_matching_rows() {
    let db = seeded_db("scan_where", 50).await;

    let all = db
        .execute_sql("SELECT id FROM scan_where WHERE tier = 'gold'", &[])
        .await
        .unwrap();
    assert_eq!(all.rows.len(), 10, "every fifth of 50 documents is gold");

    let limited = db
        .execute_sql("SELECT id FROM scan_where WHERE tier = 'gold' LIMIT 3", &[])
        .await
        .unwrap();
    assert_eq!(
        ids(&limited),
        vec!["d000", "d005", "d010"],
        "LIMIT applies after the WHERE, so it takes three gold rows"
    );
}

/// OFFSET skips matching rows before LIMIT takes them, even though the scan
/// stops as soon as it holds `offset + limit` of them.
#[tokio::test]
async fn offset_skips_before_limit_takes() {
    let db = seeded_db("scan_offset", 50).await;

    let page = db
        .execute_sql(
            "SELECT id FROM scan_offset WHERE tier = 'gold' LIMIT 2 OFFSET 2",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(ids(&page), vec!["d010", "d015"]);
}

/// ORDER BY chooses from the whole collection: a LIMIT under a sort must not
/// end the scan early, or the row the sort wants — here the last one the scan
/// reaches — would be missed.
#[tokio::test]
async fn order_by_limit_still_sees_every_row() {
    let db = seeded_db("scan_order", 20).await;

    // Sorts last by id, so a scan that stopped at its first match would never
    // see it.
    let mut last = Document::new("zzz_last");
    last.set("tier", Value::String("gold".to_owned()));
    last.set("n", Value::Integer(-1));
    db.document_put("scan_order", last).await.unwrap();

    let top = db
        .execute_sql(
            "SELECT id FROM scan_order WHERE tier = 'gold' ORDER BY id DESC LIMIT 1",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(
        ids(&top),
        vec!["zzz_last"],
        "the sort must pick from every matching row, not the first one scanned"
    );
}

/// The scan reads live rows only: an updated document contributes its latest
/// version, and a deleted one contributes nothing.
#[tokio::test]
async fn scan_returns_live_versions_only() {
    let db = seeded_db("scan_live", 5).await;

    // d001 moves from bronze to gold; d000 (gold) is deleted.
    let mut updated = Document::new("d001");
    updated.set("tier", Value::String("gold".to_owned()));
    updated.set("n", Value::Integer(1));
    db.document_put("scan_live", updated).await.unwrap();
    db.document_delete("scan_live", "d000").await.unwrap();

    let gold = db
        .execute_sql("SELECT id FROM scan_live WHERE tier = 'gold'", &[])
        .await
        .unwrap();
    assert_eq!(
        ids(&gold),
        vec!["d001"],
        "the superseded bronze version and the deleted document are both gone"
    );
}
