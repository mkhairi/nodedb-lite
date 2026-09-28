// SPDX-License-Identifier: Apache-2.0

//! Secondary indexes on typed key-value rows.
//!
//! Every KV write — SQL or the public API, put, update, delete, expiry —
//! keeps the collection's index entries current, in the same storage batch
//! as the row. A lookup on an indexed column reads the index's candidates.
//!
//! Run with:
//!   cargo nextest run -p nodedb-lite --test kv_secondary_index

use std::collections::HashMap;
use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::nodedb_query::msgpack_scan::{KvBodyShape, row_to_kv_body};
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::value::Value;

type Db = Arc<NodeDbLite<PagedbStorageMem>>;

async fn open_db() -> Db {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("open_in_memory");
    NodeDbLite::open(storage).await.expect("NodeDbLite::open")
}

async fn exec(db: &Db, sql: &str) {
    db.execute_sql(sql, &[])
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// The sorted `key` column of `sql`.
async fn keys(db: &Db, sql: &str) -> Vec<String> {
    let result = db
        .execute_sql(sql, &[])
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let at = result
        .columns
        .iter()
        .position(|c| c == "key")
        .unwrap_or_else(|| panic!("{sql}: no key column in {:?}", result.columns));
    let mut keys: Vec<String> = result
        .rows
        .iter()
        .map(|row| match &row[at] {
            Value::String(k) => k.clone(),
            other => panic!("{sql}: key column is {other:?}"),
        })
        .collect();
    keys.sort();
    keys
}

/// A typed row body `{color}` for the public KV API.
fn color_body(color: &str) -> Vec<u8> {
    let row = Value::Object(HashMap::from([(
        "color".to_string(),
        Value::String(color.to_string()),
    )]));
    row_to_kv_body(&row, KvBodyShape::Map).expect("encode body")
}

/// A `paints` KV collection with an index on `color`.
async fn paints(unique: bool) -> Db {
    let db = open_db().await;
    exec(
        &db,
        "CREATE COLLECTION paints (key TEXT PRIMARY KEY, color TEXT) WITH (engine='kv')",
    )
    .await;
    let kind = if unique { "UNIQUE INDEX" } else { "INDEX" };
    exec(
        &db,
        &format!("CREATE {kind} idx_paints_color ON paints (color)"),
    )
    .await;
    db
}

fn by_color(color: &str) -> String {
    format!("SELECT key, color FROM paints WHERE color = '{color}'")
}

#[tokio::test]
async fn kv_put_after_register_index_is_lookup_reachable() {
    let db = paints(false).await;
    exec(&db, "INSERT INTO paints (key, color) VALUES ('a', 'red')").await;
    db.kv_put("paints", "b", &color_body("red"))
        .await
        .expect("api put");
    db.kv_put("paints", "c", &color_body("blue"))
        .await
        .expect("api put");

    assert_eq!(keys(&db, &by_color("red")).await, vec!["a", "b"]);
    assert_eq!(keys(&db, &by_color("blue")).await, vec!["c"]);
    assert_eq!(
        keys(&db, "SELECT key, color FROM paints WHERE color > 'green'").await,
        vec!["a", "b"]
    );
}

#[tokio::test]
async fn kv_delete_removes_index_entry() {
    let db = paints(true).await;
    exec(&db, "INSERT INTO paints (key, color) VALUES ('a', 'red')").await;
    db.kv_put("paints", "b", &color_body("blue"))
        .await
        .expect("api put");
    exec(&db, "DELETE FROM paints WHERE key = 'a'").await;
    db.kv_delete("paints", "b").await.expect("api delete");

    assert!(keys(&db, &by_color("red")).await.is_empty());
    assert!(keys(&db, &by_color("blue")).await.is_empty());
    // The unique values are free again.
    exec(&db, "INSERT INTO paints (key, color) VALUES ('c', 'red')").await;
    db.kv_put("paints", "d", &color_body("blue"))
        .await
        .expect("freed value");
}

#[tokio::test]
async fn kv_update_moves_index_entry() {
    let db = paints(true).await;
    exec(&db, "INSERT INTO paints (key, color) VALUES ('a', 'red')").await;
    exec(&db, "UPDATE paints SET color = 'blue' WHERE key = 'a'").await;

    assert!(keys(&db, &by_color("red")).await.is_empty());
    assert_eq!(keys(&db, &by_color("blue")).await, vec!["a"]);

    // The unique index refuses a second holder, through SQL and the API,
    // and the refused write leaves nothing behind.
    let err = db
        .execute_sql("INSERT INTO paints (key, color) VALUES ('b', 'blue')", &[])
        .await
        .expect_err("duplicate color");
    assert!(err.to_string().contains("unique"), "{err}");
    let err = db
        .kv_put("paints", "c", &color_body("blue"))
        .await
        .expect_err("duplicate color through the API");
    assert!(err.to_string().contains("unique"), "{err}");
    assert_eq!(keys(&db, &by_color("blue")).await, vec!["a"]);
    assert!(db.kv_get("paints", "c").await.expect("get").is_none());
}

#[tokio::test]
async fn kv_expired_key_leaves_index() {
    let db = paints(true).await;
    db.kv_put_with_ttl("paints", "short", &color_body("red"), 1)
        .await
        .expect("ttl put");
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    assert!(keys(&db, &by_color("red")).await.is_empty());
    // An expired row holds no value a unique index counts.
    db.kv_put("paints", "long", &color_body("red"))
        .await
        .expect("the expired row no longer holds the value");
    assert_eq!(keys(&db, &by_color("red")).await, vec!["long"]);
}
