//! A vector of the wrong dimension is a typed data error on Lite.
//!
//! - A search whose query has another width than the index fails with the
//!   `DATA_EXCEPTION` code, on a populated index and an empty one, and the
//!   engine keeps serving: the next search succeeds.
//! - An insert of another width fails with the same code and inserts nothing.
//!
//! Run with:
//!   cargo nextest run -p nodedb-lite --test vector_dimension_errors

use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::error::ErrorCode;

async fn open_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("open_in_memory");
    NodeDbLite::open(storage).await.expect("NodeDbLite::open")
}

#[tokio::test]
async fn wrong_dimension_search_is_a_data_exception_and_the_engine_keeps_serving() {
    let db = open_db().await;
    let rows: Vec<(String, Vec<f32>)> = (0..20)
        .map(|i| (format!("v{i}"), vec![i as f32, 0.0, 1.0]))
        .collect();
    let refs: Vec<(&str, &[f32])> = rows
        .iter()
        .map(|(id, v)| (id.as_str(), v.as_slice()))
        .collect();
    db.batch_vector_insert("dims", &refs)
        .await
        .expect("seed vectors");

    let err = db
        .vector_search("dims", &[1.0, 0.0], 5, None, None)
        .await
        .expect_err("a 2-wide query against a 3-wide index must fail");
    assert_eq!(err.code(), ErrorCode::DATA_EXCEPTION, "{err}");
    assert!(
        err.to_string()
            .contains("vector dimension mismatch: expected 3, got 2"),
        "{err}"
    );

    let hits = db
        .vector_search("dims", &[3.0, 0.0, 1.0], 3, None, None)
        .await
        .expect("the next search succeeds");
    assert_eq!(hits.len(), 3);
}

#[tokio::test]
async fn wrong_dimension_insert_is_a_data_exception() {
    let db = open_db().await;
    db.batch_vector_insert("dims_insert", &[("a", &[1.0_f32, 0.0, 0.0][..])])
        .await
        .expect("seed one vector");

    let err = db
        .batch_vector_insert("dims_insert", &[("b", &[1.0_f32, 0.0][..])])
        .await
        .expect_err("a 2-wide vector must not enter a 3-wide index");
    assert_eq!(err.code(), ErrorCode::DATA_EXCEPTION, "{err}");

    let hits = db
        .vector_search("dims_insert", &[1.0, 0.0, 0.0], 5, None, None)
        .await
        .expect("search after the refused insert");
    assert_eq!(hits.len(), 1, "the refused vector is not searchable");
}
