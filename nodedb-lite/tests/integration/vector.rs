// SPDX-License-Identifier: Apache-2.0

use super::fixtures::open_test_db;
use nodedb_client::NodeDb;

// ─── 1K Vector Insert + Search ───────────────────────────────────────

/// Correctness-only counterpart of the 1k×32d batch workload.
/// Scale benchmark: `nodedb-bench/benches/micro/lite_vector.rs`.
#[tokio::test]
async fn vector_batch_insert_and_search_correctness() {
    let db = open_test_db().await;
    let n = 50;
    let dim = 32;

    let vectors: Vec<(String, Vec<f32>)> = (0..n)
        .map(|i| {
            let emb: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32) * 0.001).collect();
            (format!("v{i}"), emb)
        })
        .collect();

    let refs: Vec<(&str, &[f32])> = vectors
        .iter()
        .map(|(id, emb)| (id.as_str(), emb.as_slice()))
        .collect();

    db.batch_vector_insert("vecs", &refs).await.unwrap();

    let query: Vec<f32> = (0..dim).map(|d| ((25 * dim + d) as f32) * 0.001).collect();
    let results = db
        .vector_search("vecs", &query, 10, None, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 10);
    for w in results.windows(2) {
        assert!(
            w[0].distance <= w[1].distance,
            "results not sorted by distance"
        );
    }
    assert!(
        results[0].distance < 0.5,
        "top result distance {} is too large",
        results[0].distance
    );
}
