// SPDX-License-Identifier: Apache-2.0

use nodedb_client::NodeDb;
use nodedb_lite::{HybridSearchParams, NodeDbLite, PagedbStorageMem};
use nodedb_types::{Document, Value};
use std::collections::HashSet;

#[tokio::test]
async fn fusion_boosts_documents_in_both_sources() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    for (id, embedding, title) in [
        ("both", [1.0, 0.0], "rust"),
        ("vector", [0.8, 0.2], "other"),
        ("text", [0.0, 1.0], "rust"),
    ] {
        let mut doc = Document::new(id);
        doc.set("title", Value::String(title.into()));
        db.document_put("docs", doc).await.unwrap();
        db.vector_insert("docs", id, &embedding, None)
            .await
            .unwrap();
    }
    let results = db
        .hybrid_search(&HybridSearchParams {
            collection: "docs",
            query_embedding: &[1.0, 0.0],
            query_text: "rust",
            vector_k: 2,
            text_k: 2,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(results[0].id, "both");
    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0].metadata.get("title"),
        Some(&Value::String("rust".into()))
    );
    let allowed = HashSet::from(["text".to_owned()]);
    let restricted = db
        .hybrid_search(&HybridSearchParams {
            collection: "docs",
            query_embedding: &[1.0, 0.0],
            query_text: "rust",
            allowed_ids: Some(&allowed),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(restricted.len(), 1);
    assert_eq!(restricted[0].id, "text");
}

#[tokio::test]
async fn allowed_text_candidates_rank_beyond_global_candidate_limit() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    for i in 0..32 {
        let mut doc = Document::new(format!("higher-{i}"));
        doc.set("title", Value::String("rust".into()));
        db.document_put("docs", doc).await.unwrap();
    }
    let mut doc = Document::new("allowed");
    doc.set(
        "title",
        Value::String(format!("rust {}", "padding ".repeat(200))),
    );
    doc.set("body", Value::String("fieldtoken".into()));
    db.document_put("docs", doc).await.unwrap();
    let allowed = HashSet::from(["allowed".to_owned(), "unknown".to_owned()]);
    let params = HybridSearchParams {
        collection: "docs",
        query_text: "rust",
        vector_k: 0,
        text_k: 1,
        top_k: 1,
        allowed_ids: Some(&allowed),
        ..Default::default()
    };
    assert_eq!(db.hybrid_search(&params).await.unwrap()[0].id, "allowed");
    let scoped = HybridSearchParams {
        query_text: "fieldtoken",
        text_field: "title",
        ..params
    };
    assert!(db.hybrid_search(&scoped).await.unwrap().is_empty());
    assert_eq!(
        db.hybrid_search(&HybridSearchParams {
            text_field: "body",
            ..scoped
        })
        .await
        .unwrap()[0]
            .id,
        "allowed"
    );
}

#[tokio::test]
async fn empty_unknown_and_zero_result_limits_return_empty() {
    let db = NodeDbLite::open(PagedbStorageMem::open_in_memory().await.unwrap())
        .await
        .unwrap();
    db.vector_insert("docs", "existing", &[1.0, 0.0], None)
        .await
        .unwrap();
    for allowed in [HashSet::new(), HashSet::from(["unknown".to_owned()])] {
        assert!(
            db.hybrid_search(&HybridSearchParams {
                collection: "docs",
                query_embedding: &[1.0, 0.0],
                allowed_ids: Some(&allowed),
                ..Default::default()
            })
            .await
            .unwrap()
            .is_empty()
        );
    }
    assert!(
        db.hybrid_search(&HybridSearchParams {
            top_k: 0,
            ..Default::default()
        })
        .await
        .unwrap()
        .is_empty()
    );
}
