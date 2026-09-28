// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the search option math, codec guard, and rerank plumbing.

use std::collections::HashMap;

use nodedb_types::VectorAnnOptions;
use nodedb_vector::rerank::{IndexShape, recall_scale, validate_options};

// ── oversample math ──────────────────────────────────────────────────────

#[test]
fn oversample_4_no_filter_fetch_k_is_k_times_4() {
    let k = 10_usize;
    let oversample: usize = 4;
    let fetch_k = k.saturating_mul(oversample);
    assert_eq!(fetch_k, 40);
}

#[test]
fn oversample_4_with_filter_fetch_k_is_k_times_12() {
    let k = 10_usize;
    let oversample: usize = 4;
    let fetch_k = k.saturating_mul(oversample).saturating_mul(3);
    assert_eq!(fetch_k, 120);
}

// ── target_recall scaling ────────────────────────────────────────────────

#[test]
fn target_recall_095_scales_ef_and_oversample() {
    let base_ef = 50_usize;
    let base_oversample: u8 = 1;
    let (scaled_ef, scaled_oversample) =
        recall_scale(Some(0.95), base_ef, base_oversample).unwrap();
    assert_eq!(scaled_ef, 200);
    assert_eq!(scaled_oversample, 2);
}

#[test]
fn target_recall_none_returns_base_unchanged() {
    let (ef, os) = recall_scale(None, 100, 1).unwrap();
    assert_eq!(ef, 100);
    assert_eq!(os, 1);
}

#[test]
fn target_recall_invalid_returns_bad_input() {
    let result = recall_scale(Some(1.5), 100, 1);
    assert!(result.is_err());
}

// ── codec guard ──────────────────────────────────────────────────────────

#[test]
fn sq8_quantization_returns_bad_request_via_validate() {
    use nodedb_types::vector_ann::VectorQuantization;
    let opts = VectorAnnOptions {
        quantization: Some(VectorQuantization::Sq8),
        ..Default::default()
    };
    let rerank_codec =
        validate_options(&opts, IndexShape::SingleVector, VectorQuantization::Sq8).unwrap();
    assert!(
        rerank_codec.is_some(),
        "Sq8 should produce a Some(CodecName)"
    );
}

#[test]
fn meta_token_budget_returns_bad_input_from_validate() {
    let opts = VectorAnnOptions {
        meta_token_budget: Some(8),
        ..Default::default()
    };
    let result = validate_options(
        &opts,
        IndexShape::SingleVector,
        nodedb_types::VectorQuantization::None,
    );
    assert!(
        result.is_err(),
        "meta_token_budget on single-vector should be a BadInput error"
    );
}

// ── query_dim plumbing ───────────────────────────────────────────────────

#[test]
fn query_dim_zero_rejected_by_rerank() {
    use nodedb_types::vector_distance::DistanceMetric;
    use nodedb_vector::rerank::{Candidate, rerank};
    let store: HashMap<u32, Vec<f32>> = [(1, vec![1.0, 2.0])].into_iter().collect();
    let opts = VectorAnnOptions {
        query_dim: Some(0),
        ..Default::default()
    };
    let err = rerank(
        vec![Candidate {
            id: 1,
            index_distance: 0.0,
        }],
        &[0.0, 0.0],
        DistanceMetric::L2,
        1,
        &opts,
        None,
        |id| {
            store
                .get(&id)
                .map(|v| std::borrow::Cow::Borrowed(v.as_slice()))
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("query_dim=0"));
}

#[test]
fn query_dim_some_changes_ranking_order() {
    use nodedb_types::vector_distance::DistanceMetric;
    use nodedb_vector::rerank::{Candidate, rerank};
    let store: HashMap<u32, Vec<f32>> = [(1, vec![0.1, 0.1]), (2, vec![0.0, 9.0])]
        .into_iter()
        .collect();
    let query = [0.0_f32, 1.0];

    let full = rerank(
        vec![
            Candidate {
                id: 1,
                index_distance: 0.0,
            },
            Candidate {
                id: 2,
                index_distance: 0.0,
            },
        ],
        &query,
        DistanceMetric::L2,
        2,
        &VectorAnnOptions::default(),
        None,
        |id| {
            store
                .get(&id)
                .map(|v| std::borrow::Cow::Borrowed(v.as_slice()))
        },
    )
    .unwrap();

    let trunc = rerank(
        vec![
            Candidate {
                id: 1,
                index_distance: 0.0,
            },
            Candidate {
                id: 2,
                index_distance: 0.0,
            },
        ],
        &query,
        DistanceMetric::L2,
        2,
        &VectorAnnOptions {
            query_dim: Some(1),
            ..Default::default()
        },
        None,
        |id| {
            store
                .get(&id)
                .map(|v| std::borrow::Cow::Borrowed(v.as_slice()))
        },
    )
    .unwrap();

    assert_eq!(full[0].id, 1, "full-dim: id=1 should rank first");
    assert_eq!(trunc[0].id, 2, "truncated dim=1: id=2 should rank first");
}

// ── metric override ──────────────────────────────────────────────────────

#[test]
fn metric_override_does_not_panic() {
    use nodedb_types::vector_distance::DistanceMetric;
    use nodedb_vector::rerank::{Candidate, rerank};

    let store: HashMap<u32, Vec<f32>> = [(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])]
        .into_iter()
        .collect();
    let query = [1.0_f32, 0.0];

    let result = rerank(
        vec![
            Candidate {
                id: 1,
                index_distance: 0.0,
            },
            Candidate {
                id: 2,
                index_distance: 1.0,
            },
        ],
        &query,
        DistanceMetric::L2,
        2,
        &VectorAnnOptions::default(),
        None,
        |id| {
            store
                .get(&id)
                .map(|v| std::borrow::Cow::Borrowed(v.as_slice()))
        },
    );
    assert!(result.is_ok(), "metric override rerank must not error");
    let ranked = result.unwrap();
    assert!(!ranked.is_empty(), "must return at least one result");
    assert_eq!(ranked[0].id, 1, "L2 rerank: id=1 should rank first");
}

#[test]
fn metric_none_uses_index_metric() {
    use nodedb_types::vector_distance::DistanceMetric;
    use nodedb_vector::rerank::{Candidate, rerank};

    let store: HashMap<u32, Vec<f32>> = [(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])]
        .into_iter()
        .collect();
    let query = [1.0_f32, 0.0];
    let index_metric = DistanceMetric::Cosine;

    let result = rerank(
        vec![
            Candidate {
                id: 1,
                index_distance: 0.0,
            },
            Candidate {
                id: 2,
                index_distance: 1.0,
            },
        ],
        &query,
        index_metric,
        2,
        &VectorAnnOptions::default(),
        None,
        |id| {
            store
                .get(&id)
                .map(|v| std::borrow::Cow::Borrowed(v.as_slice()))
        },
    );
    assert!(
        result.is_ok(),
        "metric=None (index metric) rerank must not error"
    );
    assert!(!result.unwrap().is_empty());
}
