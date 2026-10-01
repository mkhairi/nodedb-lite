// SPDX-License-Identifier: Apache-2.0

//! Exact candidate ranking with native segment backing and WASM inline vectors.

use nodedb_types::VectorAnnOptions;
use nodedb_types::vector_distance::DistanceMetric;
use nodedb_vector::rerank::{Candidate, CodecSidecar, Ranked, rerank};

use crate::engine::vector::graph::HnswIndex;
use crate::error::LiteError;

pub(super) fn rank_candidates(
    index: &HnswIndex,
    candidates: Vec<Candidate>,
    query: &[f32],
    metric: Option<DistanceMetric>,
    k: usize,
    opts: &VectorAnnOptions,
    sidecar: Option<&CodecSidecar>,
) -> Result<Vec<Ranked>, LiteError> {
    // On native targets, use `get_vector_or_backing`: it serves graph-checkpoint-
    // only restored indexes (empty per-node local storage) from the pagedb segment
    // backing attached by `with_backing`, and decodes F16/BF16 node storage to f32.
    // On WASM the backing path is absent; `get_vector` is correct there because
    // WASM uses the full-checkpoint blob path where F32 vectors are inline.
    #[cfg(not(target_arch = "wasm32"))]
    let ranked = rerank(
        candidates,
        query,
        metric.unwrap_or_else(|| index.metric()),
        k,
        opts,
        sidecar,
        |id| index.get_vector_or_backing(id),
    )
    .map_err(|e| LiteError::Query(e.to_string()))?;

    #[cfg(target_arch = "wasm32")]
    let ranked = rerank(
        candidates,
        query,
        metric.unwrap_or_else(|| index.metric()),
        k,
        opts,
        sidecar,
        |id| index.get_vector(id).map(std::borrow::Cow::Borrowed),
    )
    .map_err(|e| LiteError::Query(e.to_string()))?;

    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use nodedb_types::VectorAnnOptions;
    use std::collections::HashMap;

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
}
