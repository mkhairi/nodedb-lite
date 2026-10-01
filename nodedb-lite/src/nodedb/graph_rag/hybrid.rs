// SPDX-License-Identifier: Apache-2.0

use super::{HybridSearchParams, fusion::search_results_to_ranked};
use crate::{nodedb::NodeDbLite, storage::engine::StorageEngine};
use nodedb_client::NodeDb;
use nodedb_types::{error::NodeDbResult, result::SearchResult};
use std::collections::{HashMap, HashSet};

impl<S: StorageEngine> NodeDbLite<S> {
    /// Hybrid search: vector similarity + BM25 text relevance fused via RRF.
    ///
    /// 1. Vector search returns `vector_k` candidates by embedding similarity.
    /// 2. Text search returns `text_k` candidates by BM25 relevance.
    /// 3. RRF fuses both rankings into a single score per document.
    /// 4. Top `top_k` results returned.
    ///
    /// Documents found by both searches get boosted (RRF scores are additive).
    pub async fn hybrid_search(
        &self,
        params: &HybridSearchParams<'_>,
    ) -> NodeDbResult<Vec<SearchResult>> {
        if params.top_k == 0 || params.allowed_ids.is_some_and(HashSet::is_empty) {
            return Ok(Vec::new());
        }
        let vector_results = if params.vector_k == 0 {
            Vec::new()
        } else {
            self.vector_search(
                params.collection,
                params.query_embedding,
                params.vector_k,
                params.filter,
                params.allowed_ids,
            )
            .await?
        };

        let text_results = if params.text_k == 0 {
            Vec::new()
        } else {
            self.text_search(
                params.collection,
                params.text_field,
                params.query_text,
                params.text_k,
                nodedb_types::TextSearchParams::default(),
                params.allowed_ids,
            )
            .await?
        };

        // RRF fusion via shared module.
        let vector_ranked = search_results_to_ranked(&vector_results, "vector");
        let text_ranked = search_results_to_ranked(&text_results, "text");
        let fused = nodedb_query::fusion::reciprocal_rank_fusion(
            &[vector_ranked, text_ranked],
            None,
            params.top_k,
        );

        // Build metadata cache for result materialization.
        let mut metadata_cache: HashMap<&str, &HashMap<String, nodedb_types::Value>> =
            HashMap::new();
        for results in [&vector_results, &text_results] {
            for result in results.iter() {
                metadata_cache
                    .entry(result.id.as_str())
                    .or_insert(&result.metadata);
            }
        }

        Ok(fused
            .into_iter()
            .map(|f| {
                let metadata = metadata_cache
                    .get(f.document_id.as_str())
                    .copied()
                    .cloned()
                    .unwrap_or_default();
                SearchResult {
                    id: f.document_id,
                    node_id: None,
                    distance: 1.0 / (1.0 + f.rrf_score as f32),
                    metadata,
                }
            })
            .collect())
    }
}
