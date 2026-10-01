// SPDX-License-Identifier: Apache-2.0

use nodedb_types::{filter::MetadataFilter, id::NodeId};
use std::collections::HashSet;

/// GraphRAG fusion parameters.
pub struct GraphRagParams<'a> {
    /// Collection to search vectors in.
    pub collection: &'a str,
    /// Query embedding.
    pub query: &'a [f32],
    /// Number of initial vector candidates.
    pub vector_k: usize,
    /// Graph expansion depth from caller and vector seeds.
    pub graph_depth: u8,
    /// Final number of results to return after fusion.
    pub top_k: usize,
    /// Optional metadata filter for vector search.
    pub filter: Option<&'a MetadataFilter>,
    /// IDs permitted in every retrieval source.
    pub allowed_ids: Option<&'a HashSet<String>>,
    /// Finite nonnegative rank smoothing constant. Default: 60. Lower values emphasize top ranks.
    pub rrf_k: f64,
    /// Caller seeds added to vector-derived seeds.
    pub seed_nodes: Option<&'a [NodeId]>,
}

impl Default for GraphRagParams<'_> {
    fn default() -> Self {
        Self {
            collection: "",
            query: &[],
            vector_k: 10,
            graph_depth: 2,
            top_k: 10,
            filter: None,
            allowed_ids: None,
            seed_nodes: None,
            rrf_k: 60.0,
        }
    }
}

/// Hybrid search parameters.
pub struct HybridSearchParams<'a> {
    /// Collection to search.
    pub collection: &'a str,
    /// Query embedding for vector similarity.
    pub query_embedding: &'a [f32],
    /// Query text for BM25 relevance.
    pub query_text: &'a str,
    /// Field the text query is scoped to. Empty searches every string field.
    pub text_field: &'a str,
    /// Number of vector candidates.
    pub vector_k: usize,
    /// Number of text candidates.
    pub text_k: usize,
    /// Final number of results to return after fusion.
    pub top_k: usize,
    /// Optional metadata filter for vector search.
    pub filter: Option<&'a MetadataFilter>,
    /// IDs permitted in every retrieval source.
    pub allowed_ids: Option<&'a HashSet<String>>,
}

impl Default for HybridSearchParams<'_> {
    fn default() -> Self {
        Self {
            collection: "",
            query_embedding: &[],
            query_text: "",
            text_field: "",
            vector_k: 10,
            text_k: 10,
            top_k: 10,
            filter: None,
            allowed_ids: None,
        }
    }
}
