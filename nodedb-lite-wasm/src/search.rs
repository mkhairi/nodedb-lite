// SPDX-License-Identifier: Apache-2.0

//! Fused vector, text, and graph retrieval.

use crate::{dispatch, types::NodeDbLiteWasm};
use nodedb_lite::{GraphRagParams, HybridSearchParams};
use nodedb_types::{filter::MetadataFilter, id::NodeId};
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

#[derive(serde::Deserialize)]
#[serde(default)]
struct HybridParams {
    collection: String,
    query_embedding: Vec<f32>,
    query_text: String,
    text_field: String,
    vector_k: usize,
    text_k: usize,
    top_k: usize,
    filter: Option<MetadataFilter>,
    allowed_ids: Option<HashSet<String>>,
}

impl Default for HybridParams {
    fn default() -> Self {
        let params = HybridSearchParams::default();
        Self {
            collection: String::new(),
            query_embedding: Vec::new(),
            query_text: String::new(),
            text_field: String::new(),
            vector_k: params.vector_k,
            text_k: params.text_k,
            top_k: params.top_k,
            filter: None,
            allowed_ids: None,
        }
    }
}

impl HybridParams {
    fn borrowed(&self) -> HybridSearchParams<'_> {
        HybridSearchParams {
            collection: &self.collection,
            query_embedding: &self.query_embedding,
            query_text: &self.query_text,
            text_field: &self.text_field,
            vector_k: self.vector_k,
            text_k: self.text_k,
            top_k: self.top_k,
            filter: self.filter.as_ref(),
            allowed_ids: self.allowed_ids.as_ref(),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(default)]
struct GraphParams {
    collection: String,
    query: Vec<f32>,
    vector_k: usize,
    graph_depth: u8,
    top_k: usize,
    filter: Option<MetadataFilter>,
    rrf_k: f64,
    allowed_ids: Option<HashSet<String>>,
    seed_nodes: Option<Vec<String>>,
}

impl Default for GraphParams {
    fn default() -> Self {
        let params = GraphRagParams::default();
        Self {
            collection: String::new(),
            query: Vec::new(),
            vector_k: params.vector_k,
            graph_depth: params.graph_depth,
            top_k: params.top_k,
            filter: None,
            rrf_k: params.rrf_k,
            allowed_ids: None,
            seed_nodes: None,
        }
    }
}

impl GraphParams {
    fn seeds(&self) -> Result<Option<Vec<NodeId>>, nodedb_types::error::NodeDbError> {
        self.seed_nodes
            .as_ref()
            .map(|seeds| {
                seeds
                    .iter()
                    .map(|id| {
                        NodeId::try_new(id.clone())
                            .map_err(nodedb_types::error::NodeDbError::bad_request)
                    })
                    .collect()
            })
            .transpose()
    }

    fn borrowed<'a>(&'a self, seeds: Option<&'a [NodeId]>) -> GraphRagParams<'a> {
        GraphRagParams {
            collection: &self.collection,
            query: &self.query,
            vector_k: self.vector_k,
            graph_depth: self.graph_depth,
            top_k: self.top_k,
            filter: self.filter.as_ref(),
            rrf_k: self.rrf_k,
            allowed_ids: self.allowed_ids.as_ref(),
            seed_nodes: seeds,
        }
    }
}

#[wasm_bindgen]
impl NodeDbLiteWasm {
    /// Fuse vector and text retrieval with owned parameters matching `HybridSearchParams`.
    /// Null `allowed_ids` permits every ID. An empty array permits none.
    #[wasm_bindgen(js_name = "hybridSearch")]
    pub async fn hybrid_search(&self, params: JsValue) -> Result<JsValue, JsError> {
        let params: HybridParams =
            serde_wasm_bindgen::from_value(params).map_err(|e| JsError::new(&e.to_string()))?;
        let borrowed = params.borrowed();
        let results = dispatch!(self, db, {
            db.hybrid_search(&borrowed)
                .await
                .map_err(|e| JsError::new(&e.to_string()))
        })?;
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Fuse vector and graph retrieval with owned parameters matching `GraphRagParams`.
    /// `seed_nodes` contains node ID strings. Null `allowed_ids` permits every ID.
    #[wasm_bindgen(js_name = "graphRagSearch")]
    pub async fn graph_rag_search(&self, params: JsValue) -> Result<JsValue, JsError> {
        let params: GraphParams =
            serde_wasm_bindgen::from_value(params).map_err(|e| JsError::new(&e.to_string()))?;
        let seeds = params.seeds().map_err(|e| JsError::new(&e.to_string()))?;
        let borrowed = params.borrowed(seeds.as_deref());
        let results = dispatch!(self, db, {
            db.graph_rag_search(&borrowed)
                .await
                .map_err(|e| JsError::new(&e.to_string()))
        })?;
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }
}
