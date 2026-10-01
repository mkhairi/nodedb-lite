// SPDX-License-Identifier: Apache-2.0

//! Fused retrieval through owned JSON parameters.

use crate::{
    NODEDB_ERR_FAILED, NODEDB_ERR_NULL, NODEDB_ERR_UTF8, NodeDbHandle, error::record_error,
    ffi_guard, handle_ref, ptr_to_str, write_c_string,
};
use nodedb_lite::{GraphRagParams, HybridSearchParams};
use nodedb_types::{filter::MetadataFilter, id::NodeId};
use std::collections::HashSet;
use std::os::raw::c_char;

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

/// Fuse vector and text retrieval using JSON matching `HybridSearchParams`.
/// `allowed_ids: null` permits every ID. `allowed_ids: []` permits none.
/// `*out_json` changes only on success. Free it with `nodedb_free_string`.
///
/// # Safety
/// `handle` must reference a live handle. `params_json` must reference a terminated UTF-8 string.
/// `out_json` must reference writable pointer storage throughout this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_hybrid_search(
    handle: *mut NodeDbHandle,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    ffi_guard(NODEDB_ERR_FAILED, || unsafe {
        run_search(handle, params_json, out_json, false)
    })
}

/// Fuse vector and graph retrieval using JSON matching `GraphRagParams`.
/// `seed_nodes` contains validated node ID strings. Null allows no caller seeds.
/// `allowed_ids: null` permits every ID. `allowed_ids: []` permits none.
/// `*out_json` changes only on success. Free it with `nodedb_free_string`.
///
/// # Safety
/// `handle` must reference a live handle. `params_json` must reference a terminated UTF-8 string.
/// `out_json` must reference writable pointer storage throughout this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_graph_rag_search(
    handle: *mut NodeDbHandle,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    ffi_guard(NODEDB_ERR_FAILED, || unsafe {
        run_search(handle, params_json, out_json, true)
    })
}

unsafe fn run_search(
    handle: *mut NodeDbHandle,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
    graph: bool,
) -> i32 {
    let Some(h) = handle_ref(handle) else {
        return NODEDB_ERR_NULL;
    };
    if params_json.is_null() || out_json.is_null() {
        return NODEDB_ERR_NULL;
    }
    let Some(json) = ptr_to_str(params_json) else {
        return NODEDB_ERR_UTF8;
    };
    let result = (|| -> nodedb_types::error::NodeDbResult<String> {
        let results = if graph {
            let params: GraphParams =
                sonic_rs::from_str(json).map_err(nodedb_types::error::NodeDbError::bad_request)?;
            let seeds = params.seeds()?;
            h.rt.block_on(h.db.graph_rag_search(&params.borrowed(seeds.as_deref())))?
        } else {
            let params: HybridParams =
                sonic_rs::from_str(json).map_err(nodedb_types::error::NodeDbError::bad_request)?;
            h.rt.block_on(h.db.hybrid_search(&params.borrowed()))?
        };
        sonic_rs::to_string(&results).map_err(nodedb_types::error::NodeDbError::storage)
    })();
    match result {
        Ok(json) => unsafe { write_c_string(out_json, json) },
        Err(error) => {
            record_error(error);
            NODEDB_ERR_FAILED
        }
    }
}
