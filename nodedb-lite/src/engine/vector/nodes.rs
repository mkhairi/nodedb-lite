// SPDX-License-Identifier: Apache-2.0

//! Binding a document's vector to its HNSW node, and releasing it.
//!
//! A document id holds at most one live node per index. Every insert path
//! binds through [`bind_node`], which tombstones the node the id held before,
//! so re-inserting an id replaces its vector instead of adding a second node.
//! Every delete path releases through [`unbind_node`].

use std::sync::Arc;

use crate::engine::vector::resident::lock_resident_or_create;
use crate::engine::vector::{HnswIndex, VectorState};
use crate::error::LiteError;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

/// Insert `embedding` into `index` as the node of `doc_id` in `index_key`,
/// under the caller's lock on the index map. The node `doc_id` held before
/// is tombstoned and its codec sidecar code dropped. Returns the new node id.
///
/// Fails when the index refuses the vector; the old node then stays live.
pub(crate) fn bind_node<S: StorageEngine>(
    state: &VectorState<S>,
    index: &mut HnswIndex,
    index_key: &str,
    doc_id: &str,
    embedding: Vec<f32>,
) -> Result<u32, LiteError> {
    let node = index.len() as u32;
    index.insert(embedding)?;
    let displaced = state
        .vector_id_map
        .lock_or_recover()
        .bind(index_key, doc_id, node);
    if let Some(old) = displaced {
        index.delete(old);
        drop_sidecar_code(state, index_key, old);
    }
    Ok(node)
}

/// Release the node of `doc_id` in `index_key`: unbind it, tombstone it in
/// `index` when the index is resident, and drop its codec sidecar code.
/// Returns the released node, or `None` when `doc_id` held none.
pub(crate) fn unbind_node<S: StorageEngine>(
    state: &VectorState<S>,
    index: Option<&mut HnswIndex>,
    index_key: &str,
    doc_id: &str,
) -> Option<u32> {
    let node = state
        .vector_id_map
        .lock_or_recover()
        .unbind_doc(index_key, doc_id)?;
    if let Some(index) = index {
        index.delete(node);
    }
    drop_sidecar_code(state, index_key, node);
    Some(node)
}

/// Load or create `index_key`'s index and bind `embedding` as the node of
/// `doc_id`, as [`bind_node`] does. Returns the new node id.
///
/// Fails when loading the index fails or the index refuses the vector.
pub(crate) async fn upsert_node<S: StorageEngine>(
    state: &Arc<VectorState<S>>,
    index_key: &str,
    doc_id: &str,
    embedding: &[f32],
) -> Result<u32, LiteError> {
    let mut resident = lock_resident_or_create(state, index_key, embedding.len()).await?;
    bind_node(
        state,
        resident.index(),
        index_key,
        doc_id,
        embedding.to_vec(),
    )
}

/// Encode `embedding` into `index_key`'s codec sidecar as `node`, installing
/// the sidecar first when the index config calls for one.
///
/// Fails only when installing the sidecar fails. An encode error is logged
/// and the row falls back to FP32 rerank. Call it without the index map
/// locked: installing a sidecar trains on the index.
pub(crate) fn encode_sidecar<S: StorageEngine>(
    state: &VectorState<S>,
    index_key: &str,
    node: u32,
    embedding: &[f32],
) -> Result<(), LiteError> {
    if !crate::engine::vector::sidecar::ensure_sidecar(state, index_key)? {
        return Ok(());
    }
    let mut sidecars = state.codec_sidecars.lock_or_recover();
    if let Some(sidecar) = sidecars.get_mut(index_key)
        && let Err(e) = sidecar.encode_and_insert(node, embedding)
    {
        tracing::warn!(
            index_key,
            id = node,
            error = %e,
            "sidecar encode_and_insert failed; row falls back to FP32 rerank"
        );
    }
    Ok(())
}

/// Drop `node`'s code from `index_key`'s codec sidecar, when one is installed.
fn drop_sidecar_code<S: StorageEngine>(state: &VectorState<S>, index_key: &str, node: u32) {
    if let Some(sidecar) = state.codec_sidecars.lock_or_recover().get_mut(index_key) {
        sidecar.remove(node);
    }
}
