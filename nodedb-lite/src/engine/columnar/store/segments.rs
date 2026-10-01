// SPDX-License-Identifier: Apache-2.0

//! Segment storage access and columnar error conversion.

use crate::error::LiteError;
use crate::storage::engine::StorageEngine;
use nodedb_types::Namespace;

/// Helper: write large segment bytes via the segment ext if available, or fall
/// back to the KV blob path.
pub(super) async fn store_segment_bytes<S: StorageEngine>(
    storage: &S,
    collection: &str,
    segment_id: u32,
    bytes: &[u8],
) -> Result<(), LiteError> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(ext) = storage.as_columnar_segment_ext() {
        return ext
            .write_columnar_segment(collection, segment_id, bytes)
            .await;
    }
    let seg_key = format!("{collection}:seg:{segment_id}");
    storage
        .put(Namespace::Columnar, seg_key.as_bytes(), bytes)
        .await
}

/// Helper: read large segment bytes via the segment ext if available, or fall
/// back to the KV blob path.
pub(super) async fn load_segment_bytes<S: StorageEngine>(
    storage: &S,
    collection: &str,
    segment_id: u32,
) -> Result<Option<Vec<u8>>, LiteError> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(ext) = storage.as_columnar_segment_ext() {
        return ext
            .open_columnar_segment(collection, segment_id)
            .await
            .map(|opt| opt.map(|b| b.into_vec()));
    }
    let seg_key = format!("{collection}:seg:{segment_id}");
    storage.get(Namespace::Columnar, seg_key.as_bytes()).await
}

/// Helper: delete large segment bytes via the segment ext if available, or
/// fall back to the KV blob path.
pub(in crate::engine::columnar) async fn remove_segment_bytes<S: StorageEngine>(
    storage: &S,
    collection: &str,
    segment_id: u32,
) -> Result<(), LiteError> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(ext) = storage.as_columnar_segment_ext() {
        return ext.delete_columnar_segment(collection, segment_id).await;
    }
    let seg_key = format!("{collection}:seg:{segment_id}");
    storage
        .delete(Namespace::Columnar, seg_key.as_bytes())
        .await
}

pub(super) fn columnar_err_to_lite(e: nodedb_columnar::ColumnarError) -> LiteError {
    LiteError::BadRequest {
        detail: e.to_string(),
    }
}
