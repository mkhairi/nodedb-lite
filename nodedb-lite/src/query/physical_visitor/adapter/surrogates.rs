// SPDX-License-Identifier: Apache-2.0

//! Array surrogate scans for physical prefilters.

use crate::error::LiteError;
use crate::runtime::now_millis_i64;
use crate::storage::engine::StorageEngine;
use nodedb_array::query::slice::Slice;
use std::sync::Arc;

/// Decode a msgpack-encoded `Slice` for array `name` and run a surrogate
/// bitmap scan against the array engine, returning the set of surrogates
/// for all live cells that match the slice predicate.
pub(crate) async fn execute_surrogate_scan<S: StorageEngine>(
    array_state: &Arc<tokio::sync::Mutex<crate::engine::array::engine::ArrayEngineState>>,
    storage: &Arc<S>,
    name: &str,
    slice_bytes: &[u8],
) -> Result<roaring::RoaringBitmap, LiteError> {
    let slice: Slice =
        zerompk::from_msgpack(slice_bytes).map_err(|e| LiteError::Serialization {
            detail: format!("decode Slice predicate: {e}"),
        })?;
    let system_as_of = now_millis_i64();
    let mut state = array_state.lock().await;
    state
        .surrogate_bitmap_scan(storage, name, slice.dim_ranges, system_as_of)
        .await
}
