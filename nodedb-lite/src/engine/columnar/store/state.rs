// SPDX-License-Identifier: Apache-2.0

//! Columnar engine state and collection metadata.

use crate::storage::engine::StorageEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::sync::outbound::columnar::ColumnarOutbound;
#[cfg(not(target_arch = "wasm32"))]
use crate::sync::outbound::timeseries::TimeseriesOutbound;
use nodedb_columnar::mutation::MutationEngine;
use nodedb_mem::ScopedMemory;
use nodedb_types::columnar::ColumnarProfile;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Per-collection segment metadata.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub(in crate::engine::columnar) struct SegmentMeta {
    pub(in crate::engine::columnar) segment_id: u32,
    pub(in crate::engine::columnar) row_count: u64,
    /// Milliseconds since Unix epoch when this segment was first written.
    /// Used by bitemporal purge to determine which superseded segments are
    /// eligible for deletion.
    #[serde(default)]
    pub(super) system_time_from_ms: i64,
    /// For bitemporal collections: the millisecond timestamp when the last
    /// live row in this segment was deleted (compacted away). `None` means
    /// the segment still has live rows. Segments with `Some(t)` where
    /// `t < cutoff_ms` are eligible for physical deletion by `purge_bitemporal_before`.
    #[serde(default)]
    pub(in crate::engine::columnar) fully_deleted_at_ms: Option<i64>,
}

/// Per-collection state. Wrapped in `Mutex` inside `ColumnarEngine`.
pub(in crate::engine::columnar) struct CollectionState {
    pub(in crate::engine::columnar) mutation: MutationEngine,
    pub(super) profile: ColumnarProfile,
    /// Whether this collection has bitemporal system-time tracking.
    pub(super) bitemporal: bool,
    /// Ordered list of flushed segments (including fully-deleted tombstones for
    /// bitemporal collections — they persist until `purge_bitemporal_before` clears them).
    pub(in crate::engine::columnar) segments: Vec<SegmentMeta>,
    /// Next segment ID to assign.
    pub(super) next_segment_id: u32,
}

pub(super) type CollectionMap = HashMap<String, Arc<Mutex<CollectionState>>>;

/// Manages all columnar collections for a NodeDbLite instance.
pub struct ColumnarEngine<S: StorageEngine> {
    pub(in crate::engine::columnar) storage: Arc<S>,
    pub(super) collections: RwLock<CollectionMap>,
    /// Optional outbound queue for plain columnar insert sync.
    /// `None` when sync is disabled or not yet configured.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) outbound: Option<Arc<ColumnarOutbound<S>>>,
    /// Optional outbound queue for timeseries-profile insert sync.
    ///
    /// Timeseries collections must use `TimeseriesPush` frames on Origin
    /// (the columnar `MutationEngine` and the timeseries engine are separate
    /// storage paths on Origin).  When this queue is present, inserts into
    /// collections with `ColumnarProfile::Timeseries` are enqueued here
    /// instead of `outbound`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) timeseries_outbound: Option<Arc<TimeseriesOutbound<S>>>,
    /// Governor handle bound to the columnar engine budget. Cloned into every
    /// `SegmentWriter` this engine creates.
    pub(super) memory: ScopedMemory,
}
