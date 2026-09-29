// SPDX-License-Identifier: Apache-2.0

//! Shared runtime state for HNSW vector search on Lite.
//!
//! Held as `Arc<VectorState<S>>` on both `NodeDbLite<S>` (user-facing
//! entry points) and `LiteQueryEngine<S>` (PhysicalPlan executor) so
//! the visitor pipeline can run vector ops without re-architecting the
//! engine boundary.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use nodedb_mem::ScopedMemory;
use nodedb_types::collection_config::VectorPrimaryConfig;
use nodedb_types::hnsw::HnswParams;
use nodedb_types::vector_dtype::VectorStorageDtype;
use nodedb_vector::rerank::CodecSidecar;

use crate::engine::vector::HnswIndex;
use crate::engine::vector::id_map::VectorIdMap;
use crate::nodedb::flush_gens::{
    FlushArtifact, FlushGens, ID_MAP_KEY, TrackedCell, TrackedMap, TrackedMapGuard,
};
use crate::storage::engine::StorageEngine;

pub struct VectorState<S: StorageEngine> {
    /// Per-collection HNSW indices. Every mutable access marks the touched
    /// collection's [`FlushArtifact::HnswGraph`] dirty.
    pub(crate) hnsw_indices: TrackedMap<HnswIndex>,
    /// Slot ↔ document id, both directions. See [`VectorIdMap`]: the reverse
    /// direction is what lets an insert replace a document's existing vector
    /// instead of appending a second one for the same id. Every mutable
    /// access marks [`FlushArtifact::HnswIdMap`] dirty.
    pub(crate) vector_id_map: TrackedCell<VectorIdMap>,
    /// Flush dirty tracking shared with the owning `NodeDbLite`.
    pub(crate) flush_gens: Arc<FlushGens>,
    pub(crate) search_ef: usize,
    pub(crate) storage: Arc<S>,
    /// index_key → trained codec sidecar (populated by S2.a.11).
    pub(crate) codec_sidecars: Arc<Mutex<HashMap<String, CodecSidecar>>>,
    /// Per-(index_key) collection config — populated when a collection is
    /// registered via DDL (C2c will wire that). Lookup is best-effort:
    /// callers that don't find an entry default to F32 storage, matching
    /// the previous behavior.
    pub(crate) per_index_config: Arc<Mutex<HashMap<String, VectorPrimaryConfig>>>,
    /// Index keys whose stored checkpoint exists but cannot be turned into a
    /// usable index — unreadable checkpoint, or a segment that cannot serve the
    /// graph with no durable vectors to rebuild from.
    ///
    /// Without this, `ensure_index_loaded` gives up WITHOUT caching anything, so
    /// every later search on that collection repeats the entire cost — read the
    /// checkpoint, deserialize the full graph, open and validate the segment,
    /// scan the durable rows — and still finds nothing. On a collection with
    /// thousands of nodes that turns one unusable segment into an operation that
    /// pins a core indefinitely while reporting no progress.
    ///
    /// This is a NEGATIVE cache for the load path only. It is not consulted once
    /// the collection is present in `hnsw_indices`, so a later insert (which
    /// creates the index through `ensure_hnsw`) resolves the collection normally
    /// without anything here needing to be cleared.
    pub(crate) unloadable: Mutex<HashSet<String>>,
    /// Memory scope for codec sidecar and rerank allocations owned by this state.
    pub(crate) memory: ScopedMemory,
}

/// Restored state for [`VectorState::from_restored`].
///
/// Grouped into a struct because the constructor otherwise carries five
/// unrelated arguments.
pub struct RestoredVectorState<S: StorageEngine> {
    pub storage: Arc<S>,
    pub search_ef: usize,
    pub indices: HashMap<String, HnswIndex>,
    pub id_map: HashMap<String, (String, u32)>,
    pub memory: ScopedMemory,
    /// Flush dirty tracking owned by the database. The restore path has
    /// already marked clean every artifact it loaded from a valid stored form.
    pub(crate) flush_gens: Arc<FlushGens>,
}

/// Get or create the HNSW index for `index_key` with the given dimensionality and
/// storage dtype. When the index already exists the `dtype` argument is ignored —
/// dtype is fixed at index-creation time and cannot be changed in place.
pub(crate) fn ensure_hnsw<'a>(
    indices: &'a mut TrackedMapGuard<'_, HnswIndex>,
    index_key: &str,
    dim: usize,
    dtype: VectorStorageDtype,
) -> &'a mut HnswIndex {
    indices.get_or_insert_with(index_key, || {
        HnswIndex::new(
            dim,
            HnswParams {
                dtype,
                ..HnswParams::default()
            },
        )
    })
}

impl<S: StorageEngine> VectorState<S> {
    pub fn new(storage: Arc<S>, search_ef: usize, memory: ScopedMemory) -> Self {
        let flush_gens = Arc::new(FlushGens::default());
        Self {
            hnsw_indices: TrackedMap::new(
                HashMap::new(),
                Arc::clone(&flush_gens),
                FlushArtifact::HnswGraph,
            ),
            vector_id_map: TrackedCell::new(
                VectorIdMap::default(),
                Arc::clone(&flush_gens),
                FlushArtifact::HnswIdMap,
                ID_MAP_KEY,
            ),
            flush_gens,
            search_ef,
            storage,
            codec_sidecars: Arc::new(Mutex::new(HashMap::new())),
            per_index_config: Arc::new(Mutex::new(HashMap::new())),
            unloadable: Mutex::new(HashSet::new()),
            memory,
        }
    }

    pub fn from_restored(restored: RestoredVectorState<S>) -> Self {
        Self {
            hnsw_indices: TrackedMap::new(
                restored.indices,
                Arc::clone(&restored.flush_gens),
                FlushArtifact::HnswGraph,
            ),
            vector_id_map: TrackedCell::new(
                VectorIdMap::from_slots(restored.id_map),
                Arc::clone(&restored.flush_gens),
                FlushArtifact::HnswIdMap,
                ID_MAP_KEY,
            ),
            flush_gens: restored.flush_gens,
            search_ef: restored.search_ef,
            storage: restored.storage,
            codec_sidecars: Arc::new(Mutex::new(HashMap::new())),
            per_index_config: Arc::new(Mutex::new(HashMap::new())),
            unloadable: Mutex::new(HashSet::new()),
            memory: restored.memory,
        }
    }

    /// Record that durable vector rows under `index_key` were written or
    /// removed, so the vector segments built from them are dirty.
    ///
    /// Call it AFTER the storage write returns, whatever its result. Marking
    /// first would let a flush capture the new generation, read the rows
    /// before the write lands, and record a segment without that row as
    /// current.
    pub(crate) fn mark_vector_rows_changed(&self, index_key: &str) {
        self.flush_gens.bump_vector_rows(index_key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::engine::{test_governor, test_scoped_memory};
    use crate::storage::pagedb_storage::PagedbStorageMem;

    #[tokio::test]
    async fn per_index_config_starts_empty() {
        let storage = Arc::new(
            PagedbStorageMem::open_in_memory()
                .await
                .expect("in-memory pagedb"),
        );
        let memory = test_scoped_memory(&test_governor(), nodedb_mem::EngineId::Vector);
        let state = VectorState::new(storage, 100, memory);
        let configs = state.per_index_config.lock().expect("lock");
        assert!(
            configs.is_empty(),
            "per_index_config must be empty on construction"
        );
    }

    fn tracked_indices() -> TrackedMap<HnswIndex> {
        TrackedMap::new(
            HashMap::new(),
            Arc::new(FlushGens::default()),
            FlushArtifact::HnswGraph,
        )
    }

    #[test]
    fn ensure_hnsw_creates_index_with_f32_default() {
        let tracked = tracked_indices();
        let mut indices = tracked.lock_or_recover();
        ensure_hnsw(&mut indices, "col", 4, VectorStorageDtype::F32);
        let idx = indices.get("col").expect("index created");
        assert_eq!(idx.params().dtype, VectorStorageDtype::F32);
    }

    #[test]
    fn ensure_hnsw_creates_index_with_bf16() {
        let tracked = tracked_indices();
        let mut indices = tracked.lock_or_recover();
        ensure_hnsw(&mut indices, "col", 4, VectorStorageDtype::BF16);
        let idx = indices.get("col").expect("index created");
        assert_eq!(idx.params().dtype, VectorStorageDtype::BF16);
    }

    #[test]
    fn ensure_hnsw_existing_index_ignores_dtype_arg() {
        let tracked = tracked_indices();
        let mut indices = tracked.lock_or_recover();
        ensure_hnsw(&mut indices, "col", 4, VectorStorageDtype::F32);
        // Call again with BF16 — dtype is fixed at creation time, must not change.
        ensure_hnsw(&mut indices, "col", 4, VectorStorageDtype::BF16);
        let idx = indices.get("col").expect("index present");
        assert_eq!(
            idx.params().dtype,
            VectorStorageDtype::F32,
            "dtype must remain F32; dtype is fixed at index-creation time"
        );
    }
}
