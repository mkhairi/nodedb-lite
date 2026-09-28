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
use nodedb_vector::rerank::CodecSidecar;

use crate::engine::vector::HnswIndex;
use crate::engine::vector::id_map::VectorIdMap;
use crate::storage::engine::StorageEngine;

pub struct VectorState<S: StorageEngine> {
    pub(crate) hnsw_indices: Mutex<HashMap<String, HnswIndex>>,
    /// Per-index node ↔ document id bindings. Lock order: `hnsw_indices`
    /// first, then this.
    pub(crate) vector_id_map: Mutex<VectorIdMap>,
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
    /// creates the index through `resident::lock_resident_or_create`) resolves
    /// the collection normally without anything here needing to be cleared.
    pub(crate) unloadable: Mutex<HashSet<String>>,
    /// Index keys eviction dropped from `hnsw_indices` after writing their
    /// checkpoint. An index absent from memory and named here must be loaded
    /// back before any use: creating an empty one in its place hides every
    /// vector it held. The lazy loader clears the mark once it has loaded
    /// the index or found it unloadable. Lock order: `hnsw_indices` first.
    pub(crate) evicted: Mutex<HashSet<String>>,
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
    pub id_map: VectorIdMap,
    pub memory: ScopedMemory,
}

impl<S: StorageEngine> VectorState<S> {
    pub fn new(storage: Arc<S>, search_ef: usize, memory: ScopedMemory) -> Self {
        Self {
            hnsw_indices: Mutex::new(HashMap::new()),
            vector_id_map: Mutex::new(VectorIdMap::new()),
            search_ef,
            storage,
            codec_sidecars: Arc::new(Mutex::new(HashMap::new())),
            per_index_config: Arc::new(Mutex::new(HashMap::new())),
            unloadable: Mutex::new(HashSet::new()),
            evicted: Mutex::new(HashSet::new()),
            memory,
        }
    }

    pub fn from_restored(restored: RestoredVectorState<S>) -> Self {
        Self {
            hnsw_indices: Mutex::new(restored.indices),
            vector_id_map: Mutex::new(restored.id_map),
            search_ef: restored.search_ef,
            storage: restored.storage,
            codec_sidecars: Arc::new(Mutex::new(HashMap::new())),
            per_index_config: Arc::new(Mutex::new(HashMap::new())),
            unloadable: Mutex::new(HashSet::new()),
            evicted: Mutex::new(HashSet::new()),
            memory: restored.memory,
        }
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
}
