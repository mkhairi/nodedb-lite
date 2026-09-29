// SPDX-License-Identifier: Apache-2.0

//! Spatial index restore.

use std::sync::Arc;

use nodedb_mem::ScopedMemory;

use crate::nodedb::flush_gens::FlushGens;
use crate::storage::engine::StorageEngine;

use crate::nodedb::core::types::NodeDbLite;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Restore spatial indices from storage.
    ///
    /// The manager records its mutations in `flush_gens`. A tree restored
    /// from its stored checkpoint starts clean. A tree the cold-open rebuild
    /// adds starts dirty.
    pub(in crate::nodedb::core::open) async fn restore_spatial_indices(
        storage: &Arc<S>,
        memory: &ScopedMemory,
        flush_gens: &Arc<FlushGens>,
    ) -> crate::engine::spatial::SpatialIndexManager {
        let mut mgr = crate::engine::spatial::SpatialIndexManager::with_gens(
            memory.clone(),
            Arc::clone(flush_gens),
        );
        match crate::engine::spatial::checkpoint::restore_spatial(storage.as_ref()).await {
            Ok((checkpoints, doc_to_entry, next_id)) if !checkpoints.is_empty() => {
                mgr.load_checkpoint(&checkpoints, doc_to_entry, next_id);
            }
            Ok(_) => {}
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "spatial checkpoint restore failed — starting with empty index; \
                     will rebuild from CRDT state on cold open"
                );
            }
        }
        mgr
    }
}
