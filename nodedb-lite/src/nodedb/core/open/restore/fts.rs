// SPDX-License-Identifier: Apache-2.0

//! Full-text search index restore.

use std::sync::Arc;

use nodedb_mem::MemoryGovernor;
use nodedb_types::error::NodeDbResult;

use crate::storage::engine::StorageEngine;

use crate::nodedb::core::types::NodeDbLite;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Restore FTS indices from a persistent checkpoint.
    ///
    /// Returns the manager and whether its state is complete. The state is
    /// incomplete — and the caller rebuilds schemaless documents from CRDT
    /// state, see `open_inner` — when no checkpoint exists (first open or
    /// after a collection drop), when the checkpoint cannot be read, and when
    /// the checkpoint predates per-field indexing.
    pub(in crate::nodedb::core::open) async fn restore_fts_indices(
        storage: &Arc<S>,
        governor: &Arc<MemoryGovernor>,
    ) -> NodeDbResult<(crate::engine::fts::FtsCollectionManager, bool)> {
        let mut mgr = crate::engine::fts::FtsCollectionManager::new(Arc::clone(governor));
        match crate::engine::fts::checkpoint::restore_fts(storage.as_ref(), Arc::clone(governor))
            .await
        {
            Ok(restored) if !restored.indices.is_empty() => {
                let complete = restored.per_field_layout;
                mgr.load_checkpoint(
                    restored.indices,
                    restored.id_to_surrogate,
                    restored.surrogate_to_id,
                    restored.next_surrogate,
                );
                Ok((mgr, complete))
            }
            // No checkpoint found — caller will rebuild from CRDT state.
            Ok(_) => Ok((mgr, false)),
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "FTS checkpoint restore failed — starting with empty index; \
                     will rebuild from CRDT state on cold open"
                );
                Ok((mgr, false))
            }
        }
    }
}
