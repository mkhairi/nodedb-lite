// SPDX-License-Identifier: Apache-2.0

//! Full-text search index restore.

use std::sync::Arc;

use nodedb_mem::MemoryGovernor;
use nodedb_types::error::NodeDbResult;

use crate::nodedb::flush_gens::FlushGens;
use crate::storage::engine::StorageEngine;

use crate::nodedb::core::types::NodeDbLite;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Restore FTS indices from a persistent checkpoint.
    ///
    /// Returns an empty `FtsCollectionManager` when no checkpoint exists (first
    /// open or after a collection drop).  The caller decides whether to fall
    /// back to `rebuild_text_indices` — see `open_inner`.
    ///
    /// The manager records its mutations in `flush_gens`. An index whose
    /// postings, doc lengths, and meta blobs all decoded starts clean, and so
    /// does a surrogate map that decoded. Everything else starts dirty,
    /// including every index the rebuild adds.
    pub(in crate::nodedb::core::open) async fn restore_fts_indices(
        storage: &Arc<S>,
        governor: &Arc<MemoryGovernor>,
        flush_gens: &Arc<FlushGens>,
    ) -> NodeDbResult<crate::engine::fts::FtsCollectionManager> {
        let mut mgr = crate::engine::fts::FtsCollectionManager::with_gens(
            Arc::clone(governor),
            Arc::clone(flush_gens),
        );
        match crate::engine::fts::checkpoint::restore_fts(storage.as_ref(), Arc::clone(governor))
            .await
        {
            Ok(restored) if !restored.indices.is_empty() => {
                mgr.load_checkpoint(restored);
            }
            Ok(_) => {
                // No checkpoint found — caller will rebuild from CRDT state.
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "FTS checkpoint restore failed — starting with empty index; \
                     will rebuild from CRDT state on cold open"
                );
            }
        }
        Ok(mgr)
    }
}
