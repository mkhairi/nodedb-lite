// SPDX-License-Identifier: Apache-2.0

//! Sparse-vector inverted index restore.

use std::sync::Arc;

use crate::nodedb::flush_gens::FlushGens;
use crate::storage::engine::StorageEngine;

use crate::nodedb::core::types::NodeDbLite;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Restore sparse-vector inverted indices from a persistent checkpoint.
    ///
    /// Returns the restored manager plus whether a checkpoint was found. The
    /// caller uses the flag to decide whether a rebuild from source documents
    /// is needed — an empty manager from a real checkpoint means "no sparse
    /// columns", which needs no rebuild. A restore failure is logged and
    /// reported as "no checkpoint" so the rebuild path repopulates the index
    /// rather than leaving searches silently empty.
    ///
    /// The manager records its mutations in `flush_gens`. An index whose
    /// stored blob decoded starts clean. Every other index starts dirty.
    pub(in crate::nodedb::core::open) async fn restore_sparse_indices(
        storage: &Arc<S>,
        flush_gens: &Arc<FlushGens>,
    ) -> (crate::engine::sparse_vector::SparseVectorManager, bool) {
        let mut mgr =
            crate::engine::sparse_vector::SparseVectorManager::with_gens(Arc::clone(flush_gens));
        match crate::engine::sparse_vector::checkpoint::restore_sparse(storage.as_ref()).await {
            Ok(Some(restored)) => {
                mgr.load_checkpoint(restored.indices, &restored.decoded);
                (mgr, true)
            }
            Ok(None) => (mgr, false),
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "sparse vector checkpoint restore failed — rebuilding from source documents"
                );
                (mgr, false)
            }
        }
    }
}
