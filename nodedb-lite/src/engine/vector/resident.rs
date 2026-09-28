// SPDX-License-Identifier: Apache-2.0

//! Resident access to HNSW indexes that eviction can drop from memory.
//!
//! Eviction writes an index's checkpoint to storage and removes it from
//! `hnsw_indices`. Every insert, search and delete reaches its index through
//! this module, which loads a stored index back before handing it out:
//!
//! - [`lock_resident`] locks the index map with the index loaded when storage
//!   holds one.
//! - [`lock_resident_or_create`] does the same, then creates an empty index
//!   only when neither memory nor storage holds one. Creating one over an
//!   evicted index would hide every vector it held.
//!
//! A load awaits storage, so eviction can run between the load and the lock.
//! The eviction mark (`VectorState::evicted`) catches that: an index absent
//! from memory and still marked is loaded again before the lock is returned.

use std::collections::HashMap;
use std::sync::{Arc, MutexGuard};

use nodedb_types::hnsw::HnswParams;
use nodedb_types::vector_dtype::VectorStorageDtype;

use crate::engine::vector::HnswIndex;
use crate::engine::vector::VectorState;
use crate::engine::vector::search::lazy_load::ensure_index_loaded;
use crate::error::LiteError;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

/// The locked index map.
pub(crate) type IndexMap<'a> = MutexGuard<'a, HashMap<String, HnswIndex>>;

/// Lock the index map with `index_key`'s index resident when memory or
/// storage holds one. Fails when reading the stored index fails.
pub(crate) async fn lock_resident<'a, S: StorageEngine>(
    state: &'a Arc<VectorState<S>>,
    index_key: &str,
) -> Result<IndexMap<'a>, LiteError> {
    loop {
        ensure_index_loaded(state, index_key).await?;
        let indices = state.hnsw_indices.lock_or_recover();
        if indices.contains_key(index_key) || !state.evicted.lock_or_recover().contains(index_key) {
            return Ok(indices);
        }
    }
}

/// The locked index map holding `key`'s index, loaded or created.
pub(crate) struct ResidentIndex<'a> {
    indices: IndexMap<'a>,
    key: String,
    dim: usize,
    dtype: VectorStorageDtype,
}

impl ResidentIndex<'_> {
    /// The index. It exists: it was loaded, or is created on first use.
    pub(crate) fn index(&mut self) -> &mut HnswIndex {
        get_or_create(&mut self.indices, &self.key, self.dim, self.dtype)
    }
}

/// Lock the index map with `index_key`'s index loaded, or ready to be
/// created at `dim` with the storage dtype configured for `index_key`
/// (F32 when none is). An existing index keeps its own dimension and dtype.
/// Fails as [`lock_resident`] does.
pub(crate) async fn lock_resident_or_create<'a, S: StorageEngine>(
    state: &'a Arc<VectorState<S>>,
    index_key: &str,
    dim: usize,
) -> Result<ResidentIndex<'a>, LiteError> {
    let dtype = state
        .per_index_config
        .lock_or_recover()
        .get(index_key)
        .map(|cfg| cfg.storage_dtype)
        .unwrap_or(VectorStorageDtype::F32);
    let indices = lock_resident(state, index_key).await?;
    Ok(ResidentIndex {
        indices,
        key: index_key.to_string(),
        dim,
        dtype,
    })
}

/// Refuse vectors whose width differs from `index_key`'s index, loading a
/// stored index first, or from each other when there is no index.
///
/// Runs before an insert writes its durable rows, so a refused vector never
/// reaches storage.
pub(crate) async fn check_insert_widths<S: StorageEngine>(
    state: &Arc<VectorState<S>>,
    index_key: &str,
    widths: impl IntoIterator<Item = usize>,
) -> Result<(), LiteError> {
    let mut expected = lock_resident(state, index_key)
        .await?
        .get(index_key)
        .map(HnswIndex::dim);
    for got in widths {
        match expected {
            Some(dim) => nodedb_vector::error::check_dim(dim, got)?,
            None => expected = Some(got),
        }
    }
    Ok(())
}

/// Get or create the index for `index_key`. An existing index ignores `dim`
/// and `dtype`: both are fixed when the index is created.
fn get_or_create<'a>(
    indices: &'a mut HashMap<String, HnswIndex>,
    index_key: &str,
    dim: usize,
    dtype: VectorStorageDtype,
) -> &'a mut HnswIndex {
    indices.entry(index_key.to_string()).or_insert_with(|| {
        HnswIndex::new(
            dim,
            HnswParams {
                dtype,
                ..HnswParams::default()
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use nodedb_types::Namespace;

    use super::*;
    use crate::query::engine::{test_governor, test_scoped_memory};
    use crate::storage::pagedb_storage::PagedbStorageMem;

    async fn state() -> Arc<VectorState<PagedbStorageMem>> {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.unwrap());
        let memory = test_scoped_memory(&test_governor(), nodedb_mem::EngineId::Vector);
        Arc::new(VectorState::new(storage, 64, memory))
    }

    #[test]
    fn get_or_create_uses_the_dtype_only_at_creation() {
        let mut indices: HashMap<String, HnswIndex> = HashMap::new();
        get_or_create(&mut indices, "col", 4, VectorStorageDtype::BF16);
        assert_eq!(
            indices.get("col").map(|i| i.params().dtype),
            Some(VectorStorageDtype::BF16)
        );
        get_or_create(&mut indices, "col", 4, VectorStorageDtype::F32);
        assert_eq!(
            indices.get("col").map(|i| i.params().dtype),
            Some(VectorStorageDtype::BF16),
            "dtype is fixed at index-creation time"
        );
    }

    /// An index evicted to storage comes back on the next insert instead of
    /// being replaced by an empty one.
    #[tokio::test]
    async fn an_evicted_index_is_loaded_not_recreated() {
        let state = state().await;
        // The durable row every insert writes first: a graph-only checkpoint
        // is rebuilt from it.
        state
            .storage
            .batch_write(&[crate::engine::vector::durable::put_op(
                "docs",
                "a",
                &[1.0, 0.0, 0.0],
            )])
            .await
            .unwrap();
        {
            let mut resident = lock_resident_or_create(&state, "docs", 3).await.unwrap();
            resident.index().insert(vec![1.0, 0.0, 0.0]).unwrap();
        }
        // Evict by hand: checkpoint, mark, drop.
        let blob = state
            .hnsw_indices
            .lock_or_recover()
            .get("docs")
            .unwrap()
            .checkpoint_to_bytes()
            .unwrap();
        state
            .storage
            .put(
                Namespace::Vector,
                b"hnsw:docs",
                &crate::storage::checksum::wrap(&blob),
            )
            .await
            .unwrap();
        {
            let mut indices = state.hnsw_indices.lock_or_recover();
            state.evicted.lock_or_recover().insert("docs".into());
            indices.remove("docs");
        }

        let mut resident = lock_resident_or_create(&state, "docs", 3).await.unwrap();
        assert_eq!(resident.index().len(), 1, "the evicted vector is back");
        assert!(!state.evicted.lock_or_recover().contains("docs"));
    }

    /// With no index in memory or storage, an insert creates one.
    #[tokio::test]
    async fn a_new_collection_gets_an_empty_index() {
        let state = state().await;
        let mut resident = lock_resident_or_create(&state, "fresh", 2).await.unwrap();
        assert_eq!(resident.index().dim(), 2);
        assert_eq!(resident.index().len(), 0);
    }
}
