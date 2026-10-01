// SPDX-License-Identifier: Apache-2.0

//! `StorageEngine` implementation for native targets.

use async_trait::async_trait;
use pagedb::vfs::Vfs;

use nodedb_types::Namespace;

use crate::error::LiteError;
use crate::storage::engine::{CompactionOutcome, KvPair, PrefixScan, StorageEngine, WriteOp};
use crate::storage::pagedb_storage::types::PagedbStorage;

#[async_trait]
impl<V: Vfs + Clone + Send + Sync + 'static> StorageEngine for PagedbStorage<V>
where
    <V as Vfs>::LockHandle: Sync,
    <V as Vfs>::File: Sync,
{
    async fn get(&self, ns: Namespace, key: &[u8]) -> Result<Option<Vec<u8>>, LiteError> {
        self.get_rows(ns, key).await
    }

    async fn put(&self, ns: Namespace, key: &[u8], value: &[u8]) -> Result<(), LiteError> {
        self.put_rows(ns, key, value).await
    }

    async fn delete(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
        self.delete_rows(ns, key).await
    }

    async fn scan_prefix(&self, ns: Namespace, prefix: &[u8]) -> Result<Vec<KvPair>, LiteError> {
        self.scan_prefix_rows(ns, prefix).await
    }

    async fn scan_prefix_bounded(
        &self,
        ns: Namespace,
        prefix: &[u8],
        limit: usize,
    ) -> Result<Vec<KvPair>, LiteError> {
        self.scan_prefix_bounded_rows(ns, prefix, limit).await
    }

    async fn scan_prefix_budgeted(
        &self,
        ns: Namespace,
        prefix: &[u8],
        max_records: usize,
        max_bytes: usize,
    ) -> Result<PrefixScan, LiteError> {
        self.scan_budgeted(ns, prefix, None, max_records, max_bytes, false)
            .await
    }

    async fn scan_prefix_from_budgeted(
        &self,
        ns: Namespace,
        prefix: &[u8],
        after_key: Option<&[u8]>,
        max_records: usize,
        max_bytes: usize,
    ) -> Result<PrefixScan, LiteError> {
        self.scan_budgeted(ns, prefix, after_key, max_records, max_bytes, true)
            .await
    }

    async fn batch_write(&self, ops: &[WriteOp]) -> Result<(), LiteError> {
        self.batch_write_rows(ops).await
    }

    async fn count(&self, ns: Namespace) -> Result<u64, LiteError> {
        self.count_rows(ns).await
    }

    async fn scan_range(
        &self,
        ns: Namespace,
        start: &[u8],
        limit: usize,
    ) -> Result<Vec<KvPair>, LiteError> {
        self.scan_range_rows(ns, start, limit).await
    }

    async fn scan_range_bounded(
        &self,
        ns: Namespace,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<KvPair>, LiteError> {
        self.scan_range_bounded_rows(ns, start, end, limit).await
    }

    async fn compact(&self) -> Result<CompactionOutcome, LiteError> {
        self.compact_rows().await
    }

    fn as_vector_segment_ext(
        &self,
    ) -> Option<&dyn crate::storage::vector_segment_ext::VectorSegmentExt> {
        Some(self)
    }

    fn as_array_segment_ext(
        &self,
    ) -> Option<&dyn crate::storage::array_segment_ext::ArraySegmentExt> {
        Some(self)
    }

    fn as_fts_segment_ext(&self) -> Option<&dyn crate::storage::fts_segment_ext::FtsSegmentExt> {
        Some(self)
    }

    fn as_columnar_segment_ext(
        &self,
    ) -> Option<&dyn crate::storage::columnar_segment_ext::ColumnarSegmentExt> {
        Some(self)
    }

    fn as_graph_segment_ext(
        &self,
    ) -> Option<&dyn crate::storage::graph_segment_ext::GraphSegmentExt> {
        Some(self)
    }

    fn as_spatial_segment_ext(
        &self,
    ) -> Option<&dyn crate::storage::spatial_segment_ext::SpatialSegmentExt> {
        Some(self)
    }
}
