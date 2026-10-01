// SPDX-License-Identifier: Apache-2.0

//! Namespaced prefix and range scans with continuation budgets.

use nodedb_types::Namespace;
use pagedb::vfs::Vfs;

use crate::error::LiteError;
use crate::storage::engine::{
    KvPair, PrefixScan, PrefixScanLimit, check_prefix_cursor, prefix_scan_budget_error,
};
use crate::storage::pagedb_storage::keys::{ns_end, prefix_key, strip_prefix};
use crate::storage::pagedb_storage::types::PagedbStorage;

impl<V: Vfs + Clone> PagedbStorage<V> {
    pub(super) async fn scan_prefix_rows(
        &self,
        ns: Namespace,
        prefix: &[u8],
    ) -> Result<Vec<KvPair>, LiteError> {
        let ns_prefix = prefix_key(ns, prefix);
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        let raw = txn.scan_prefix(&ns_prefix).await.map_err(LiteError::from)?;
        Ok(raw
            .into_iter()
            .map(|(k, v)| (strip_prefix(&k).to_vec(), v.to_vec()))
            .collect())
    }

    pub(super) async fn scan_prefix_bounded_rows(
        &self,
        ns: Namespace,
        prefix: &[u8],
        limit: usize,
    ) -> Result<Vec<KvPair>, LiteError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let ns_prefix = prefix_key(ns, prefix);
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        let raw = txn
            .scan_prefix_from(&ns_prefix, &ns_prefix, limit)
            .await
            .map_err(LiteError::from)?;
        Ok(raw
            .into_iter()
            .map(|(k, v)| (strip_prefix(&k).to_vec(), v.to_vec()))
            .collect())
    }

    pub(super) async fn scan_range_rows(
        &self,
        ns: Namespace,
        start: &[u8],
        limit: usize,
    ) -> Result<Vec<KvPair>, LiteError> {
        let start_key = prefix_key(ns, start);
        let end_key = ns_end(ns);
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        let raw = txn
            .scan(&start_key, &end_key)
            .await
            .map_err(LiteError::from)?;
        Ok(raw
            .into_iter()
            .take(limit)
            .map(|(k, v)| (strip_prefix(&k).to_vec(), v.to_vec()))
            .collect())
    }

    pub(super) async fn scan_range_bounded_rows(
        &self,
        ns: Namespace,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<KvPair>, LiteError> {
        let start_key = match start {
            Some(s) => prefix_key(ns, s),
            None => vec![ns as u8],
        };
        let end_key = match end {
            Some(e) => prefix_key(ns, e),
            None => ns_end(ns),
        };
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        let raw = txn
            .scan(&start_key, &end_key)
            .await
            .map_err(LiteError::from)?;
        let effective_limit = limit.unwrap_or(usize::MAX);
        Ok(raw
            .into_iter()
            .take(effective_limit)
            .map(|(k, v)| (strip_prefix(&k).to_vec(), v.to_vec()))
            .collect())
    }

    pub(super) async fn scan_budgeted(
        &self,
        ns: Namespace,
        prefix: &[u8],
        after_key: Option<&[u8]>,
        max_records: usize,
        max_bytes: usize,
        reject_oversized: bool,
    ) -> Result<PrefixScan, LiteError> {
        if max_records == 0 {
            return Ok(PrefixScan::default());
        }
        check_prefix_cursor(prefix, after_key)?;
        let ns_prefix = prefix_key(ns, prefix);
        let mut start = prefix_key(ns, after_key.unwrap_or(prefix));
        if after_key.is_some() {
            start.push(0);
        }
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        // Physical keys add one namespace byte per retained record.
        let raw = txn
            .scan_prefix_from_bounded(
                &ns_prefix,
                &start,
                max_records,
                max_bytes.saturating_add(max_records),
            )
            .await
            .map_err(LiteError::from)?;
        let mut result = PrefixScan {
            entries: Vec::with_capacity(raw.entries.len()),
            limit: raw.limit.map(|limit| match limit {
                pagedb::ScanLimit::Records => PrefixScanLimit::Records,
                pagedb::ScanLimit::Bytes => PrefixScanLimit::Bytes,
            }),
        };
        let mut bytes = 0usize;
        for (key, value) in raw.entries {
            let key = strip_prefix(&key);
            let next_bytes = key
                .len()
                .checked_add(value.len())
                .and_then(|n| bytes.checked_add(n));
            let Some(next_bytes) = next_bytes.filter(|&n| n <= max_bytes) else {
                result.limit = Some(PrefixScanLimit::Bytes);
                break;
            };
            bytes = next_bytes;
            result.entries.push((key.to_vec(), value.to_vec()));
        }
        if reject_oversized
            && result.entries.is_empty()
            && result.limit == Some(PrefixScanLimit::Bytes)
        {
            return Err(prefix_scan_budget_error(prefix, after_key, max_bytes));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pagedb::vfs::memory::MemVfs;

    use crate::storage::engine::StorageEngine;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    #[tokio::test]
    async fn continuation_is_exclusive_prefix_confined_and_namespace_free() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let keys: &[&[u8]] = &[b"p", b"p\0", b"p\0\0", b"p\0a", b"pa"];
        for key in keys {
            storage.put(Namespace::Graph, key, b"v").await.unwrap();
        }
        storage
            .put(Namespace::Graph, b"q", b"neighbor")
            .await
            .unwrap();
        storage
            .put(Namespace::Vector, b"p", b"other namespace")
            .await
            .unwrap();
        let mut cursor = None;
        let mut seen = Vec::new();
        loop {
            let batch = storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", cursor.as_deref(), 2, 100)
                .await
                .unwrap();
            assert!(batch.entries.len() <= 2);
            if let Some((key, _)) = batch.entries.last() {
                cursor = Some(key.clone());
            }
            seen.extend(batch.entries);
            if batch.limit.is_none() {
                break;
            }
        }
        assert_eq!(
            seen,
            keys.iter()
                .map(|key| (key.to_vec(), b"v".to_vec()))
                .collect::<Vec<_>>()
        );
        assert!(seen.windows(2).all(|rows| rows[0].0 < rows[1].0));
        assert!(matches!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"q"), 1, 100,)
                .await,
            Err(LiteError::BadRequest { .. })
        ));
        assert!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"q"), 0, 0,)
                .await
                .unwrap()
                .entries
                .is_empty()
        );
    }

    #[tokio::test]
    async fn continuation_byte_boundary_and_oversized_rows_preserve_cursor() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        for (key, value) in [
            (b"p1".as_slice(), b"abc".as_slice()),
            (b"p2", b"0123456789"),
            (b"p3", b"x"),
        ] {
            storage.put(Namespace::Graph, key, value).await.unwrap();
        }
        let batch = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", None, 3, 5)
            .await
            .unwrap();
        assert_eq!(batch.entries, vec![(b"p1".to_vec(), b"abc".to_vec())]);
        assert_eq!(batch.limit, Some(PrefixScanLimit::Bytes));
        let error = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"p1"), 3, 5)
            .await
            .unwrap_err();
        assert!(matches!(error, LiteError::Backpressure { .. }));
        assert!(error.to_string().contains("5 key-plus-value bytes"));
        let resumed = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"p1"), 3, 15)
            .await
            .unwrap();
        assert_eq!(resumed.entries.len(), 2);
        assert_eq!(resumed.entries[0].0, b"p2");
        assert_eq!(resumed.entries[1].0, b"p3");
        assert_eq!(resumed.limit, None);
        assert!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", None, 1, 4)
                .await
                .is_err()
        );
        assert!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"missing", None, 1, 0)
                .await
                .unwrap()
                .entries
                .is_empty()
        );
        let old = storage
            .scan_prefix_budgeted(Namespace::Graph, b"p", 1, 4)
            .await
            .unwrap();
        assert!(old.entries.is_empty());
        assert_eq!(old.limit, Some(PrefixScanLimit::Bytes));
    }
    async fn make_storage() -> PagedbStorage<MemVfs> {
        PagedbStorage::open_in_memory().await.unwrap()
    }

    #[tokio::test]
    async fn scan_prefix_basic() {
        let s = make_storage().await;
        s.put(Namespace::Vector, b"vec:001", b"a").await.unwrap();
        s.put(Namespace::Vector, b"vec:002", b"b").await.unwrap();
        s.put(Namespace::Vector, b"vec:003", b"c").await.unwrap();
        s.put(Namespace::Vector, b"other:001", b"d").await.unwrap();

        let results = s.scan_prefix(Namespace::Vector, b"vec:").await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, b"vec:001");
        assert_eq!(results[1].0, b"vec:002");
        assert_eq!(results[2].0, b"vec:003");
    }

    #[tokio::test]
    async fn scan_prefix_empty_returns_all() {
        let s = make_storage().await;
        s.put(Namespace::Meta, b"a", b"1").await.unwrap();
        s.put(Namespace::Meta, b"b", b"2").await.unwrap();
        s.put(Namespace::Vector, b"c", b"3").await.unwrap();

        let results = s.scan_prefix(Namespace::Meta, b"").await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn scan_prefix_no_match() {
        let s = make_storage().await;
        s.put(Namespace::Graph, b"edge:1", b"data").await.unwrap();
        let results = s.scan_prefix(Namespace::Graph, b"node:").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn scan_range_with_limit() {
        let s = make_storage().await;
        for i in 0u8..10 {
            s.put(Namespace::Vector, &[i], &[i * 2]).await.unwrap();
        }
        let results = s.scan_range(Namespace::Vector, &[0], 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, &[0u8]);
        assert_eq!(results[1].0, &[1u8]);
        assert_eq!(results[2].0, &[2u8]);
    }

    #[tokio::test]
    async fn scan_range_bounded_with_start_and_end() {
        let s = make_storage().await;
        for i in 0u8..10 {
            s.put(Namespace::Graph, &[i], &[i]).await.unwrap();
        }
        // Keys [2, 3, 4] — start inclusive, end exclusive.
        let results = s
            .scan_range_bounded(Namespace::Graph, Some(&[2]), Some(&[5]), None)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, &[2u8]);
        assert_eq!(results[1].0, &[3u8]);
        assert_eq!(results[2].0, &[4u8]);
    }

    /// Keys in namespace N must not appear in a scan of namespace N+1, and vice versa. Verifies the single-byte prefix boundary.
    #[tokio::test]
    async fn scan_range_bounded_namespace_isolation() {
        let s = make_storage().await;

        // Write keys into two consecutive namespaces.
        for i in 0u8..5 {
            s.put(Namespace::Vector, &[i], b"vec").await.unwrap();
        }
        for i in 0u8..5 {
            s.put(Namespace::Graph, &[i], b"graph").await.unwrap();
        }

        // Full unbounded scan of Vector must return only Vector entries.
        let vec_results = s
            .scan_range_bounded(Namespace::Vector, None, None, None)
            .await
            .unwrap();
        assert_eq!(
            vec_results.len(),
            5,
            "Vector scan leaked into another namespace"
        );
        assert!(vec_results.iter().all(|(_, v)| v == b"vec"));

        // Full unbounded scan of Graph must return only Graph entries.
        let graph_results = s
            .scan_range_bounded(Namespace::Graph, None, None, None)
            .await
            .unwrap();
        assert_eq!(
            graph_results.len(),
            5,
            "Graph scan leaked into another namespace"
        );
        assert!(graph_results.iter().all(|(_, v)| v == b"graph"));
    }
}
