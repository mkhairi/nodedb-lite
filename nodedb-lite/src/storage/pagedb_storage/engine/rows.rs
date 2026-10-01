// SPDX-License-Identifier: Apache-2.0

//! Namespaced rows, ordered batches, and compaction.

use std::collections::HashSet;

use bytes::Bytes;
use nodedb_types::Namespace;
use pagedb::vfs::Vfs;

use crate::error::LiteError;
use crate::storage::engine::{CompactionOutcome, WriteOp};
use crate::storage::pagedb_storage::keys::{KeyBuf, prefix_key};
use crate::storage::pagedb_storage::types::PagedbStorage;

impl<V: Vfs + Clone> PagedbStorage<V> {
    pub(super) async fn get_rows(
        &self,
        ns: Namespace,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, LiteError> {
        let composite = KeyBuf::new(ns, key);
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        // pagedb hands back a `Bytes` sharing the cached page; `StorageEngine` is defined in owned `Vec<u8>`, so the borrow ends at this boundary.
        txn.get(composite.as_slice())
            .await
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(LiteError::from)
    }

    pub(super) async fn put_rows(
        &self,
        ns: Namespace,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), LiteError> {
        let composite = prefix_key(ns, key);
        let mut txn = self.db.begin_write().await.map_err(LiteError::from)?;
        txn.put(&composite, value).await.map_err(LiteError::from)?;
        txn.commit().await.map(|_| ()).map_err(LiteError::from)
    }

    pub(super) async fn delete_rows(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
        let composite = prefix_key(ns, key);
        let mut txn = self.db.begin_write().await.map_err(LiteError::from)?;
        txn.delete(&composite).await.map_err(LiteError::from)?;
        txn.commit().await.map(|_| ()).map_err(LiteError::from)
    }

    pub(super) async fn batch_write_rows(&self, ops: &[WriteOp]) -> Result<(), LiteError> {
        if ops.is_empty() {
            return Ok(());
        }

        let mut txn = self.db.begin_write().await.map_err(LiteError::from)?;

        // Detect duplicate keys (a key that appears in both a Put and a Delete, or appears multiple times). When duplicates exist we fall through to sequential per-op application to preserve original-order semantics. Uniqueness check: if all keys are distinct we can use the fast batch path.
        let all_keys: Vec<Vec<u8>> = ops
            .iter()
            .map(|op| match op {
                WriteOp::Put { ns, key, .. } => prefix_key(*ns, key),
                WriteOp::Delete { ns, key } => prefix_key(*ns, key),
            })
            .collect();
        let unique_count = all_keys.iter().collect::<HashSet<_>>().len();

        if unique_count < all_keys.len() {
            // Duplicate keys present — apply in order to preserve last-write semantics.
            for (op, composite) in ops.iter().zip(all_keys) {
                match op {
                    WriteOp::Put { value, .. } => {
                        txn.put(&composite, value).await.map_err(LiteError::from)?;
                    }
                    WriteOp::Delete { .. } => {
                        txn.delete(&composite).await.map_err(LiteError::from)?;
                    }
                }
            }
        } else {
            // All keys distinct — partition into sorted puts + sorted deletes, then call the batch APIs within the same WriteTxn (both commit atomically). `put_batch` takes `Bytes` so the tree can store the buffer without re-copying it; `delete_batch` still takes owned key vectors.
            let mut puts: Vec<(Bytes, Bytes)> = Vec::new();
            let mut deletes: Vec<Vec<u8>> = Vec::new();

            for (op, composite) in ops.iter().zip(all_keys) {
                match op {
                    WriteOp::Put { value, .. } => {
                        puts.push((Bytes::from(composite), Bytes::from(value.clone())));
                    }
                    WriteOp::Delete { .. } => {
                        deletes.push(composite);
                    }
                }
            }

            puts.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
            deletes.sort_unstable();

            if !puts.is_empty() {
                txn.put_batch(puts).await.map_err(LiteError::from)?;
            }
            if !deletes.is_empty() {
                txn.delete_batch(deletes).await.map_err(LiteError::from)?;
            }
        }

        txn.commit().await.map(|_| ()).map_err(LiteError::from)
    }

    pub(super) async fn count_rows(&self, ns: Namespace) -> Result<u64, LiteError> {
        // No count primitive in pagedb B+ tree — scan the prefix and count.
        let ns_prefix = vec![ns as u8];
        let txn = self.db.begin_read().await.map_err(LiteError::from)?;
        let raw = txn.scan_prefix(&ns_prefix).await.map_err(LiteError::from)?;
        Ok(raw.len() as u64)
    }

    pub(super) async fn compact_rows(&self) -> Result<CompactionOutcome, LiteError> {
        let stats = self.db.compact_now().await.map_err(LiteError::from)?;
        // `compact_now` repacks and truncates; it does not touch retired segment files. Reclaiming those is `gc_now`, which picks up the retirements that a reader pin deferred past their commit.
        let gc = self.db.gc_now().await.map_err(LiteError::from)?;
        Ok(CompactionOutcome {
            reclaimed_pages: stats.main_db_pages_reclaimed,
            segments_repacked: stats.segments_repacked,
            file_bytes_freed: stats.bytes_truncated,
            reclaimed_segments: gc.reclaimed_segments,
            segment_bytes_freed: gc.reclaimed_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use pagedb::vfs::memory::MemVfs;

    use nodedb_types::Namespace;

    use crate::storage::engine::{StorageEngine, WriteOp};
    use crate::storage::pagedb_storage::types::PagedbStorage;

    async fn make_storage() -> PagedbStorage<MemVfs> {
        PagedbStorage::open_in_memory().await.unwrap()
    }

    #[tokio::test]
    async fn put_get_roundtrip() {
        let s = make_storage().await;
        s.put(Namespace::Vector, b"v1", b"hello").await.unwrap();
        let val = s.get(Namespace::Vector, b"v1").await.unwrap();
        assert_eq!(val.as_deref(), Some(b"hello".as_slice()));
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let s = make_storage().await;
        let val = s.get(Namespace::Vector, b"nope").await.unwrap();
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn put_overwrites() {
        let s = make_storage().await;
        s.put(Namespace::Graph, b"k", b"first").await.unwrap();
        s.put(Namespace::Graph, b"k", b"second").await.unwrap();
        let val = s.get(Namespace::Graph, b"k").await.unwrap();
        assert_eq!(val.as_deref(), Some(b"second".as_slice()));
    }

    #[tokio::test]
    async fn delete_removes_key() {
        let s = make_storage().await;
        s.put(Namespace::Crdt, b"k", b"val").await.unwrap();
        s.delete(Namespace::Crdt, b"k").await.unwrap();
        assert!(s.get(Namespace::Crdt, b"k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_is_noop() {
        let s = make_storage().await;
        s.delete(Namespace::Meta, b"ghost").await.unwrap();
    }

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let s = make_storage().await;
        s.put(Namespace::Vector, b"k", b"vec").await.unwrap();
        s.put(Namespace::Graph, b"k", b"graph").await.unwrap();

        assert_eq!(
            s.get(Namespace::Vector, b"k").await.unwrap().as_deref(),
            Some(b"vec".as_slice())
        );
        assert_eq!(
            s.get(Namespace::Graph, b"k").await.unwrap().as_deref(),
            Some(b"graph".as_slice())
        );
    }

    #[tokio::test]
    async fn batch_write_atomic() {
        let s = make_storage().await;
        s.put(Namespace::Crdt, b"to_delete", b"old").await.unwrap();

        s.batch_write(&[
            WriteOp::Put {
                ns: Namespace::Crdt,
                key: b"new1".to_vec(),
                value: b"val1".to_vec(),
            },
            WriteOp::Put {
                ns: Namespace::Crdt,
                key: b"new2".to_vec(),
                value: b"val2".to_vec(),
            },
            WriteOp::Delete {
                ns: Namespace::Crdt,
                key: b"to_delete".to_vec(),
            },
        ])
        .await
        .unwrap();

        assert!(s.get(Namespace::Crdt, b"new1").await.unwrap().is_some());
        assert!(s.get(Namespace::Crdt, b"new2").await.unwrap().is_some());
        assert!(
            s.get(Namespace::Crdt, b"to_delete")
                .await
                .unwrap()
                .is_none()
        );
    }

    /// Same-key put-then-delete in a batch: the delete must win.
    #[tokio::test]
    async fn batch_write_same_key_put_then_delete() {
        let s = make_storage().await;
        s.batch_write(&[
            WriteOp::Put {
                ns: Namespace::Meta,
                key: b"clash".to_vec(),
                value: b"written".to_vec(),
            },
            WriteOp::Delete {
                ns: Namespace::Meta,
                key: b"clash".to_vec(),
            },
        ])
        .await
        .unwrap();
        // Delete came after Put in the ops slice, so the key must be absent.
        assert!(s.get(Namespace::Meta, b"clash").await.unwrap().is_none());
    }

    /// Same-key delete-then-put in a batch: the put must win.
    #[tokio::test]
    async fn batch_write_same_key_delete_then_put() {
        let s = make_storage().await;
        s.put(Namespace::Meta, b"exists", b"old").await.unwrap();
        s.batch_write(&[
            WriteOp::Delete {
                ns: Namespace::Meta,
                key: b"exists".to_vec(),
            },
            WriteOp::Put {
                ns: Namespace::Meta,
                key: b"exists".to_vec(),
                value: b"new".to_vec(),
            },
        ])
        .await
        .unwrap();
        // Put came after Delete, so the key must be present with the new value.
        assert_eq!(
            s.get(Namespace::Meta, b"exists").await.unwrap().as_deref(),
            Some(b"new".as_slice())
        );
    }

    #[tokio::test]
    async fn batch_write_empty_is_noop() {
        let s = make_storage().await;
        s.batch_write(&[]).await.unwrap();
    }

    #[tokio::test]
    async fn count_entries() {
        let s = make_storage().await;
        assert_eq!(s.count(Namespace::Vector).await.unwrap(), 0);

        s.put(Namespace::Vector, b"v1", b"a").await.unwrap();
        s.put(Namespace::Vector, b"v2", b"b").await.unwrap();
        s.put(Namespace::Graph, b"g1", b"c").await.unwrap();

        assert_eq!(s.count(Namespace::Vector).await.unwrap(), 2);
        assert_eq!(s.count(Namespace::Graph).await.unwrap(), 1);
        assert_eq!(s.count(Namespace::Crdt).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn large_value_roundtrip() {
        let s = make_storage().await;
        let large = vec![0xABu8; 1_000_000];
        s.put(Namespace::Vector, b"hnsw:layer0", &large)
            .await
            .unwrap();
        let val = s.get(Namespace::Vector, b"hnsw:layer0").await.unwrap();
        assert_eq!(val.unwrap().len(), 1_000_000);
    }

    /// In-memory engine: `compact()` is a successful no-op (nothing to reclaim).
    #[tokio::test]
    async fn compact_mem_is_ok_noop() {
        let s = make_storage().await;
        s.put(Namespace::Vector, b"v1", b"hello").await.unwrap();
        s.put(Namespace::Graph, b"g1", b"world").await.unwrap();
        let outcome = s.compact().await.unwrap();
        // Data still readable after compaction.
        assert_eq!(
            s.get(Namespace::Vector, b"v1").await.unwrap().as_deref(),
            Some(b"hello".as_slice())
        );
        // MemVfs has no file truncation, but the call must succeed regardless.
        let _ = outcome.reclaimed_pages;
    }

    /// Disk-backed engine on a tempdir: write rows (including churn that leaves dead pages), then `compact()` must succeed and report a non-negative outcome. Data must remain intact afterward.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn compact_default_disk_is_ok() {
        use pagedb::vfs::DefaultVfs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compact-test.db");
        let s = PagedbStorage::<DefaultVfs>::open(
            &path,
            crate::storage::encryption::Encryption::Plaintext,
        )
        .await
        .unwrap();

        // Churn: write then overwrite/delete a batch of keys so the deferred-free list has pages to reclaim.
        for i in 0u32..200 {
            let key = i.to_be_bytes();
            s.put(Namespace::Meta, &key, &vec![0xCDu8; 512])
                .await
                .unwrap();
        }
        for i in 0u32..150 {
            let key = i.to_be_bytes();
            s.delete(Namespace::Meta, &key).await.unwrap();
        }

        let outcome = s.compact().await.unwrap();

        // Surviving keys still readable.
        let survivor = 175u32.to_be_bytes();
        assert!(s.get(Namespace::Meta, &survivor).await.unwrap().is_some());

        // Outcome fields are well-formed (u64/u32 — always >= 0); just touch them so the assertion documents the reported shape.
        let _ = (
            outcome.reclaimed_pages,
            outcome.segments_repacked,
            outcome.file_bytes_freed,
        );
    }
}
