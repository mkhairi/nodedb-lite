// SPDX-License-Identifier: Apache-2.0

//! KV state the public API keeps in front of storage: the write buffer with
//! its read overlay, the read cache, and the write-order lock.
//!
//! `NodeDbLite` and the query engine share it:
//! - a SQL-path `TRUNCATE` forgets what the buffer and cache hold for the
//!   collection it cleared, so a buffered put cannot reach storage on the
//!   next flush and a cached value is not served after its row is gone;
//! - public-API and SQL writes take one write-order lock, so both enqueue
//!   for sync in the order they applied.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use crate::error::LiteError;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::{StorageEngine, WriteOp};

/// Buffered KV writes for batch commit.
///
/// All public KV read and write methods acquire `Mutex<KvWriteBuffer>` before
/// inspecting or mutating the overlay, so every read-through-overlay access is
/// serialized against concurrent writes. Reads always lock — there is no
/// lock-free fast path.
pub(crate) struct KvWriteBuffer {
    /// Pending write operations for batch commit.
    pub ops: Vec<WriteOp>,
    /// Read overlay: maps composite KV key → value (None = deleted).
    /// Lets `kv_get` see uncommitted writes without hitting storage.
    pub overlay: HashMap<Vec<u8>, Option<Vec<u8>>>,
}

/// The write buffer and read cache of the public KV API.
pub struct KvLocalState {
    /// Buffered KV writes awaiting batch commit to storage.
    /// `kv_flush()`, the 1024-operation threshold, and `flush()` drain this buffer.
    /// The HashMap overlay lets reads see uncommitted writes.
    pub(crate) write_buf: Mutex<KvWriteBuffer>,
    /// In-memory LRU cache for the KV get hot path.
    ///
    /// Stores raw encoded bytes (8-byte LE deadline + user value) keyed by the
    /// composite KV key (`{collection}\0{user_key}`). TTL expiry is re-checked
    /// on every cache hit so no entry is served past its deadline.
    ///
    /// Capacity is controlled by [`crate::config::LiteConfig::kv_cache_capacity`].
    pub(crate) cache: Mutex<lru::LruCache<Vec<u8>, Vec<u8>>>,
    /// Serializes every KV write that syncs to Origin, public API and SQL.
    ///
    /// A write holds it from its local apply through its outbound enqueue,
    /// so the outbound queue holds writes in the order they were applied.
    /// Without it, two writes to one key could apply in one order and
    /// enqueue in the other, and Origin would keep the value Lite overwrote.
    /// It also makes a read-modify-write such as `kv_increment` atomic.
    pub(crate) write_order: tokio::sync::Mutex<()>,
    /// Secondary indexes every commit of the buffer maintains. Set once by
    /// the query engine.
    indexes: std::sync::OnceLock<std::sync::Arc<crate::index::IndexCatalog>>,
}

impl KvLocalState {
    pub(crate) fn new(cache_capacity: NonZeroUsize) -> Self {
        Self {
            write_buf: Mutex::new(KvWriteBuffer {
                ops: Vec::with_capacity(1024),
                overlay: HashMap::new(),
            }),
            cache: Mutex::new(lru::LruCache::new(cache_capacity)),
            write_order: tokio::sync::Mutex::new(()),
            indexes: std::sync::OnceLock::new(),
        }
    }

    /// Maintain `catalog`'s key-value indexes from every buffer commit.
    pub(crate) fn set_index_catalog(&self, catalog: std::sync::Arc<crate::index::IndexCatalog>) {
        // Set once, when the query engine is built; a second call keeps the
        // first catalog.
        let _ = self.indexes.set(catalog);
    }

    /// Whether writes to `collection` maintain a key-value index. Such writes
    /// commit directly rather than wait in the buffer, so a unique index
    /// refuses the write that breaks it instead of the batch it lands in.
    pub(crate) fn is_indexed(&self, collection: &str) -> bool {
        self.indexes
            .get()
            .is_some_and(|c| c.has_defs(collection, crate::index::IndexEngine::KeyValue))
    }

    /// Refuse KV writes `ops` a unique index forbids, writing nothing.
    pub(crate) async fn check<S: StorageEngine>(
        &self,
        storage: &S,
        ops: &[WriteOp],
    ) -> Result<(), LiteError> {
        match self.indexes.get() {
            Some(catalog) => crate::query::kv_ops::indexes::kv_check(storage, catalog, ops).await,
            None => Ok(()),
        }
    }

    /// Commit KV writes `ops` with the index entries they imply. Every KV
    /// write reaches storage through here.
    pub(crate) async fn commit<S: StorageEngine>(
        &self,
        storage: &S,
        ops: Vec<WriteOp>,
    ) -> Result<(), LiteError> {
        match self.indexes.get() {
            Some(catalog) => crate::query::kv_ops::indexes::kv_commit(storage, catalog, ops).await,
            None if ops.is_empty() => Ok(()),
            None => storage.batch_write(&ops).await,
        }
    }

    /// Commit every buffered write to `storage` as one batch. Returns the
    /// number of writes committed.
    ///
    /// A SQL KV write calls this before it runs, so a buffered public-API
    /// write to the same key cannot land after it and overwrite it.
    pub(crate) async fn flush_to<S: StorageEngine>(&self, storage: &S) -> Result<usize, LiteError> {
        let ops: Vec<WriteOp> = {
            let mut buf = self.write_buf.lock_or_recover();
            if buf.ops.is_empty() {
                return Ok(0);
            }
            let ops = std::mem::take(&mut buf.ops);
            buf.overlay.clear();
            ops
        };
        let count = ops.len();
        self.commit(storage, ops).await?;
        Ok(count)
    }

    /// Drop every buffered write and cached value whose key belongs to
    /// `collection`. Returns the number of buffered writes dropped.
    pub(crate) fn forget_collection(&self, collection: &str) -> usize {
        let mut prefix = collection.as_bytes().to_vec();
        prefix.push(0);
        let dropped = {
            let mut buf = self.write_buf.lock_or_recover();
            let before = buf.ops.len();
            buf.ops.retain(|op| {
                let key = match op {
                    WriteOp::Put { key, .. } | WriteOp::Delete { key, .. } => key,
                };
                !key.starts_with(&prefix)
            });
            buf.overlay.retain(|key, _| !key.starts_with(&prefix));
            before - buf.ops.len()
        };
        let mut cache = self.cache.lock_or_recover();
        let stale: Vec<Vec<u8>> = cache
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .map(|(key, _)| key.clone())
            .collect();
        for key in stale {
            cache.pop(&key);
        }
        dropped
    }
}

#[cfg(test)]
mod tests {
    use nodedb_types::Namespace;

    use super::*;

    fn key(collection: &str, k: &str) -> Vec<u8> {
        let mut key = collection.as_bytes().to_vec();
        key.push(0);
        key.extend_from_slice(k.as_bytes());
        key
    }

    #[test]
    fn forget_collection_drops_only_that_collection() {
        let state = KvLocalState::new(NonZeroUsize::new(8).expect("cap"));
        {
            let mut buf = state.write_buf.lock_or_recover();
            for (c, k) in [("a", "1"), ("a", "2"), ("b", "1")] {
                buf.ops.push(WriteOp::Put {
                    ns: Namespace::Kv,
                    key: key(c, k),
                    value: vec![1],
                });
                buf.overlay.insert(key(c, k), Some(vec![1]));
            }
            buf.ops.push(WriteOp::Delete {
                ns: Namespace::Kv,
                key: key("a", "3"),
            });
        }
        {
            let mut cache = state.cache.lock_or_recover();
            cache.put(key("a", "1"), vec![1]);
            cache.put(key("b", "1"), vec![1]);
        }

        assert_eq!(state.forget_collection("a"), 3);
        let buf = state.write_buf.lock_or_recover();
        assert_eq!(buf.ops.len(), 1);
        assert_eq!(buf.overlay.len(), 1);
        assert!(buf.overlay.contains_key(&key("b", "1")));
        let cache = state.cache.lock_or_recover();
        assert_eq!(cache.len(), 1);
        assert!(cache.peek(&key("b", "1")).is_some());
    }
}
