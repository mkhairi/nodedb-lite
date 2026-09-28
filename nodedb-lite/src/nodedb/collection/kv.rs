//! KV collection operations for Lite.
//!
//! Reads and writes go to the KV store in `Namespace::Kv`. KV never touches
//! Loro. With sync on, every put and delete is also recorded in the KV
//! outbound queue before it applies locally, and the push loop sends it to
//! Origin. See `kv_sync`.
//!
//! Writes are buffered in memory and flushed as a single KV transaction
//! on `kv_flush()` or when the buffer exceeds `KV_FLUSH_THRESHOLD`. An
//! in-memory overlay lets reads see uncommitted writes without hitting the
//! KV store.
//!
//! ## Value encoding
//!
//! Every value stored in the KV store is prefixed by an 8-byte little-endian
//! u64 representing the expiry deadline in milliseconds since the Unix epoch.
//! A value of `0` means no expiry. This prefix is transparent to callers —
//! all public methods encode/decode it automatically.

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::super::{LockExt, NodeDbLite};
use crate::storage::engine::{StorageEngine, WriteOp};

/// Flush the write buffer when it reaches this many operations.
pub(super) const KV_FLUSH_THRESHOLD: usize = 1024;

/// Size of the deadline prefix in bytes (u64 LE).
const DEADLINE_PREFIX_LEN: usize = 8;

/// Build the composite KV key: `{collection}\0{key}`.
pub(super) fn kv_key(collection: &str, key: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(collection.len() + 1 + key.len());
    k.extend_from_slice(collection.as_bytes());
    k.push(0);
    k.extend_from_slice(key);
    k
}

/// Extract `(collection, key_bytes)` from a composite KV key.
pub(super) fn split_kv_key(composite: &[u8]) -> Option<(&str, &[u8])> {
    let sep = composite.iter().position(|&b| b == 0)?;
    let coll = std::str::from_utf8(&composite[..sep]).ok()?;
    let key = &composite[sep + 1..];
    Some((coll, key))
}

/// Encode a value with a deadline prefix.
///
/// `deadline_ms = 0` encodes as "no expiry".
pub(super) fn encode_value(deadline_ms: u64, value: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(DEADLINE_PREFIX_LEN + value.len());
    encoded.extend_from_slice(&deadline_ms.to_le_bytes());
    encoded.extend_from_slice(value);
    encoded
}

/// Decode a stored value into `(deadline_ms, user_bytes)`.
///
/// Returns `None` if the stored bytes are too short (corrupt entry).
pub(super) fn decode_value(stored: &[u8]) -> Option<(u64, &[u8])> {
    if stored.len() < DEADLINE_PREFIX_LEN {
        return None;
    }
    let deadline = u64::from_le_bytes(stored[..DEADLINE_PREFIX_LEN].try_into().ok()?);
    Some((deadline, &stored[DEADLINE_PREFIX_LEN..]))
}

/// Return `true` if the deadline has passed (key is expired).
///
/// A deadline of `0` means no expiry and is never considered expired.
pub(super) fn is_expired(deadline_ms: u64) -> bool {
    deadline_ms != 0 && crate::runtime::now_millis() >= deadline_ms
}

impl<S: StorageEngine> NodeDbLite<S> {
    /// KV PUT: store a key-value pair with no expiry.
    ///
    /// Buffered in memory — call `kv_flush()` to commit, or let
    /// the auto-flush threshold handle it.
    pub async fn kv_put(&self, collection: &str, key: &str, value: &[u8]) -> NodeDbResult<()> {
        self.kv_put_with_deadline(collection, key, value, 0).await
    }

    /// KV PUT WITH TTL: store a key-value pair that expires after `ttl_ms` ms.
    ///
    /// After `ttl_ms` milliseconds, `kv_get` will return `None` for this key
    /// and lazy-delete it. The deadline survives a database reopen.
    pub async fn kv_put_with_ttl(
        &self,
        collection: &str,
        key: &str,
        value: &[u8],
        ttl_ms: u64,
    ) -> NodeDbResult<()> {
        let deadline = crate::runtime::now_millis().saturating_add(ttl_ms);
        self.kv_put_with_deadline(collection, key, value, deadline)
            .await
    }

    /// Internal: write a key with an explicit deadline (0 = no expiry).
    ///
    /// The write is recorded for sync before it applies, under the
    /// write-order lock.
    async fn kv_put_with_deadline(
        &self,
        collection: &str,
        key: &str,
        value: &[u8],
        deadline_ms: u64,
    ) -> NodeDbResult<()> {
        self.kv_check_pressure()?;
        let _order = self.kv_local.write_order.lock().await;
        if self.kv_local.is_indexed(collection) {
            return self
                .kv_put_indexed(collection, key, value, deadline_ms)
                .await;
        }
        self.kv_record_put(collection, key.as_bytes(), value, deadline_ms)
            .await?;
        self.kv_buffer_put(collection, key, value, deadline_ms)
            .await
    }

    /// Put into a collection with a key-value index: refused before it is
    /// recorded for sync when a unique index forbids it, then committed
    /// directly after the buffered writes ahead of it, with its entries.
    /// Call with the write-order lock held.
    async fn kv_put_indexed(
        &self,
        collection: &str,
        key: &str,
        value: &[u8],
        deadline_ms: u64,
    ) -> NodeDbResult<()> {
        let rkey = kv_key(collection, key.as_bytes());
        let op = WriteOp::Put {
            ns: Namespace::Kv,
            key: rkey.clone(),
            value: encode_value(deadline_ms, value),
        };
        self.kv_local
            .check(&*self.storage, std::slice::from_ref(&op))
            .await
            .map_err(NodeDbError::from)?;
        self.kv_record_put(collection, key.as_bytes(), value, deadline_ms)
            .await?;
        self.kv_flush_inner().await?;
        self.kv_local
            .commit(&*self.storage, vec![op])
            .await
            .map_err(NodeDbError::from)?;
        self.kv_local.cache.lock_or_recover().pop(&rkey);
        Ok(())
    }

    /// Refuse a KV write while the memory governor is at Emergency pressure.
    pub(super) fn kv_check_pressure(&self) -> NodeDbResult<()> {
        if self.governor.worst_engine_pressure() == nodedb_mem::PressureLevel::Emergency {
            return Err(NodeDbError::storage(
                crate::error::LiteError::Backpressure {
                    detail: "KV write rejected: memory governor is at Emergency pressure".into(),
                },
            ));
        }
        Ok(())
    }

    /// Buffer a put of `value` at `key` with the expiry `deadline_ms`.
    /// Records nothing for sync.
    pub(super) async fn kv_buffer_put(
        &self,
        collection: &str,
        key: &str,
        value: &[u8],
        deadline_ms: u64,
    ) -> NodeDbResult<()> {
        let rkey = kv_key(collection, key.as_bytes());
        let encoded = encode_value(deadline_ms, value);

        // Scope all mutex work so no guard is live at the await point.
        let should_flush = {
            let mut buf = self.kv_local.write_buf.lock_or_recover();
            buf.overlay.insert(rkey.clone(), Some(encoded.clone()));
            buf.ops.push(WriteOp::Put {
                ns: Namespace::Kv,
                key: rkey.clone(),
                value: encoded,
            });
            buf.ops.len() >= KV_FLUSH_THRESHOLD
        };

        // Invalidate any cached value for this key so subsequent reads go to storage.
        {
            self.kv_local.cache.lock_or_recover().pop(&rkey);
        }

        if should_flush {
            self.kv_flush_inner().await?;
        }

        Ok(())
    }

    /// KV GET: retrieve a value by key.
    ///
    /// Returns `None` for missing or expired keys. Expired keys are lazily
    /// deleted from storage on read.
    ///
    /// Checks the in-memory write buffer first (for uncommitted writes),
    /// then falls through to the KV store.
    pub async fn kv_get(&self, collection: &str, key: &str) -> NodeDbResult<Option<Vec<u8>>> {
        let rkey = kv_key(collection, key.as_bytes());

        // Always acquire the write-buffer lock to check the overlay first.
        // This prevents torn reads that could occur if an unconditional
        // Acquire load of a length counter raced with a concurrent writer.
        // Scope the guard so it is not live at any await point.
        let overlay_result: Option<Option<Vec<u8>>> = {
            let buf = self.kv_local.write_buf.lock_or_recover();
            buf.overlay.get(&rkey).map(|entry| match entry {
                Some(stored) => decode_value(stored).and_then(|(deadline, user_bytes)| {
                    if is_expired(deadline) {
                        None
                    } else {
                        Some(user_bytes.to_vec())
                    }
                }),
                None => None,
            })
        };
        if let Some(result) = overlay_result {
            return Ok(result);
        }

        // Cache check: look up the composite key before hitting storage.
        // Guard scoped to the block; no await inside.
        let cache_result: Option<Option<Vec<u8>>> = {
            let mut cache = self.kv_local.cache.lock_or_recover();
            if let Some(encoded) = cache.get(&rkey) {
                match decode_value(encoded) {
                    Some((deadline, user_bytes)) if !is_expired(deadline) => {
                        Some(Some(user_bytes.to_vec()))
                    }
                    _ => {
                        cache.pop(&rkey);
                        None
                    }
                }
            } else {
                None
            }
        };
        if let Some(result) = cache_result {
            return Ok(result);
        }

        // Fall through to storage.
        let stored = self
            .storage
            .get(Namespace::Kv, &rkey)
            .await
            .map_err(NodeDbError::storage)?;

        match stored {
            None => Ok(None),
            Some(raw) => {
                let decoded = decode_value(&raw);
                match decoded {
                    None => Ok(None),
                    Some((deadline, user_bytes)) => {
                        if is_expired(deadline) {
                            // Lazy expiration: schedule a delete.
                            self.kv_lazy_delete(rkey).await?;
                            Ok(None)
                        } else {
                            let result = user_bytes.to_vec();
                            // Populate cache with the raw encoded bytes before returning.
                            // Guard scoped to block; no await follows inside this branch.
                            {
                                self.kv_local.cache.lock_or_recover().put(rkey, raw);
                            }
                            Ok(Some(result))
                        }
                    }
                }
            }
        }
    }

    /// Internal: queue a lazy delete for an expired key.
    async fn kv_lazy_delete(&self, rkey: Vec<u8>) -> NodeDbResult<()> {
        let should_flush = {
            let mut buf = self.kv_local.write_buf.lock_or_recover();
            buf.overlay.insert(rkey.clone(), None);
            buf.ops.push(WriteOp::Delete {
                ns: Namespace::Kv,
                key: rkey.clone(),
            });
            buf.ops.len() >= KV_FLUSH_THRESHOLD
        };
        // Evict the expired entry so future reads don't serve stale data.
        {
            self.kv_local.cache.lock_or_recover().pop(&rkey);
        }
        if should_flush {
            self.kv_flush_inner().await?;
        }
        Ok(())
    }

    /// KV DELETE: remove a key.
    ///
    /// Returns `true` when a live value was present and removed, `false` when
    /// the key was absent (or already expired) — mirroring `HashMap::remove`.
    ///
    /// The delete is recorded for sync even when the key is absent here,
    /// because Origin can hold the key when this replica does not.
    pub async fn kv_delete(&self, collection: &str, key: &str) -> NodeDbResult<bool> {
        let _order = self.kv_local.write_order.lock().await;
        // Capture prior presence so the returned bool means "a live value was
        // removed" rather than an unconditional `true`.
        let existed = self.kv_get(collection, key).await?.is_some();
        self.kv_record_delete(collection, key.as_bytes()).await?;

        let rkey = kv_key(collection, key.as_bytes());

        let should_flush = {
            let mut buf = self.kv_local.write_buf.lock_or_recover();
            buf.overlay.insert(rkey.clone(), None);
            buf.ops.push(WriteOp::Delete {
                ns: Namespace::Kv,
                key: rkey.clone(),
            });
            buf.ops.len() >= KV_FLUSH_THRESHOLD
        };

        // Invalidate the cache so subsequent reads don't return stale data.
        {
            self.kv_local.cache.lock_or_recover().pop(&rkey);
        }

        if should_flush {
            self.kv_flush_inner().await?;
        }

        Ok(existed)
    }

    /// Flush buffered KV writes to storage as a single transaction.
    /// Returns the number of writes flushed.
    pub async fn kv_flush(&self) -> NodeDbResult<usize> {
        self.kv_flush_inner().await
    }

    /// Internal: flush the write buffer to storage.
    /// `pub(in crate::nodedb)` so the global `flush()` can drain the KV buffer.
    pub(in crate::nodedb) async fn kv_flush_inner(&self) -> NodeDbResult<usize> {
        self.kv_local
            .flush_to(&*self.storage)
            .await
            .map_err(NodeDbError::storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LiteConfig;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    async fn open_db() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory()
            .await
            .expect("open in-memory storage");
        NodeDbLite::open(storage).await.expect("open NodeDbLite")
    }

    async fn open_db_with_cache_capacity(
        cap: usize,
    ) -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory()
            .await
            .expect("open in-memory storage");
        let config = LiteConfig {
            kv_cache_capacity: cap,
            ..LiteConfig::default()
        };
        NodeDbLite::open_with_config(storage, config)
            .await
            .expect("open NodeDbLite with config")
    }

    /// Two consecutive gets on the same key: both return the same value.
    /// The second read is served from the in-process cache.
    #[tokio::test]
    async fn cache_hits_on_repeated_get() {
        let db = open_db().await;
        db.kv_put("col", "key", b"hello").await.unwrap();
        // Prime the cache.
        db.kv_flush().await.unwrap();
        let v1 = db.kv_get("col", "key").await.unwrap();
        assert_eq!(v1.as_deref(), Some(b"hello".as_ref()));
        // Second get — served from cache.
        let v2 = db.kv_get("col", "key").await.unwrap();
        assert_eq!(v2.as_deref(), Some(b"hello".as_ref()));
        // Verify the cache actually holds the entry.
        assert_eq!(db.kv_local.cache.lock_or_recover().len(), 1);
    }

    /// After a put-get-put sequence the second get must return the new value,
    /// not the stale cached one.
    #[tokio::test]
    async fn cache_invalidated_on_put() {
        let db = open_db().await;
        db.kv_put("col", "key", b"v1").await.unwrap();
        db.kv_flush().await.unwrap();
        let _ = db.kv_get("col", "key").await.unwrap(); // populate cache
        db.kv_put("col", "key", b"v2").await.unwrap();
        db.kv_flush().await.unwrap();
        let v = db.kv_get("col", "key").await.unwrap();
        assert_eq!(v.as_deref(), Some(b"v2".as_ref()));
    }

    /// After a put-get-delete sequence a subsequent get must return None.
    #[tokio::test]
    async fn cache_invalidated_on_delete() {
        let db = open_db().await;
        db.kv_put("col", "key", b"v").await.unwrap();
        db.kv_flush().await.unwrap();
        let _ = db.kv_get("col", "key").await.unwrap(); // populate cache
        db.kv_delete("col", "key").await.unwrap();
        db.kv_flush().await.unwrap();
        let v = db.kv_get("col", "key").await.unwrap();
        assert!(v.is_none(), "deleted key must not be returned from cache");
    }

    /// A key written with a very short TTL must not be served from cache after expiry.
    #[tokio::test]
    async fn expired_cached_value_evicted() {
        let db = open_db().await;
        // 1 ms TTL — expired almost immediately.
        db.kv_put_with_ttl("col", "key", b"v", 1).await.unwrap();
        db.kv_flush().await.unwrap();
        let _ = db.kv_get("col", "key").await.unwrap(); // may or may not cache
        // Sleep long enough that the deadline has definitely passed.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let v = db.kv_get("col", "key").await.unwrap();
        assert!(v.is_none(), "expired key must return None");
        // Cache must not hold the evicted entry.
        assert_eq!(
            db.kv_local.cache.lock_or_recover().len(),
            0,
            "expired entry must be evicted from cache"
        );
    }

    /// The cache must not grow beyond the configured capacity.
    #[tokio::test]
    async fn cache_capacity_eviction() {
        const CAP: usize = 5;
        let db = open_db_with_cache_capacity(CAP).await;
        let col = "cap_test";

        // Write N+10 keys and flush so they land in storage.
        for i in 0..(CAP + 10) {
            db.kv_put(col, &i.to_string(), b"x").await.unwrap();
        }
        db.kv_flush().await.unwrap();

        // Read all keys — each miss populates the cache.
        for i in 0..(CAP + 10) {
            let _ = db.kv_get(col, &i.to_string()).await.unwrap();
        }

        let cache_len = db.kv_local.cache.lock_or_recover().len();
        assert!(
            cache_len <= CAP,
            "cache must not exceed capacity {CAP}, got {cache_len}"
        );
    }
}
