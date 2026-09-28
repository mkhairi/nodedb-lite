// SPDX-License-Identifier: Apache-2.0

//! KV reads over key ranges and expiry sweeps: range scan, cursor scan,
//! key listing, shape subscription, and eager expiry compaction.

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::super::{LockExt, NodeDbLite};
use super::kv::{KV_FLUSH_THRESHOLD, decode_value, is_expired, kv_key, split_kv_key};
use crate::storage::engine::{StorageEngine, WriteOp};

impl<S: StorageEngine> NodeDbLite<S> {
    /// KV RANGE SCAN: ordered key scan with optional bounds and limit.
    ///
    /// Returns `(key, value)` pairs where `start <= key < end`, ordered by
    /// key in lexicographic byte order. Expired keys are skipped and lazily
    /// deleted.
    ///
    /// - `start = None` means scan from the beginning of the collection.
    /// - `end = None` means scan to the end of the collection.
    /// - `limit = None` means no cap on results.
    ///
    /// Flushes the write buffer before scanning so the KV store reflects all pending
    /// writes.
    pub async fn kv_range_scan(
        &self,
        collection: &str,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        limit: Option<usize>,
    ) -> NodeDbResult<Vec<(Vec<u8>, Vec<u8>)>> {
        self.kv_flush_inner().await?;

        let col_prefix_end = {
            let mut p = collection.as_bytes().to_vec();
            p.push(0);
            p
        };

        // Build absolute start key (collection\0[user_start]).
        let start_key: Option<Vec<u8>> = Some(match start {
            Some(s) => {
                let mut k = col_prefix_end.clone();
                k.extend_from_slice(s);
                k
            }
            None => col_prefix_end.clone(),
        });

        // Build absolute end key (collection\0[user_end]).
        let end_key: Option<Vec<u8>> = end.map(|e| {
            let mut k = col_prefix_end.clone();
            k.extend_from_slice(e);
            k
        });

        let entries = self
            .storage
            .scan_range_bounded(
                Namespace::Kv,
                start_key.as_deref(),
                end_key.as_deref(),
                limit.map(|l| l + 32), // over-fetch slightly to account for skipped expired keys
            )
            .await
            .map_err(NodeDbError::storage)?;

        let mut results: Vec<(Vec<u8>, Vec<u8>)> =
            Vec::with_capacity(limit.unwrap_or(entries.len()).min(entries.len()));
        let mut expired_keys: Vec<Vec<u8>> = Vec::new();

        for (composite_key, raw_value) in entries {
            if let Some(limit) = limit
                && results.len() >= limit
            {
                break;
            }
            let Some((coll, user_key_bytes)) = split_kv_key(&composite_key) else {
                continue;
            };
            if coll != collection {
                break;
            }
            let Some((deadline, user_bytes)) = decode_value(&raw_value) else {
                continue;
            };
            if is_expired(deadline) {
                expired_keys.push(kv_key(collection, user_key_bytes));
                continue;
            }
            results.push((user_key_bytes.to_vec(), user_bytes.to_vec()));
        }

        // Lazy-delete expired keys discovered during scan.
        if !expired_keys.is_empty() {
            let should_flush = {
                let mut buf = self.kv_local.write_buf.lock_or_recover();
                for rkey in &expired_keys {
                    buf.overlay.insert(rkey.clone(), None);
                    buf.ops.push(WriteOp::Delete {
                        ns: Namespace::Kv,
                        key: rkey.clone(),
                    });
                }
                buf.ops.len() >= KV_FLUSH_THRESHOLD
            };
            // Evict expired keys from the cache.
            {
                let mut cache = self.kv_local.cache.lock_or_recover();
                for rkey in &expired_keys {
                    cache.pop(rkey);
                }
            }
            if should_flush {
                self.kv_flush_inner().await?;
            }
        }

        Ok(results)
    }

    /// KV COMPACT EXPIRED: eagerly remove all expired keys in a collection.
    ///
    /// Flushes the write buffer, then scans all keys in the collection and
    /// deletes any whose TTL deadline has passed. Returns the count of keys
    /// removed.
    pub async fn kv_compact_expired(&self, collection: &str) -> NodeDbResult<usize> {
        self.kv_flush_inner().await?;

        let col_prefix = {
            let mut p = collection.as_bytes().to_vec();
            p.push(0);
            p
        };

        let entries = self
            .storage
            .scan_range_bounded(Namespace::Kv, Some(&col_prefix), None, None)
            .await
            .map_err(NodeDbError::storage)?;

        let now = crate::runtime::now_millis();
        let mut delete_ops: Vec<WriteOp> = Vec::new();

        for (composite_key, raw_value) in entries {
            let Some((coll, _user_key_bytes)) = split_kv_key(&composite_key) else {
                continue;
            };
            if coll != collection {
                break;
            }
            if let Some((deadline, _)) = decode_value(&raw_value)
                && deadline != 0
                && now >= deadline
            {
                // composite_key is the user-key (namespace byte
                // already stripped by scan_range_bounded). WriteOp
                // re-prepends the namespace byte via make_key internally.
                delete_ops.push(WriteOp::Delete {
                    ns: Namespace::Kv,
                    key: composite_key,
                });
            }
        }

        let count = delete_ops.len();
        if count > 0 {
            self.storage
                .batch_write(&delete_ops)
                .await
                .map_err(NodeDbError::storage)?;
        }

        Ok(count)
    }

    /// KV SCAN: iterate keys in sorted order starting from `cursor`.
    ///
    /// Returns up to `count` key-value pairs where key >= cursor (inclusive).
    /// Pass an empty cursor to start from the beginning of the collection.
    ///
    /// Flushes the write buffer first to ensure the KV store has all data, then
    /// uses the storage's B-tree range scan — O(log N + count).
    pub async fn kv_scan(
        &self,
        collection: &str,
        cursor: &str,
        count: usize,
    ) -> NodeDbResult<Vec<(String, Vec<u8>)>> {
        // Flush pending writes so storage is up to date.
        self.kv_flush_inner().await?;

        let start = kv_key(collection, cursor.as_bytes());
        let entries = self
            .storage
            .scan_range(Namespace::Kv, &start, count)
            .await
            .map_err(NodeDbError::storage)?;

        let mut results = Vec::with_capacity(entries.len());
        for (composite_key, raw_value) in entries {
            let Some((coll, key_bytes)) = split_kv_key(&composite_key) else {
                continue;
            };
            if coll != collection {
                break;
            }
            let Some((deadline, user_bytes)) = decode_value(&raw_value) else {
                continue;
            };
            if is_expired(deadline) {
                continue;
            }
            if let Ok(key_str) = std::str::from_utf8(key_bytes) {
                results.push((key_str.to_string(), user_bytes.to_vec()));
            }
        }

        Ok(results)
    }

    /// List all keys in a KV collection.
    pub async fn kv_keys(&self, collection: &str) -> NodeDbResult<Vec<String>> {
        // Flush pending writes first.
        self.kv_flush_inner().await?;

        let prefix = kv_key(collection, b"");
        let entries = self
            .storage
            .scan_range(Namespace::Kv, &prefix, usize::MAX)
            .await
            .map_err(NodeDbError::storage)?;

        let mut keys = Vec::with_capacity(entries.len());
        for (composite_key, raw_value) in entries {
            let Some((coll, key_bytes)) = split_kv_key(&composite_key) else {
                continue;
            };
            if coll != collection {
                break;
            }
            // Skip expired keys.
            if let Some((deadline, _)) = decode_value(&raw_value) {
                if is_expired(deadline) {
                    continue;
                }
            } else {
                continue;
            }
            if let Ok(key_str) = std::str::from_utf8(key_bytes) {
                keys.push(key_str.to_string());
            }
        }
        Ok(keys)
    }

    /// Subscribe to a subset of KV keys matching a pattern.
    pub async fn kv_subscribe_shape(
        &self,
        collection: &str,
        key_pattern: &str,
    ) -> NodeDbResult<Vec<String>> {
        let all_keys = self.kv_keys(collection).await?;
        let matched: Vec<String> = all_keys
            .into_iter()
            .filter(|k| glob_matches(key_pattern, k))
            .collect();
        Ok(matched)
    }
}

/// Simple glob matching for shape subscriptions.
fn glob_matches(pattern: &str, input: &str) -> bool {
    let pat = pattern.as_bytes();
    let inp = input.as_bytes();
    let mut pi = 0;
    let mut ii = 0;
    let mut star_pi = usize::MAX;
    let mut star_ii = 0;

    while ii < inp.len() {
        if pi < pat.len() && (pat[pi] == b'?' || pat[pi] == inp[ii]) {
            pi += 1;
            ii += 1;
        } else if pi < pat.len() && pat[pi] == b'*' {
            star_pi = pi;
            star_ii = ii;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ii += 1;
            ii = star_ii;
        } else {
            return false;
        }
    }

    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }

    pi == pat.len()
}
