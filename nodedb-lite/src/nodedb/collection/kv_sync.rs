// SPDX-License-Identifier: Apache-2.0

//! Sync recording for the public KV API, and the KV counter.
//!
//! With sync on, a public put or delete records itself in the KV outbound
//! queue before it applies locally. The push loop sends each record to
//! Origin as a `KvPushMsg`. Every public write holds the write-order lock
//! across the record and the apply, so the queue order is the apply order.
//!
//! A full outbound queue refuses the write with `Backpressure` before it
//! changes local state.

use nodedb_physical::kv_atomic::compute;
use nodedb_physical::physical_plan::KvCounterShape;
use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::super::{LockExt, NodeDbLite};
use super::kv::{decode_value, is_expired, kv_key};
use crate::query::kv_ops::writes::atomic_error;
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// KV INCREMENT: add `delta` to the counter at `key` and return the new
    /// value.
    ///
    /// The counter follows the rules the SQL `INCR` follows:
    /// - a raw value is decimal text in and decimal text out;
    /// - a typed row moves its first integer column in key order;
    /// - an absent or expired key starts at 0 and stores decimal text;
    /// - the key keeps its expiry.
    ///
    /// A value that is not a counter, or a sum outside `i64`, is an error
    /// and writes nothing. The new value syncs to Origin like any put.
    pub async fn kv_increment(&self, collection: &str, key: &str, delta: i64) -> NodeDbResult<i64> {
        self.kv_check_pressure()?;
        let _order = self.kv_local.write_order.lock().await;
        let (current, deadline) = self.kv_live_entry(collection, key).await?;
        let (value, written) = compute::incr(current.as_deref(), delta, &KvCounterShape::Raw)
            .map_err(|e| NodeDbError::from(atomic_error(collection, e)))?;
        self.kv_record_stored(collection, key.as_bytes(), &written, deadline)
            .await?;
        self.kv_buffer_put(collection, key, &written, deadline)
            .await?;
        Ok(value)
    }

    /// The live value at `key` and its expiry deadline, reading the write
    /// buffer first. An absent or expired key reads as `(None, 0)`.
    async fn kv_live_entry(
        &self,
        collection: &str,
        key: &str,
    ) -> NodeDbResult<(Option<Vec<u8>>, u64)> {
        let rkey = kv_key(collection, key.as_bytes());
        let buffered: Option<Option<Vec<u8>>> = {
            let buf = self.kv_local.write_buf.lock_or_recover();
            buf.overlay.get(&rkey).cloned()
        };
        let stored = match buffered {
            Some(entry) => entry,
            None => self
                .storage
                .get(Namespace::Kv, &rkey)
                .await
                .map_err(NodeDbError::storage)?,
        };
        let Some(raw) = stored else {
            return Ok((None, 0));
        };
        let (deadline, body) = decode_value(&raw).ok_or_else(|| {
            NodeDbError::storage(format!(
                "corrupt KV entry for key '{key}' in collection '{collection}'"
            ))
        })?;
        if is_expired(deadline) {
            Ok((None, 0))
        } else {
            Ok((Some(body.to_vec()), deadline))
        }
    }

    /// Record a put of the raw value `value` for sync. A no-op with sync
    /// off.
    pub(super) async fn kv_record_put(
        &self,
        collection: &str,
        key: &[u8],
        value: &[u8],
        deadline_ms: u64,
    ) -> NodeDbResult<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(queue) = &self.kv_outbound {
            let op = crate::sync::PendingKvOp::put_raw(key, value, deadline_ms)?;
            queue
                .enqueue(&crate::sync::PendingKvWrite::new(collection, key, op))
                .await?;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (collection, key, value, deadline_ms);
        Ok(())
    }

    /// Record a put of the stored body `body` for sync. A typed row body
    /// is sent as its columns. A no-op with sync off.
    async fn kv_record_stored(
        &self,
        collection: &str,
        key: &[u8],
        body: &[u8],
        deadline_ms: u64,
    ) -> NodeDbResult<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(queue) = &self.kv_outbound {
            let op = crate::sync::PendingKvOp::put_stored(key, body, deadline_ms)?;
            queue
                .enqueue(&crate::sync::PendingKvWrite::new(collection, key, op))
                .await?;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (collection, key, body, deadline_ms);
        Ok(())
    }

    /// Record a delete of `key` for sync. A no-op with sync off.
    pub(super) async fn kv_record_delete(&self, collection: &str, key: &[u8]) -> NodeDbResult<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(queue) = &self.kv_outbound {
            queue
                .enqueue(&crate::sync::PendingKvWrite::new(
                    collection,
                    key,
                    crate::sync::PendingKvOp::Delete,
                ))
                .await?;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (collection, key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nodedb_types::value::Value;

    use crate::config::LiteConfig;
    use crate::storage::pagedb_storage::PagedbStorageMem;
    use crate::sync::{PendingKvOp, PendingKvWrite};

    use super::*;

    async fn open_db(sync_enabled: bool) -> Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.expect("storage");
        let config = LiteConfig {
            sync_enabled,
            ..LiteConfig::default()
        };
        NodeDbLite::open_with_config(storage, config)
            .await
            .expect("open")
    }

    async fn queued(db: &NodeDbLite<PagedbStorageMem>) -> Vec<PendingKvWrite> {
        let queue = db.kv_outbound.as_ref().expect("sync on opens the queue");
        queue
            .drain(usize::MAX)
            .await
            .expect("drain")
            .into_iter()
            .map(|(_, write)| write)
            .collect()
    }

    fn row_value(write: &PendingKvWrite, column: &str) -> Option<Value> {
        match &write.op {
            PendingKvOp::Put { row, .. } => match nodedb_types::value_from_msgpack(row) {
                Ok(Value::Object(map)) => map.get(column).cloned(),
                other => panic!("queued row is not a map: {other:?}"),
            },
            PendingKvOp::Delete => None,
        }
    }

    #[tokio::test]
    async fn a_put_and_a_delete_queue_in_write_order() {
        let db = open_db(true).await;
        db.kv_put("cfg", "k1", b"v1").await.expect("put");
        db.kv_delete("cfg", "k1").await.expect("delete");

        let writes = queued(&db).await;
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].collection, "cfg");
        assert_eq!(writes[0].key, b"k1".to_vec());
        assert_eq!(
            row_value(&writes[0], "value"),
            Some(Value::String("v1".into()))
        );
        assert_eq!(writes[1].op, PendingKvOp::Delete);
        assert_eq!(writes[0].seq, 0, "a seq is assigned at first send");
    }

    #[tokio::test]
    async fn a_put_with_ttl_carries_its_deadline() {
        let db = open_db(true).await;
        db.kv_put_with_ttl("cfg", "k1", b"v1", 60_000)
            .await
            .expect("put");
        let writes = queued(&db).await;
        match &writes[0].op {
            PendingKvOp::Put { expire_at_ms, .. } => assert!(*expire_at_ms > 0),
            PendingKvOp::Delete => panic!("expected a put"),
        }
    }

    #[tokio::test]
    async fn an_increment_stores_decimal_text_and_queues_the_new_value() {
        let db = open_db(true).await;
        assert_eq!(db.kv_increment("ctr", "hits", 5).await.expect("incr"), 5);
        assert_eq!(db.kv_increment("ctr", "hits", -2).await.expect("incr"), 3);
        assert_eq!(
            db.kv_get("ctr", "hits").await.expect("get"),
            Some(b"3".to_vec())
        );

        let writes = queued(&db).await;
        assert_eq!(writes.len(), 2);
        assert_eq!(
            row_value(&writes[1], "value"),
            Some(Value::String("3".into()))
        );
    }

    #[tokio::test]
    async fn an_increment_of_a_non_counter_writes_nothing() {
        let db = open_db(true).await;
        db.kv_put("ctr", "name", b"alice").await.expect("put");
        assert!(db.kv_increment("ctr", "name", 1).await.is_err());
        assert_eq!(queued(&db).await.len(), 1, "only the put is queued");
    }

    #[tokio::test]
    async fn sync_off_queues_nothing() {
        let db = open_db(false).await;
        assert!(db.kv_outbound.is_none());
        db.kv_put("cfg", "k1", b"v1").await.expect("put");
        assert_eq!(db.kv_increment("ctr", "n", 1).await.expect("incr"), 1);
        assert_eq!(
            db.kv_get("cfg", "k1").await.expect("get"),
            Some(b"v1".to_vec())
        );
    }
}
