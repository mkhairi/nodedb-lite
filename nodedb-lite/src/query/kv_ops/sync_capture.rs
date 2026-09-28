// SPDX-License-Identifier: Apache-2.0
//! Records SQL KV writes for the sync push to Origin.
//!
//! A SQL KV write runs through [`record_kv_write`]:
//! 1. It takes the write-order lock the public KV API shares, so writes
//!    enqueue in the order they applied. It commits the public API's
//!    buffered writes, so none of them lands after the SQL write.
//! 2. It checks the outbound queue has room for one record per touched key.
//!    A full queue refuses the write before it changes local state.
//! 3. It runs the write.
//! 4. When the write changed a row, it reads each touched key back from
//!    storage and enqueues that post-image. A live entry is a put, and an
//!    absent or expired entry is a delete.
//!
//! One post-image rule covers every write shape. Counters, CAS, field
//! writes and on-conflict merges all enqueue the row they left behind.
//! `TRUNCATE` runs through [`truncate_recorded`], which enqueues a delete
//! for every key the collection held.
//!
//! With sync off there is no outbound queue, and only the lock is taken.

use std::collections::BTreeSet;
use std::future::Future;

use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::reads::{decode_value, is_expired, kv_key, split_kv_key};
use super::writes::kv_truncate;

/// One key a SQL KV write touches: `(collection, key)`.
pub(crate) type KvTouched = (String, Vec<u8>);

/// Run the SQL KV `write` that touches `touched`, and record the post-image
/// of each touched key for sync when the write changed a row.
pub(crate) async fn record_kv_write<S, F>(
    engine: &LiteQueryEngine<S>,
    touched: Vec<KvTouched>,
    write: F,
) -> Result<QueryResult, LiteError>
where
    S: StorageEngine,
    F: Future<Output = Result<QueryResult, LiteError>>,
{
    let _order = engine.kv_local.write_order.lock().await;
    engine.kv_local.flush_to(&*engine.storage).await?;
    ensure_room(engine, touched.len()).await?;
    let result = write.await?;
    if result.rows_affected > 0 {
        for (collection, key) in &touched {
            enqueue_post_image(engine, collection, key).await?;
        }
    }
    Ok(result)
}

/// `TRUNCATE` a KV collection and record a delete for every key it held.
///
/// The public API's buffered writes are committed first, so every key the
/// collection held is in storage when the keys are collected.
pub(crate) async fn truncate_recorded<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    let _order = engine.kv_local.write_order.lock().await;
    engine.kv_local.flush_to(&*engine.storage).await?;
    let keys = if has_outbound(engine) {
        held_keys(engine, collection).await?
    } else {
        BTreeSet::new()
    };
    ensure_room(engine, keys.len()).await?;
    let result = kv_truncate(engine, collection).await?;
    for key in &keys {
        enqueue_delete(engine, collection, key).await?;
    }
    Ok(result)
}

/// Every key `collection` holds in storage.
async fn held_keys<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<BTreeSet<Vec<u8>>, LiteError> {
    let prefix = kv_key(collection, b"");
    let mut keys = BTreeSet::new();
    let entries = engine
        .storage
        .scan_range_bounded(Namespace::Kv, Some(&prefix), None, None)
        .await?;
    for (composite, _) in &entries {
        let Some((coll, key)) = split_kv_key(composite) else {
            continue;
        };
        if coll != collection {
            break;
        }
        keys.insert(key.to_vec());
    }
    Ok(keys)
}

#[cfg(not(target_arch = "wasm32"))]
fn has_outbound<S: StorageEngine>(engine: &LiteQueryEngine<S>) -> bool {
    engine.kv_outbound.is_some()
}

#[cfg(target_arch = "wasm32")]
fn has_outbound<S: StorageEngine>(_engine: &LiteQueryEngine<S>) -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
async fn ensure_room<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    writes: usize,
) -> Result<(), LiteError> {
    match &engine.kv_outbound {
        Some(queue) => queue.ensure_room(writes).await,
        None => Ok(()),
    }
}

#[cfg(target_arch = "wasm32")]
async fn ensure_room<S: StorageEngine>(
    _engine: &LiteQueryEngine<S>,
    _writes: usize,
) -> Result<(), LiteError> {
    Ok(())
}

/// Enqueue the entry `key` holds in storage now.
#[cfg(not(target_arch = "wasm32"))]
async fn enqueue_post_image<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
) -> Result<(), LiteError> {
    use crate::sync::{PendingKvOp, PendingKvWrite};

    let Some(queue) = &engine.kv_outbound else {
        return Ok(());
    };
    let stored = engine
        .storage
        .get(Namespace::Kv, &kv_key(collection, key))
        .await?;
    let op = match stored {
        None => PendingKvOp::Delete,
        Some(raw) => {
            let (deadline, body) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
                detail: format!(
                    "corrupt KV entry for key '{}' in collection '{collection}'",
                    String::from_utf8_lossy(key)
                ),
            })?;
            if is_expired(deadline) {
                PendingKvOp::Delete
            } else {
                PendingKvOp::put_stored(key, body, deadline)?
            }
        }
    };
    queue
        .enqueue(&PendingKvWrite::new(collection, key, op))
        .await
}

#[cfg(target_arch = "wasm32")]
async fn enqueue_post_image<S: StorageEngine>(
    _engine: &LiteQueryEngine<S>,
    _collection: &str,
    _key: &[u8],
) -> Result<(), LiteError> {
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn enqueue_delete<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
) -> Result<(), LiteError> {
    use crate::sync::{PendingKvOp, PendingKvWrite};

    match &engine.kv_outbound {
        Some(queue) => {
            queue
                .enqueue(&PendingKvWrite::new(collection, key, PendingKvOp::Delete))
                .await
        }
        None => Ok(()),
    }
}

#[cfg(target_arch = "wasm32")]
async fn enqueue_delete<S: StorageEngine>(
    _engine: &LiteQueryEngine<S>,
    _collection: &str,
    _key: &[u8],
) -> Result<(), LiteError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nodedb_physical::physical_plan::KvCounterShape;
    use nodedb_types::value::Value;

    use super::*;
    use crate::PagedbStorageMem;
    use crate::nodedb::LockExt;
    use crate::query::engine::test_engine;
    use crate::query::kv_ops::writes::{kv_delete, kv_incr, kv_insert_if_absent, kv_put};
    use crate::storage::engine::WriteOp;
    use crate::sync::{KvOutbound, PendingKvOp, PendingKvWrite};

    async fn synced_engine(cap: usize) -> LiteQueryEngine<PagedbStorageMem> {
        let mut engine = test_engine().await;
        let queue = KvOutbound::open_with_cap(Arc::clone(&engine.storage), cap)
            .await
            .expect("open kv outbound");
        engine.set_kv_outbound(Arc::new(queue));
        engine
    }

    async fn queued(engine: &LiteQueryEngine<PagedbStorageMem>) -> Vec<PendingKvWrite> {
        let queue = engine.kv_outbound.as_ref().expect("queue");
        queue
            .drain(usize::MAX)
            .await
            .expect("drain")
            .into_iter()
            .map(|(_, write)| write)
            .collect()
    }

    fn column(write: &PendingKvWrite, name: &str) -> Option<Value> {
        match &write.op {
            PendingKvOp::Put { row, .. } => match nodedb_types::value_from_msgpack(row) {
                Ok(Value::Object(map)) => map.get(name).cloned(),
                other => panic!("queued row is not a map: {other:?}"),
            },
            PendingKvOp::Delete => None,
        }
    }

    fn touched(key: &[u8]) -> Vec<KvTouched> {
        vec![("kv".to_string(), key.to_vec())]
    }

    #[tokio::test]
    async fn a_sql_put_queues_its_row() {
        let engine = synced_engine(16).await;
        record_kv_write(
            &engine,
            touched(b"k1"),
            kv_put(&engine, "kv", b"k1", b"v1", 0),
        )
        .await
        .expect("put");

        let writes = queued(&engine).await;
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].key, b"k1".to_vec());
        assert_eq!(
            column(&writes[0], "value"),
            Some(Value::String("v1".into()))
        );
    }

    #[tokio::test]
    async fn a_write_that_changes_nothing_queues_nothing() {
        let engine = synced_engine(16).await;
        kv_put(&engine, "kv", b"k1", b"v1", 0).await.expect("seed");
        let result = record_kv_write(
            &engine,
            touched(b"k1"),
            kv_insert_if_absent(&engine, "kv", b"k1", b"v2", 0),
        )
        .await
        .expect("insert if absent");
        assert_eq!(result.rows_affected, 0);
        assert!(queued(&engine).await.is_empty());
    }

    #[tokio::test]
    async fn a_counter_queues_the_value_it_left() {
        let engine = synced_engine(16).await;
        for _ in 0..2 {
            record_kv_write(
                &engine,
                touched(b"n"),
                kv_incr(&engine, "kv", b"n", 3, 0, &KvCounterShape::Raw),
            )
            .await
            .expect("incr");
        }
        let writes = queued(&engine).await;
        assert_eq!(writes.len(), 2);
        assert_eq!(column(&writes[1], "value"), Some(Value::String("6".into())));
    }

    #[tokio::test]
    async fn a_sql_delete_queues_a_delete() {
        let engine = synced_engine(16).await;
        kv_put(&engine, "kv", b"k1", b"v1", 0).await.expect("seed");
        let keys = vec![b"k1".to_vec()];
        record_kv_write(&engine, touched(b"k1"), kv_delete(&engine, "kv", &keys))
            .await
            .expect("delete");
        let writes = queued(&engine).await;
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].op, PendingKvOp::Delete);
    }

    #[tokio::test]
    async fn a_full_queue_refuses_the_write_before_it_applies() {
        let engine = synced_engine(1).await;
        record_kv_write(&engine, touched(b"a"), kv_put(&engine, "kv", b"a", b"1", 0))
            .await
            .expect("first put");
        let refused =
            record_kv_write(&engine, touched(b"b"), kv_put(&engine, "kv", b"b", b"2", 0)).await;
        assert!(matches!(refused, Err(LiteError::Backpressure { .. })));
        let stored = engine
            .storage
            .get(Namespace::Kv, &kv_key("kv", b"b"))
            .await
            .expect("get");
        assert_eq!(stored, None, "the refused write left no local row");
    }

    #[tokio::test]
    async fn truncate_queues_a_delete_for_stored_and_buffered_keys() {
        // `b` sits in the public API's write buffer, not in storage yet.
        let engine = synced_engine(16).await;
        kv_put(&engine, "kv", b"a", b"1", 0).await.expect("seed");
        kv_put(&engine, "other", b"z", b"9", 0).await.expect("seed");
        {
            let mut buf = engine.kv_local.write_buf.lock_or_recover();
            let rkey = kv_key("kv", b"b");
            buf.overlay.insert(rkey.clone(), Some(vec![0; 8]));
            buf.ops.push(WriteOp::Put {
                ns: Namespace::Kv,
                key: rkey,
                value: vec![0; 8],
            });
        }

        truncate_recorded(&engine, "kv").await.expect("truncate");

        let writes = queued(&engine).await;
        let deleted: Vec<Vec<u8>> = writes
            .iter()
            .map(|w| {
                assert_eq!(w.collection, "kv");
                assert_eq!(w.op, PendingKvOp::Delete);
                w.key.clone()
            })
            .collect();
        assert_eq!(deleted, vec![b"a".to_vec(), b"b".to_vec()]);
    }
}
