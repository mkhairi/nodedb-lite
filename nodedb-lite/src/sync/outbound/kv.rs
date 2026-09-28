//! KV write outbound queue for Lite sync.
//!
//! Every KV write that reaches this replica's KV store is recorded here as
//! the row every KV read returns, `{key, value…}`, or as a delete. The push
//! loop sends each entry to Origin as a `KvPushMsg`, and Origin answers with
//! a `KvPushAckMsg`.
//!
//! # Durability and order
//!
//! One [`DurableOutboundQueue`] over [`Namespace::KvPushPending`] holds puts
//! and deletes together, so the queue order is the write order, and a delete
//! is never sent ahead of the put it follows. A KV write is `async`, so it
//! enqueues straight into durable storage. There is no staging buffer.
//!
//! # Identity
//!
//! An entry's batch ID is its durable key read as a big-endian `u64`. The
//! key is assigned once and survives a restart, so the ack of a push sent
//! before a restart still names the entry it acknowledges.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nodedb_types::Namespace;
use nodedb_types::sync::wire::KvPushOp;
use nodedb_types::value::Value;
use tokio::sync::Mutex;

use super::durable_queue::DurableOutboundQueue;
use crate::error::LiteError;
use crate::query::kv_ops::body::decode_kv_map;
use crate::storage::engine::StorageEngine;

/// The write a [`PendingKvWrite`] carries.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub enum PendingKvOp {
    /// Store `row`, the `{key, value…}` row as standard MessagePack.
    /// `expire_at_ms` is the absolute expiry, `0` for none.
    Put { row: Vec<u8>, expire_at_ms: u64 },
    /// Remove the key.
    Delete,
}

impl PendingKvOp {
    /// A put of the raw value `value` at `key`.
    ///
    /// The row is `{key, value}`. A UTF-8 value is a string, and any other
    /// value is binary, so the bytes reach Origin unchanged.
    pub fn put_raw(key: &[u8], value: &[u8], expire_at_ms: u64) -> Result<Self, LiteError> {
        let value = match std::str::from_utf8(value) {
            Ok(text) => Value::String(text.to_string()),
            Err(_) => Value::Bytes(value.to_vec()),
        };
        let mut row = HashMap::with_capacity(2);
        row.insert("value".to_string(), value);
        Self::put_row(key, row, expire_at_ms)
    }

    /// A put of the stored body `body` at `key`.
    ///
    /// A map body is a typed row, sent as its columns with the key added. A
    /// raw body is a raw value. A map-shaped body that does not decode is a
    /// `Serialization` error.
    pub fn put_stored(key: &[u8], body: &[u8], expire_at_ms: u64) -> Result<Self, LiteError> {
        match decode_kv_map(body)? {
            Some(columns) => Self::put_row(key, columns, expire_at_ms),
            None => Self::put_raw(key, body, expire_at_ms),
        }
    }

    fn put_row(
        key: &[u8],
        mut row: HashMap<String, Value>,
        expire_at_ms: u64,
    ) -> Result<Self, LiteError> {
        row.entry("key".to_string())
            .or_insert_with(|| Value::String(String::from_utf8_lossy(key).into_owned()));
        let row = nodedb_types::value_to_msgpack(&Value::Object(row)).map_err(|e| {
            LiteError::Serialization {
                detail: format!("kv outbound row encode: {e}"),
            }
        })?;
        Ok(Self::Put { row, expire_at_ms })
    }

    /// The wire form of this write.
    pub fn to_wire(&self) -> KvPushOp {
        match self {
            Self::Put { row, expire_at_ms } => KvPushOp::Put {
                row: row.clone(),
                expire_at_ms: *expire_at_ms,
            },
            Self::Delete => KvPushOp::Delete,
        }
    }
}

/// One KV write awaiting sync to Origin.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    zerompk::ToMessagePack,
    zerompk::FromMessagePack,
)]
pub struct PendingKvWrite {
    /// Target KV collection.
    pub collection: String,
    /// The entry's key bytes.
    pub key: Vec<u8>,
    /// The write.
    pub op: PendingKvOp,
    /// Stable idempotent-producer seq. `0` until the first send assigns one.
    /// The assigned seq is persisted, so a re-send after a reconnect reuses
    /// it and Origin deduplicates.
    pub seq: u64,
}

impl PendingKvWrite {
    /// A write with no seq assigned yet.
    pub fn new(collection: &str, key: &[u8], op: PendingKvOp) -> Self {
        Self {
            collection: collection.to_string(),
            key: key.to_vec(),
            op,
            seq: 0,
        }
    }
}

/// Durable outbound queue for KV write sync.
pub struct KvOutbound<S: StorageEngine> {
    durable: DurableOutboundQueue<S>,
    /// Batch IDs sent to Origin and not yet acknowledged.
    in_flight: Mutex<HashSet<u64>>,
}

impl<S: StorageEngine> KvOutbound<S> {
    /// Open the queue over [`Namespace::KvPushPending`].
    pub async fn open(storage: Arc<S>) -> Result<Self, LiteError> {
        Self::open_with_cap(storage, DurableOutboundQueue::<S>::DEFAULT_CAP).await
    }

    /// Open with a custom cap.
    pub async fn open_with_cap(storage: Arc<S>, cap: usize) -> Result<Self, LiteError> {
        let durable =
            DurableOutboundQueue::open_with_cap(storage, Namespace::KvPushPending, cap).await?;
        Ok(Self {
            durable,
            in_flight: Mutex::new(HashSet::new()),
        })
    }

    /// Durably enqueue `write`.
    ///
    /// Returns [`LiteError::Backpressure`] when the queue is at its cap.
    pub async fn enqueue(&self, write: &PendingKvWrite) -> Result<(), LiteError> {
        let payload = encode(write)?;
        self.durable.enqueue(&payload).await
    }

    /// Check that `writes` more writes fit under the cap.
    ///
    /// Returns [`LiteError::Backpressure`] when they do not.
    pub async fn ensure_room(&self, writes: usize) -> Result<(), LiteError> {
        self.durable.ensure_room(writes).await
    }

    /// Up to `limit` pending writes in FIFO order, skipping the ones in
    /// flight. Returns `(durable_key, write)` pairs.
    pub async fn drain(&self, limit: usize) -> Result<Vec<(Vec<u8>, PendingKvWrite)>, LiteError> {
        let in_flight = self.in_flight.lock().await;
        let pairs = self.durable.drain_batch(limit).await?;
        let mut out = Vec::with_capacity(pairs.len());
        for (key, payload) in pairs {
            if in_flight.contains(&kv_batch_id(&key)?) {
                continue;
            }
            let write: PendingKvWrite =
                zerompk::from_msgpack(&payload).map_err(|e| LiteError::Serialization {
                    detail: format!("kv outbound decode: {e}"),
                })?;
            out.push((key, write));
        }
        Ok(out)
    }

    /// Record that the entry with `batch_id` was sent and awaits its ack.
    pub async fn mark_in_flight(&self, batch_id: u64) {
        self.in_flight.lock().await.insert(batch_id);
    }

    /// Retire the entry with `batch_id`: forget it is in flight and delete
    /// its durable record.
    pub async fn retire(&self, batch_id: u64) -> Result<(), LiteError> {
        self.in_flight.lock().await.remove(&batch_id);
        self.durable
            .ack_keys(&[batch_id.to_be_bytes().to_vec()])
            .await
    }

    /// Forget every in-flight entry so the next drain re-sends them.
    pub async fn clear_in_flight(&self) {
        self.in_flight.lock().await.clear();
    }

    /// Persist `write` under `durable_key`, carrying its assigned seq.
    pub async fn update_entry(
        &self,
        durable_key: &[u8],
        write: &PendingKvWrite,
    ) -> Result<(), LiteError> {
        let payload = encode(write)?;
        self.durable.update_entry(durable_key, &payload).await
    }

    /// Number of pending entries in durable storage.
    pub async fn pending_count(&self) -> Result<u64, LiteError> {
        self.durable.len().await
    }
}

/// The batch ID of the KV outbound entry stored under `durable_key`.
pub fn kv_batch_id(durable_key: &[u8]) -> Result<u64, LiteError> {
    let bytes: [u8; 8] = durable_key
        .try_into()
        .map_err(|_| LiteError::Serialization {
            detail: format!(
                "kv outbound: durable key is {} bytes, not an 8-byte ID",
                durable_key.len()
            ),
        })?;
    Ok(u64::from_be_bytes(bytes))
}

fn encode(write: &PendingKvWrite) -> Result<Vec<u8>, LiteError> {
    zerompk::to_msgpack_vec(write).map_err(|e| LiteError::Serialization {
        detail: format!("kv outbound encode: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    async fn open_queue(storage: Arc<PagedbStorageMem>) -> KvOutbound<PagedbStorageMem> {
        KvOutbound::open(storage).await.expect("open kv outbound")
    }

    fn put(key: &str) -> PendingKvWrite {
        PendingKvWrite::new(
            "cfg",
            key.as_bytes(),
            PendingKvOp::Put {
                row: vec![0x80],
                expire_at_ms: 0,
            },
        )
    }

    #[tokio::test]
    async fn writes_drain_in_the_order_they_were_enqueued() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.expect("storage"));
        let q = open_queue(storage).await;
        q.enqueue(&put("a")).await.expect("enqueue");
        q.enqueue(&PendingKvWrite::new("cfg", b"a", PendingKvOp::Delete))
            .await
            .expect("enqueue");

        let drained = q.drain(usize::MAX).await.expect("drain");
        assert_eq!(drained.len(), 2);
        assert!(matches!(drained[0].1.op, PendingKvOp::Put { .. }));
        assert_eq!(drained[1].1.op, PendingKvOp::Delete);
    }

    #[tokio::test]
    async fn an_in_flight_write_is_not_drained_again_until_cleared() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.expect("storage"));
        let q = open_queue(storage).await;
        q.enqueue(&put("a")).await.expect("enqueue");
        let (key, _) = q.drain(usize::MAX).await.expect("drain").remove(0);
        let batch_id = kv_batch_id(&key).expect("batch id");

        q.mark_in_flight(batch_id).await;
        assert!(q.drain(usize::MAX).await.expect("drain").is_empty());

        q.clear_in_flight().await;
        assert_eq!(q.drain(usize::MAX).await.expect("drain").len(), 1);
    }

    #[tokio::test]
    async fn an_ack_retires_the_write() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.expect("storage"));
        let q = open_queue(storage).await;
        q.enqueue(&put("a")).await.expect("enqueue");
        q.enqueue(&put("b")).await.expect("enqueue");
        let (key, _) = q.drain(usize::MAX).await.expect("drain").remove(0);
        let batch_id = kv_batch_id(&key).expect("batch id");

        q.mark_in_flight(batch_id).await;
        q.retire(batch_id).await.expect("retire");

        assert_eq!(q.pending_count().await.expect("count"), 1);
        let left = q.drain(usize::MAX).await.expect("drain");
        assert_eq!(left[0].1.key, b"b".to_vec());
    }

    #[tokio::test]
    async fn an_assigned_seq_survives_a_reopen_under_the_same_batch_id() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.expect("storage"));
        let (key, batch_id) = {
            let q = open_queue(Arc::clone(&storage)).await;
            q.enqueue(&put("a")).await.expect("enqueue");
            let (key, mut write) = q.drain(usize::MAX).await.expect("drain").remove(0);
            write.seq = 7;
            q.update_entry(&key, &write).await.expect("persist seq");
            let batch_id = kv_batch_id(&key).expect("batch id");
            (key, batch_id)
        };

        let q = open_queue(storage).await;
        let drained = q.drain(usize::MAX).await.expect("drain");
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, key);
        assert_eq!(drained[0].1.seq, 7);
        assert_eq!(kv_batch_id(&drained[0].0).expect("batch id"), batch_id);
    }

    fn row_of(op: &PendingKvOp) -> Value {
        match op {
            PendingKvOp::Put { row, .. } => nodedb_types::value_from_msgpack(row).expect("row"),
            PendingKvOp::Delete => panic!("expected a put"),
        }
    }

    #[test]
    fn a_raw_value_becomes_the_key_value_row() {
        let op = PendingKvOp::put_raw(b"k1", b"v1", 0).expect("row");
        let row = row_of(&op);
        assert_eq!(row.get("key"), Some(&Value::String("k1".into())));
        assert_eq!(row.get("value"), Some(&Value::String("v1".into())));
        let expected = nodedb_query::msgpack_scan::kv_row_msgpack("k1", b"v1");
        assert_eq!(
            nodedb_types::value_from_msgpack(&expected).expect("row"),
            row
        );
    }

    #[test]
    fn a_binary_value_crosses_as_bytes() {
        let op = PendingKvOp::put_raw(b"k1", &[0xff, 0xfe], 0).expect("row");
        assert_eq!(
            row_of(&op).get("value"),
            Some(&Value::Bytes(vec![0xff, 0xfe]))
        );
    }

    #[test]
    fn a_typed_body_becomes_its_column_row_with_the_key() {
        let mut columns = HashMap::new();
        columns.insert("n".to_string(), Value::Integer(8));
        let body = nodedb_types::value_to_msgpack(&Value::Object(columns)).expect("body");
        let op = PendingKvOp::put_stored(b"k3", &body, 0).expect("row");
        let row = row_of(&op);
        assert_eq!(row.get("n"), Some(&Value::Integer(8)));
        assert_eq!(row.get("key"), Some(&Value::String("k3".into())));
    }

    #[test]
    fn a_raw_stored_body_becomes_the_value_row() {
        let op = PendingKvOp::put_stored(b"k4", b"12", 0).expect("row");
        assert_eq!(row_of(&op).get("value"), Some(&Value::String("12".into())));
    }

    #[tokio::test]
    async fn room_is_checked_before_a_batch_of_writes() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.expect("storage"));
        let q = KvOutbound::open_with_cap(storage, 2)
            .await
            .expect("open kv outbound");
        q.enqueue(&put("a")).await.expect("enqueue");
        q.ensure_room(1).await.expect("one more fits");
        assert!(matches!(
            q.ensure_room(2).await,
            Err(LiteError::Backpressure { .. })
        ));
    }

    #[tokio::test]
    async fn a_full_queue_refuses_with_backpressure() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.expect("storage"));
        let q = KvOutbound::open_with_cap(storage, 1)
            .await
            .expect("open kv outbound");
        q.enqueue(&put("a")).await.expect("enqueue");
        assert!(matches!(
            q.enqueue(&put("b")).await,
            Err(LiteError::Backpressure { .. })
        ));
    }
}
