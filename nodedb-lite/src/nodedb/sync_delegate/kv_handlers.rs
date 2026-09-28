// SPDX-License-Identifier: Apache-2.0

//! Free functions behind the KV `SyncDelegate` methods.
//!
//! Called from the thin delegation methods in `delegate_impl.rs`. With sync
//! off there is no KV outbound queue: nothing is pending and every call is a
//! no-op.

use crate::error::LiteError;
use crate::nodedb::core::NodeDbLite;
use crate::storage::engine::StorageEngine;
use crate::sync::outbound::kv::PendingKvWrite;

pub(super) async fn pending_kv_writes_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
) -> Result<Vec<(Vec<u8>, PendingKvWrite)>, LiteError> {
    match &db.kv_outbound {
        Some(q) => q.drain(crate::sync::PUSH_DRAIN_LIMIT).await,
        None => Ok(Vec::new()),
    }
}

pub(super) async fn persist_kv_write_seq_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
    key: &[u8],
    write: &PendingKvWrite,
) -> Result<(), LiteError> {
    match &db.kv_outbound {
        Some(q) => q.update_entry(key, write).await,
        None => Ok(()),
    }
}

pub(super) async fn mark_kv_write_in_flight_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
    batch_id: u64,
) {
    if let Some(q) = &db.kv_outbound {
        q.mark_in_flight(batch_id).await;
    }
}

pub(super) async fn retire_kv_write_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
    batch_id: u64,
) -> Result<(), LiteError> {
    match &db.kv_outbound {
        Some(q) => q.retire(batch_id).await,
        None => Ok(()),
    }
}
