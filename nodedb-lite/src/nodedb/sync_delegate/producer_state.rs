// SPDX-License-Identifier: Apache-2.0

//! Free functions behind the producer-state and collection-meta
//! `SyncDelegate` methods.
//!
//! Called from the thin delegation methods in `delegate_impl.rs`.

use crate::nodedb::collection::CollectionMeta;
use crate::nodedb::core::NodeDbLite;
use crate::storage::engine::StorageEngine;

/// Durable storage key for the Origin-assigned producer ID.
const META_SYNC_PRODUCER_ID: &[u8] = b"sync.producer_id";

/// Durable storage key for the Origin-echoed accepted epoch.
const META_SYNC_ACCEPTED_EPOCH: &[u8] = b"sync.accepted_epoch";

pub(super) async fn persist_producer_state_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
    producer_id: u64,
    accepted_epoch: u64,
) {
    let ns = nodedb_types::Namespace::Meta;
    if let Err(e) = db
        .storage
        .put(ns, META_SYNC_PRODUCER_ID, &producer_id.to_be_bytes())
        .await
    {
        tracing::warn!(error = %e, "SyncDelegate: persist_producer_state: producer_id write failed");
    }
    if let Err(e) = db
        .storage
        .put(ns, META_SYNC_ACCEPTED_EPOCH, &accepted_epoch.to_be_bytes())
        .await
    {
        tracing::warn!(error = %e, "SyncDelegate: persist_producer_state: accepted_epoch write failed");
    }
}

pub(super) async fn load_producer_state_impl<S: StorageEngine>(db: &NodeDbLite<S>) -> (u64, u64) {
    let ns = nodedb_types::Namespace::Meta;
    let producer_id = match db.storage.get(ns, META_SYNC_PRODUCER_ID).await {
        Ok(Some(bytes)) if bytes.len() == 8 => {
            u64::from_be_bytes(bytes.try_into().unwrap_or([0; 8]))
        }
        _ => 0,
    };
    let accepted_epoch = match db.storage.get(ns, META_SYNC_ACCEPTED_EPOCH).await {
        Ok(Some(bytes)) if bytes.len() == 8 => {
            u64::from_be_bytes(bytes.try_into().unwrap_or([0; 8]))
        }
        _ => 0,
    };
    (producer_id, accepted_epoch)
}

pub(super) async fn get_collection_meta_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
    name: &str,
) -> Option<CollectionMeta> {
    let key = format!("collection:{name}");
    match db
        .storage
        .get(nodedb_types::Namespace::Meta, key.as_bytes())
        .await
    {
        Ok(Some(bytes)) => match sonic_rs::from_slice(&bytes) {
            Ok(meta) => Some(meta),
            Err(e) => {
                tracing::warn!(collection = name, error = %e, "get_collection_meta: decode failed");
                None
            }
        },
        Ok(None) => db.implicit_collection_meta(name),
        Err(e) => {
            tracing::warn!(collection = name, error = %e, "get_collection_meta: storage read failed");
            None
        }
    }
}
