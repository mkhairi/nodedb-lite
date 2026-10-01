// SPDX-License-Identifier: Apache-2.0

//! FTS-specific staging owns every entry until durable persistence completes.

use super::{PendingFtsDelete, PendingFtsIndex, staging::Staging};
use crate::{
    error::LiteError, storage::engine::StorageEngine,
    sync::outbound::durable_queue::DurableOutboundQueue,
};
use nodedb_types::Namespace;
use std::{
    collections::HashMap,
    sync::{Arc, atomic::AtomicU64},
};
use tokio::sync::Mutex;

/// Unflushed staging remains volatile. Durable queues preserve each kind's FIFO order.
pub struct FtsOutbound<S: StorageEngine> {
    pub(super) staging_indexes: Staging<PendingFtsIndex>,
    pub(super) staging_deletes: Staging<PendingFtsDelete>,
    pub(super) durable_indexes: DurableOutboundQueue<S>,
    pub(super) durable_deletes: DurableOutboundQueue<S>,
    pub(super) ids: AtomicU64,
    pub(super) flush_mutex: Mutex<()>,
    pub(super) in_flight_indexes: Mutex<HashMap<u64, Vec<u8>>>,
    pub(super) in_flight_deletes_map: Mutex<HashMap<u64, Vec<u8>>>,
}

impl<S: StorageEngine> FtsOutbound<S> {
    pub async fn open(storage: Arc<S>) -> Result<Self, LiteError> {
        Self::open_with_cap(storage, DurableOutboundQueue::<S>::DEFAULT_CAP).await
    }

    pub async fn open_with_cap(storage: Arc<S>, cap: usize) -> Result<Self, LiteError> {
        let durable_indexes = DurableOutboundQueue::open_with_cap(
            Arc::clone(&storage),
            Namespace::FtsIndexPending,
            cap,
        )
        .await?;
        let durable_deletes =
            DurableOutboundQueue::open_with_cap(storage, Namespace::FtsDeletePending, cap).await?;
        Ok(Self {
            staging_indexes: Staging::new(),
            staging_deletes: Staging::new(),
            durable_indexes,
            durable_deletes,
            ids: AtomicU64::new(1),
            flush_mutex: Mutex::new(()),
            in_flight_indexes: Mutex::new(HashMap::new()),
            in_flight_deletes_map: Mutex::new(HashMap::new()),
        })
    }
}
