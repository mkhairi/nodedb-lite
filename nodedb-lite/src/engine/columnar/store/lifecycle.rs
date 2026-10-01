// SPDX-License-Identifier: Apache-2.0

//! Collection restoration and engine lock access.

use super::decode::rebuild_pk_from_column;
use super::segments::load_segment_bytes;
use super::state::{CollectionMap, CollectionState, ColumnarEngine, SegmentMeta};
use crate::error::LiteError;
use crate::storage::engine::StorageEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::sync::outbound::columnar::ColumnarOutbound;
#[cfg(not(target_arch = "wasm32"))]
use crate::sync::outbound::timeseries::TimeseriesOutbound;
use nodedb_columnar::delete_bitmap::DeleteBitmap;
use nodedb_columnar::mutation::MutationEngine;
use nodedb_columnar::reader::SegmentReader;
use nodedb_mem::ScopedMemory;
use nodedb_types::Namespace;
use nodedb_types::columnar::{ColumnarProfile, ColumnarSchema};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

pub(super) const META_COLUMNAR_SCHEMA_PREFIX: &str = "columnar_schema:";
pub(super) const META_COLUMNAR_COLLECTIONS: &[u8] = b"meta:columnar_collections";

impl<S: StorageEngine> ColumnarEngine<S> {
    /// Create a new empty columnar engine.
    pub fn new(storage: Arc<S>, memory: ScopedMemory) -> Self {
        Self {
            storage,
            collections: RwLock::new(HashMap::new()),
            #[cfg(not(target_arch = "wasm32"))]
            outbound: None,
            #[cfg(not(target_arch = "wasm32"))]
            timeseries_outbound: None,
            memory,
        }
    }

    /// Attach a sync outbound queue for plain columnar collections.
    ///
    /// Must be called before any inserts if columnar sync is desired.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_outbound(&mut self, outbound: Arc<ColumnarOutbound<S>>) {
        self.outbound = Some(outbound);
    }

    /// Attach a sync outbound queue for timeseries-profile collections.
    ///
    /// When set, inserts into collections with `ColumnarProfile::Timeseries`
    /// are routed here instead of `outbound`, so the transport can send them
    /// as `TimeseriesPush` frames to Origin's timeseries engine.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_timeseries_outbound(&mut self, outbound: Arc<TimeseriesOutbound<S>>) {
        self.timeseries_outbound = Some(outbound);
    }

    /// Restore columnar collections from storage on startup.
    pub async fn restore(storage: Arc<S>, memory: ScopedMemory) -> Result<Self, LiteError> {
        let engine = Self::new(Arc::clone(&storage), memory);

        let list_bytes = storage
            .get(Namespace::Meta, META_COLUMNAR_COLLECTIONS)
            .await?;
        let names: Vec<String> = match list_bytes {
            Some(bytes) => zerompk::from_msgpack(&bytes).map_err(|e| LiteError::Storage {
                detail: format!("columnar collection list deserialization: {e}"),
            })?,
            None => Vec::new(),
        };

        let mut loaded: CollectionMap = HashMap::new();
        for name in names {
            let meta_key = format!("{META_COLUMNAR_SCHEMA_PREFIX}{name}");
            #[derive(serde::Deserialize, zerompk::ToMessagePack, zerompk::FromMessagePack)]
            struct StoredSchema {
                schema: ColumnarSchema,
                profile: ColumnarProfile,
                #[serde(default)]
                bitemporal: bool,
            }
            if let Some(schema_bytes) = storage.get(Namespace::Meta, meta_key.as_bytes()).await?
                && let Ok(stored) = zerompk::from_msgpack::<StoredSchema>(&schema_bytes)
            {
                let seg_meta_key = format!("{name}:meta");
                let segments: Vec<SegmentMeta> = storage
                    .get(Namespace::Columnar, seg_meta_key.as_bytes())
                    .await?
                    .and_then(|b| zerompk::from_msgpack(&b).ok())
                    .unwrap_or_default();

                let next_id = segments.iter().map(|s| s.segment_id + 1).max().unwrap_or(1);

                let mut mutation = MutationEngine::new(name.clone(), stored.schema.clone());

                for seg_meta in &segments {
                    // Skip fully-deleted segments — they have no physical segment file.
                    if seg_meta.fully_deleted_at_ms.is_some() {
                        continue;
                    }

                    if let Some(seg_bytes) =
                        load_segment_bytes(&*storage, &name, seg_meta.segment_id).await?
                        && let Ok(reader) = SegmentReader::open(&seg_bytes)
                        && let Ok(pk_col) = reader.read_column(0)
                    {
                        rebuild_pk_from_column(&mut mutation, &pk_col, seg_meta.segment_id);
                    }

                    let del_key = format!("{name}:del:{}", seg_meta.segment_id);
                    if let Some(del_bytes) =
                        storage.get(Namespace::Columnar, del_key.as_bytes()).await?
                        && let Ok(bitmap) = DeleteBitmap::from_bytes(&del_bytes)
                    {
                        for row_idx in bitmap.iter() {
                            let _ = row_idx;
                        }
                    }
                }

                loaded.insert(
                    name,
                    Arc::new(Mutex::new(CollectionState {
                        mutation,
                        profile: stored.profile,
                        bitemporal: stored.bitemporal,
                        segments,
                        next_segment_id: next_id,
                    })),
                );
            }
        }

        *engine
            .collections
            .write()
            .map_err(|_| LiteError::LockPoisoned)? = loaded;

        // outbound is wired after restore by the caller (NodeDbLite::open_inner).
        Ok(engine)
    }

    // -- Internal helpers --

    pub(in crate::engine::columnar) fn lookup(
        &self,
        name: &str,
    ) -> Result<Arc<Mutex<CollectionState>>, LiteError> {
        let guard = self
            .collections
            .read()
            .map_err(|_| LiteError::LockPoisoned)?;
        guard
            .get(name)
            .cloned()
            .ok_or_else(|| LiteError::collection_not_found("columnar", name))
    }

    pub(in crate::engine::columnar) fn lock_state<'a>(
        state: &'a Arc<Mutex<CollectionState>>,
    ) -> Result<std::sync::MutexGuard<'a, CollectionState>, LiteError> {
        state.lock().map_err(|_| LiteError::LockPoisoned)
    }
}
