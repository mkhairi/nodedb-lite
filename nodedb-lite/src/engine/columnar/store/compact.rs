// SPDX-License-Identifier: Apache-2.0

//! Segment compaction and bitemporal tombstone purging.

use super::segments::{
    columnar_err_to_lite, load_segment_bytes, remove_segment_bytes, store_segment_bytes,
};
use super::state::ColumnarEngine;
use crate::error::LiteError;
use crate::runtime::now_millis_i64;
use crate::storage::engine::StorageEngine;
use nodedb_columnar::delete_bitmap::DeleteBitmap;
use nodedb_types::Namespace;
use nodedb_types::columnar::{ColumnarProfile, ColumnarSchema};
use std::collections::{HashMap, HashSet};

impl<S: StorageEngine> ColumnarEngine<S> {
    /// Check if any segments need compaction and run it.
    pub async fn try_compact_collection(&self, collection: &str) -> Result<bool, LiteError> {
        let state_arc = self.lookup(collection)?;

        // Snapshot the data we need, plus capture per-segment delete bitmaps.
        struct Snapshot {
            schema: ColumnarSchema,
            profile_tag: u8,
            to_compact: Vec<u32>,
            bitmaps: HashMap<u32, DeleteBitmap>,
        }

        let snap = {
            let s = Self::lock_state(&state_arc)?;
            let mut to_compact = Vec::new();
            for seg_meta in &s.segments {
                // Skip tombstoned segments — their physical file is already gone.
                if seg_meta.fully_deleted_at_ms.is_some() {
                    continue;
                }
                if let Some(bitmap) = s.mutation.delete_bitmap(seg_meta.segment_id as u64)
                    && bitmap.should_compact(seg_meta.row_count, 0.2)
                {
                    to_compact.push(seg_meta.segment_id);
                }
            }
            if to_compact.is_empty() {
                return Ok(false);
            }
            let schema = s.mutation.schema().clone();
            let profile_tag = match &s.profile {
                ColumnarProfile::Plain => 0,
                ColumnarProfile::Timeseries { .. } => 1,
                ColumnarProfile::Spatial { .. } => 2,
            };
            let mut bitmaps = HashMap::new();
            for &seg_id in &to_compact {
                if let Some(b) = s.mutation.delete_bitmap(seg_id as u64) {
                    bitmaps.insert(seg_id, b.clone());
                }
            }
            Snapshot {
                schema,
                profile_tag,
                to_compact,
                bitmaps,
            }
        };

        for seg_id in &snap.to_compact {
            let seg_bytes = match load_segment_bytes(&*self.storage, collection, *seg_id).await? {
                Some(b) => b,
                None => continue,
            };

            let empty_bitmap = DeleteBitmap::new();
            let bitmap = snap.bitmaps.get(seg_id).unwrap_or(&empty_bitmap);

            let result = nodedb_columnar::compaction::compact_segment(
                &seg_bytes,
                bitmap,
                &snap.schema,
                snap.profile_tag,
                &self.memory,
                None,
            )
            .map_err(columnar_err_to_lite)?;

            if let Some(new_seg_bytes) = result.segment {
                store_segment_bytes(&*self.storage, collection, *seg_id, &new_seg_bytes).await?;

                // Update row count under the inner lock (scoped so the guard
                // never crosses the await below — clippy is strict).
                {
                    let mut s = Self::lock_state(&state_arc)?;
                    if let Some(meta) = s.segments.iter_mut().find(|m| m.segment_id == *seg_id) {
                        meta.row_count = result.live_rows as u64;
                    }
                }

                let del_key = format!("{collection}:del:{seg_id}");
                self.storage
                    .delete(Namespace::Columnar, del_key.as_bytes())
                    .await?;
            } else {
                // All rows deleted. For bitemporal collections, tombstone the
                // segment meta (retain the entry with fully_deleted_at_ms set)
                // so `purge_bitemporal_before` can physically remove it later.
                // For non-bitemporal collections, remove immediately.
                let is_bitemporal = {
                    let s = Self::lock_state(&state_arc)?;
                    s.bitemporal
                };

                remove_segment_bytes(&*self.storage, collection, *seg_id).await?;
                let del_key = format!("{collection}:del:{seg_id}");
                self.storage
                    .delete(Namespace::Columnar, del_key.as_bytes())
                    .await?;

                {
                    let mut s = Self::lock_state(&state_arc)?;
                    if is_bitemporal {
                        // Mark as fully deleted instead of removing from the list.
                        if let Some(meta) = s.segments.iter_mut().find(|m| m.segment_id == *seg_id)
                        {
                            meta.row_count = 0;
                            meta.fully_deleted_at_ms = Some(now_millis_i64());
                        }
                    } else {
                        s.segments.retain(|m| m.segment_id != *seg_id);
                    }
                }
            }
        }

        // Persist updated metadata.
        let meta_bytes = {
            let s = Self::lock_state(&state_arc)?;
            zerompk::to_msgpack_vec(&s.segments).map_err(|e| LiteError::Serialization {
                detail: e.to_string(),
            })?
        };
        let meta_key = format!("{collection}:meta");
        self.storage
            .put(Namespace::Columnar, meta_key.as_bytes(), &meta_bytes)
            .await?;

        Ok(true)
    }

    /// Purge fully-deleted segment tombstones for a bitemporal collection where
    /// `fully_deleted_at_ms < cutoff_ms`. Non-bitemporal collections always
    /// return `rows_affected: 0` — they have no tombstones.
    ///
    /// Returns the number of tombstoned segment entries removed.
    pub async fn purge_bitemporal_before(
        &self,
        collection: &str,
        cutoff_ms: i64,
    ) -> Result<u64, LiteError> {
        let state_arc = self.lookup(collection)?;

        let (is_bitemporal, to_purge): (bool, Vec<u32>) = {
            let s = Self::lock_state(&state_arc)?;
            let purge: Vec<u32> = s
                .segments
                .iter()
                .filter(|m| m.fully_deleted_at_ms.is_some_and(|t| t < cutoff_ms))
                .map(|m| m.segment_id)
                .collect();
            (s.bitemporal, purge)
        };

        if !is_bitemporal {
            return Ok(0);
        }

        if to_purge.is_empty() {
            return Ok(0);
        }

        let purged_ids: HashSet<u32> = to_purge.iter().copied().collect();

        // Remove purged segment IDs from the in-memory list.
        {
            let mut s = Self::lock_state(&state_arc)?;
            s.segments.retain(|m| !purged_ids.contains(&m.segment_id));
        }

        // Persist the updated segment metadata list.
        let meta_bytes = {
            let s = Self::lock_state(&state_arc)?;
            zerompk::to_msgpack_vec(&s.segments).map_err(|e| LiteError::Serialization {
                detail: e.to_string(),
            })?
        };
        let meta_key = format!("{collection}:meta");
        self.storage
            .put(Namespace::Columnar, meta_key.as_bytes(), &meta_bytes)
            .await?;

        Ok(to_purge.len() as u64)
    }
}
