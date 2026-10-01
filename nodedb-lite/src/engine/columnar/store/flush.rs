// SPDX-License-Identifier: Apache-2.0

//! Memtable segment writes and collection flushing.

use super::segments::{columnar_err_to_lite, store_segment_bytes};
use super::state::{ColumnarEngine, SegmentMeta};
use crate::error::LiteError;
use crate::runtime::now_millis_i64;
use crate::storage::engine::StorageEngine;
use nodedb_columnar::writer::SegmentWriter;
use nodedb_types::Namespace;
use nodedb_types::columnar::ColumnarProfile;

impl<S: StorageEngine> ColumnarEngine<S> {
    /// Flush the memtable for a collection to a new segment.
    pub async fn flush_collection(&self, collection: &str) -> Result<(), LiteError> {
        let state_arc = self.lookup(collection)?;

        // Drain memtable + collect everything we need under the inner lock.
        struct FlushPayload {
            segment_id: u32,
            segment_bytes: Vec<u8>,
            meta_key: String,
            meta_bytes: Vec<u8>,
            del_ops: Vec<(String, Vec<u8>)>,
        }

        let payload = {
            let mut s = Self::lock_state(&state_arc)?;
            if s.mutation.memtable().is_empty() {
                return Ok(());
            }

            let segment_id = s.next_segment_id;
            s.next_segment_id += 1;

            let (schema, columns, row_count) = s.mutation.memtable_mut().drain_optimized();

            let profile_tag = match &s.profile {
                ColumnarProfile::Plain => 0,
                ColumnarProfile::Timeseries { .. } => 1,
                ColumnarProfile::Spatial { .. } => 2,
            };

            let writer = SegmentWriter::new(profile_tag, self.memory.clone());
            let segment_bytes = writer
                .write_segment(&schema, &columns, row_count, None)
                .map_err(columnar_err_to_lite)?;

            let system_time_from_ms = if s.bitemporal { now_millis_i64() } else { 0 };
            s.segments.push(SegmentMeta {
                segment_id,
                row_count: row_count as u64,
                system_time_from_ms,
                fully_deleted_at_ms: None,
            });
            let meta_key = format!("{collection}:meta");
            let meta_bytes =
                zerompk::to_msgpack_vec(&s.segments).map_err(|e| LiteError::Serialization {
                    detail: e.to_string(),
                })?;

            s.mutation
                .on_memtable_flushed(segment_id as u64)
                .map_err(|e| LiteError::Storage {
                    detail: format!("on_memtable_flushed: {e}"),
                })?;

            let mut del_ops: Vec<(String, Vec<u8>)> = Vec::new();
            for (&seg_id, bitmap) in s.mutation.delete_bitmaps() {
                if !bitmap.is_empty() {
                    let del_key = format!("{collection}:del:{seg_id}");
                    let del_bytes = bitmap.to_bytes().map_err(columnar_err_to_lite)?;
                    del_ops.push((del_key, del_bytes));
                }
            }

            FlushPayload {
                segment_id,
                segment_bytes,
                meta_key,
                meta_bytes,
                del_ops,
            }
        };

        // Storage I/O with lock dropped.
        // Large segment bytes go through the segment ext (or KV fallback).
        store_segment_bytes(
            &*self.storage,
            collection,
            payload.segment_id,
            &payload.segment_bytes,
        )
        .await?;

        // Small B+ tree entries: segment metadata list and delete bitmaps.
        self.storage
            .put(
                Namespace::Columnar,
                payload.meta_key.as_bytes(),
                &payload.meta_bytes,
            )
            .await?;
        for (del_key, del_bytes) in &payload.del_ops {
            self.storage
                .put(Namespace::Columnar, del_key.as_bytes(), del_bytes)
                .await?;
        }

        Ok(())
    }

    /// Flush all collections' memtables.
    pub async fn flush_all(&self) -> Result<(), LiteError> {
        let names: Vec<String> = self
            .collections
            .read()
            .map_err(|_| LiteError::LockPoisoned)?
            .keys()
            .cloned()
            .collect();
        for name in names {
            self.flush_collection(&name).await?;
        }
        Ok(())
    }
}
