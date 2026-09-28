// SPDX-License-Identifier: Apache-2.0

//! Write a serialized FTS checkpoint to storage and remove what it replaced.

use std::collections::HashSet;

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::format::{CHECKPOINT_PREFIX, PostingsBlob, SEGMENT_INDEX_PREFIX, postings_prefix};
use crate::storage::engine::{StorageEngine, WriteOp};

/// Add a delete for every checkpoint key in storage that `ops` does not
/// rewrite. Segment-index sentinels belong to the segment store and stay.
async fn sweep_stale_keys<S: StorageEngine>(
    storage: &S,
    ops: &mut Vec<WriteOp>,
) -> NodeDbResult<()> {
    let written: HashSet<Vec<u8>> = ops
        .iter()
        .filter_map(|op| match op {
            WriteOp::Put {
                ns: Namespace::Fts,
                key,
                ..
            } => Some(key.clone()),
            _ => None,
        })
        .collect();
    let existing = storage
        .scan_prefix(Namespace::Fts, CHECKPOINT_PREFIX)
        .await
        .map_err(|e| NodeDbError::storage(format!("fts checkpoint scan: {e}")))?;
    for (key, _) in existing {
        if !key.starts_with(SEGMENT_INDEX_PREFIX) && !written.contains(&key) {
            ops.push(WriteOp::Delete {
                ns: Namespace::Fts,
                key,
            });
        }
    }
    Ok(())
}

/// Write pre-serialized FTS state to storage.
///
/// `ops` contains B+ tree writes (collections, layout, surrogates, doclens,
/// meta). `segment_writes` contains `(index_key, posting_blob)` pairs that
/// are written via `FtsSegmentExt` when available, or unpacked into per-term
/// KV entries on the fallback path.
///
/// Every checkpoint key and segment this write does not replace is removed,
/// so state from a dropped index or a retracted term does not come back on
/// restore.
///
/// Callers serialize inside the FTS mutex (sync, no I/O) and call this
/// function after releasing the lock to perform async I/O.
pub(crate) async fn write_serialized_fts<S>(
    storage: &S,
    mut ops: Vec<WriteOp>,
    segment_writes: Vec<(String, Vec<u8>)>,
) -> NodeDbResult<()>
where
    S: StorageEngine,
{
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(seg_ext) = storage.as_fts_segment_ext() {
        // pagedb path: write posting blobs as encrypted segments, remove
        // segments no index wrote, then flush the B+ tree batch.
        for (index_key, blob) in &segment_writes {
            seg_ext
                .write_fts_segment(index_key, blob)
                .await
                .map_err(|e| {
                    NodeDbError::storage(format!("fts segment write '{index_key}': {e}"))
                })?;
        }
        let written: HashSet<&str> = segment_writes.iter().map(|(k, _)| k.as_str()).collect();
        let existing = seg_ext
            .list_fts_segments("")
            .await
            .map_err(|e| NodeDbError::storage(format!("fts segment list: {e}")))?;
        for index_key in existing {
            if !written.contains(index_key.as_str()) {
                seg_ext.delete_fts_segment(&index_key).await.map_err(|e| {
                    NodeDbError::storage(format!("fts segment delete '{index_key}': {e}"))
                })?;
            }
        }
        sweep_stale_keys(storage, &mut ops).await?;
        storage
            .batch_write(&ops)
            .await
            .map_err(|e| NodeDbError::storage(format!("fts checkpoint batch_write: {e}")))?;
        return Ok(());
    }

    // KV fallback path (WASM / legacy backends / test doubles): unpack the posting
    // blobs back into per-term KV entries.
    for (index_key, blob) in &segment_writes {
        let entries = zerompk::from_msgpack::<PostingsBlob>(blob)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?;
        let prefix = postings_prefix(index_key);
        for (scoped_term, postings) in entries {
            let bytes = zerompk::to_msgpack_vec(&postings)
                .map_err(|e| NodeDbError::serialization("msgpack", e))?;
            ops.push(WriteOp::Put {
                ns: Namespace::Fts,
                key: format!("{prefix}{scoped_term}").into_bytes(),
                value: bytes,
            });
        }
    }

    sweep_stale_keys(storage, &mut ops).await?;
    storage
        .batch_write(&ops)
        .await
        .map_err(|e| NodeDbError::storage(format!("fts checkpoint batch_write: {e}")))?;
    Ok(())
}
