// SPDX-License-Identifier: Apache-2.0

//! Write a serialized FTS checkpoint to storage and remove what it replaced.

use std::collections::HashSet;

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::format::{
    CHECKPOINT_PREFIX, COLLECTIONS_KEY, DB, LAYOUT_KEY, META_SUBKEYS, PostingsBlob,
    SEGMENT_INDEX_PREFIX, SURROGATES_KEY, TID, postings_prefix,
};
use super::serialize::{FtsFlush, FtsIndexFlush};
use crate::nodedb::flush_gens::{ArtifactFlush, FlushGens};
use crate::storage::engine::{StorageEngine, WriteOp};

// ── Dirty-aware write ──────────────────────────────────────────────────────

/// Whether a stored checkpoint key belongs to one of the `current` indexes.
///
/// An index key may itself contain `:`, so ownership is checked against the
/// key set rather than parsed out of the stored key: `fts:{k}:doclens`,
/// `fts:{k}:meta:{subkey}`, and `fts:{k}:mt:{DB}:{TID}:{k}:{term}`.
fn owned_by_current(key: &[u8], current: &HashSet<String>) -> bool {
    let Ok(key) = std::str::from_utf8(key) else {
        return false;
    };
    let Some(rest) = key.strip_prefix("fts:") else {
        return false;
    };
    if let Some(index_key) = rest.strip_suffix(":doclens")
        && current.contains(index_key)
    {
        return true;
    }
    for &subkey in META_SUBKEYS {
        if let Some(index_key) = rest.strip_suffix(&format!(":meta:{subkey}"))
            && current.contains(index_key)
        {
            return true;
        }
    }
    let scope = format!(":mt:{DB}:{TID}:");
    let mut from = 0;
    while let Some(at) = rest[from..].find(&scope) {
        let index_key = &rest[..from + at];
        let term = &rest[from + at + scope.len()..];
        if current.contains(index_key) && term.starts_with(&format!("{index_key}:")) {
            return true;
        }
        from += at + 1;
    }
    false
}

/// Deletes for every stored checkpoint key no current index owns. The list,
/// layout and surrogate keys, the segment-index sentinels, and every key in
/// `ops` stay.
async fn sweep_unowned_keys<S: StorageEngine>(
    storage: &S,
    ops: &mut Vec<WriteOp>,
    current: &HashSet<String>,
) -> NodeDbResult<()> {
    let touched: HashSet<Vec<u8>> = ops
        .iter()
        .filter_map(|op| match op {
            WriteOp::Put {
                ns: Namespace::Fts,
                key,
                ..
            }
            | WriteOp::Delete {
                ns: Namespace::Fts,
                key,
            } => Some(key.clone()),
            _ => None,
        })
        .collect();
    let existing = storage
        .scan_prefix(Namespace::Fts, CHECKPOINT_PREFIX)
        .await
        .map_err(|e| NodeDbError::storage(format!("fts checkpoint scan: {e}")))?;
    for (key, _) in existing {
        let global = key == COLLECTIONS_KEY || key == LAYOUT_KEY || key == SURROGATES_KEY;
        if !global
            && !key.starts_with(SEGMENT_INDEX_PREFIX)
            && !touched.contains(&key)
            && !owned_by_current(&key, current)
        {
            ops.push(WriteOp::Delete {
                ns: Namespace::Fts,
                key,
            });
        }
    }
    Ok(())
}

/// Write the postings of the index under `index_key` and return the KV ops
/// the batch must carry for them.
///
/// With FTS segments, `Some` writes the blob as the index's segment and
/// `None` deletes the segment. Nothing is added to the batch. Without them
/// the postings live as one KV entry per term: the returned ops put every
/// term of the blob and delete every stored term of this index the blob no
/// longer holds, so a removed term cannot come back on restore.
async fn write_postings<S: StorageEngine>(
    storage: &S,
    index_key: &str,
    postings: Option<&[u8]>,
) -> NodeDbResult<Vec<WriteOp>> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(seg_ext) = storage.as_fts_segment_ext() {
        let result = match postings {
            Some(blob) => seg_ext.write_fts_segment(index_key, blob).await,
            None => seg_ext.delete_fts_segment(index_key).await,
        };
        result
            .map_err(|e| NodeDbError::storage(format!("fts segment write '{index_key}': {e}")))?;
        return Ok(Vec::new());
    }

    let entries: PostingsBlob = match postings {
        Some(blob) => zerompk::from_msgpack::<PostingsBlob>(blob)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?,
        None => Vec::new(),
    };
    let prefix = postings_prefix(index_key);
    let mut ops: Vec<WriteOp> = Vec::with_capacity(entries.len());
    let mut kept: HashSet<Vec<u8>> = HashSet::with_capacity(entries.len());
    for (scoped_term, term_postings) in entries {
        let bytes = zerompk::to_msgpack_vec(&term_postings)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?;
        let key = format!("{prefix}{scoped_term}").into_bytes();
        kept.insert(key.clone());
        ops.push(WriteOp::Put {
            ns: Namespace::Fts,
            key,
            value: bytes,
        });
    }
    // A stored term is this index's only when its scoped term names this
    // index. The prefix also matches the terms of an index whose key extends
    // this one's, e.g. `a:b` and `a:b:mt:c`, and those are left alone.
    let owned_scope = format!("{DB}:{TID}:{index_key}:");
    let stored = storage
        .scan_prefix(Namespace::Fts, prefix.as_bytes())
        .await
        .map_err(|e| NodeDbError::storage(format!("fts postings scan '{index_key}': {e}")))?;
    for (raw_key, _) in stored {
        let owned = raw_key
            .strip_prefix(prefix.as_bytes())
            .is_some_and(|term| term.starts_with(owned_scope.as_bytes()));
        if owned && !kept.contains(&raw_key) {
            ops.push(WriteOp::Delete {
                ns: Namespace::Fts,
                key: raw_key,
            });
        }
    }
    Ok(ops)
}

/// Write the FTS state `plan_fts` planned.
///
/// Each planned index's postings are written first: a pagedb segment when
/// the storage has FTS segments, per-term KV entries otherwise. A failed
/// postings write returns the error before the batch, so nothing is marked
/// and the next flush retries it. When the index list changed, every stored
/// key and segment no current index owns is removed too. The batch is
/// skipped when it is empty.
///
/// Returns the index writes made durable. Records the captured generations,
/// never the current ones: a mutation made since keeps its artifact dirty.
pub(crate) async fn write_planned_fts<S>(
    storage: &S,
    flush: FtsFlush,
    gens: &FlushGens,
) -> NodeDbResult<Vec<ArtifactFlush>>
where
    S: StorageEngine,
{
    let FtsFlush {
        mut ops,
        indexes,
        surrogates,
        catalog,
        current,
    } = flush;

    let mut written: Vec<ArtifactFlush> = Vec::with_capacity(indexes.len());
    for index in indexes {
        let FtsIndexFlush {
            planned,
            ops: index_ops,
            postings,
        } = index;
        let posting_ops = write_postings(storage, planned.key(), postings.as_deref()).await?;
        ops.extend(index_ops);
        ops.extend(posting_ops);
        written.push(planned);
    }

    if catalog.is_some() {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(seg_ext) = storage.as_fts_segment_ext() {
            let handled: HashSet<&str> = written.iter().map(ArtifactFlush::key).collect();
            let existing = seg_ext
                .list_fts_segments("")
                .await
                .map_err(|e| NodeDbError::storage(format!("fts segment list: {e}")))?;
            for index_key in existing {
                if !current.contains(&index_key) && !handled.contains(index_key.as_str()) {
                    seg_ext.delete_fts_segment(&index_key).await.map_err(|e| {
                        NodeDbError::storage(format!("fts segment delete '{index_key}': {e}"))
                    })?;
                }
            }
        }
        sweep_unowned_keys(storage, &mut ops, &current).await?;
    }

    if !ops.is_empty() {
        storage
            .batch_write(&ops)
            .await
            .map_err(|e| NodeDbError::storage(format!("fts checkpoint batch_write: {e}")))?;
    }

    for planned in written.iter().chain(surrogates.iter()) {
        gens.mark_flushed(planned);
        gens.record_write(planned);
    }
    if let Some(bytes) = catalog {
        gens.mark_meta_written(COLLECTIONS_KEY, bytes);
    }
    Ok(written)
}
