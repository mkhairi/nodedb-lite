// SPDX-License-Identifier: Apache-2.0

//! Serialize [`FtsCollectionManager`](crate::engine::fts::FtsCollectionManager)
//! state into write ops, without I/O.

use std::collections::{HashMap, HashSet};

use nodedb_fts::backend::FtsBackend;
use nodedb_fts::backend::memory::MemoryBackend;
use nodedb_fts::{FtsIndex, IndexScope};
use nodedb_types::Namespace;
use nodedb_types::Surrogate;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::format::{
    COLLECTIONS_KEY, DB, FtsSurrogateState, LAYOUT_KEY, LAYOUT_PER_FIELD, META_SUBKEYS,
    PostingsBlob, SURROGATES_KEY, TID, compact_to_ser, doclens_key, meta_key, scoped_term_prefix,
};
use crate::nodedb::flush_gens::{ArtifactFlush, FTS_SURROGATES_KEY, FlushArtifact, FlushGens};
use crate::storage::engine::WriteOp;

fn put(key: Vec<u8>, value: Vec<u8>) -> WriteOp {
    WriteOp::Put {
        ns: Namespace::Fts,
        key,
        value,
    }
}

fn to_msgpack<T: zerompk::ToMessagePack>(value: &T) -> NodeDbResult<Vec<u8>> {
    zerompk::to_msgpack_vec(value).map_err(|e| NodeDbError::serialization("msgpack", e))
}

/// Serialize all term postings of `idx` into one msgpack blob. Each stored
/// term carries the [`scoped_term_prefix`] of `index_key`.
///
/// Returns `None` if the memtable has no terms (empty index — nothing to write).
fn serialize_postings_blob(
    index_key: &str,
    idx: &FtsIndex<MemoryBackend>,
) -> NodeDbResult<Option<Vec<u8>>> {
    let mt = idx.memtable();
    let scope = IndexScope::document(index_key);
    let prefix = scoped_term_prefix(index_key);
    let mut entries: PostingsBlob = Vec::new();

    for term in mt.terms(DB, TID, scope) {
        let postings = mt.get_postings(DB, TID, scope, &term);
        if postings.is_empty() {
            continue;
        }
        entries.push((
            format!("{prefix}{term}"),
            postings.iter().map(compact_to_ser).collect(),
        ));
    }

    if entries.is_empty() {
        return Ok(None);
    }
    to_msgpack(&entries).map(Some)
}

/// Collect KV `WriteOp`s for doc-lengths and meta blobs (always on B+ tree).
fn metadata_ops_for_index(
    index_key: &str,
    idx: &FtsIndex<MemoryBackend>,
    ops: &mut Vec<WriteOp>,
) -> NodeDbResult<()> {
    let mt = idx.memtable();
    let scope = IndexScope::document(index_key);

    // ── Doc lengths (per-doc lengths needed by BM25 scoring) ─────────────────
    let mut surrogates: Vec<u32> = mt
        .terms(DB, TID, scope)
        .iter()
        .flat_map(|t| {
            mt.get_postings(DB, TID, scope, t)
                .into_iter()
                .map(|p| p.doc_id.0)
        })
        .collect();
    surrogates.sort_unstable();
    surrogates.dedup();

    let mut doclens: Vec<(u32, u32)> = Vec::with_capacity(surrogates.len());
    for &s in &surrogates {
        if let Some(len) = idx
            .backend()
            .read_doc_length(DB, TID, scope, Surrogate(s))
            .map_err(|e| NodeDbError::storage(format!("fts doc_len: {e}")))?
        {
            doclens.push((s, len));
        }
    }
    if !doclens.is_empty() {
        ops.push(put(
            doclens_key(index_key).into_bytes(),
            to_msgpack(&doclens)?,
        ));
    }

    // ── Meta blobs (fieldnorms, analyzer, language) ───────────────────────────
    for &subkey in META_SUBKEYS {
        if let Some(data) = idx
            .backend()
            .read_meta(DB, TID, scope, subkey)
            .map_err(|e| NodeDbError::storage(format!("fts meta read: {e}")))?
        {
            ops.push(put(meta_key(index_key, subkey).into_bytes(), data));
        }
    }

    Ok(())
}

/// `(index_key, blob)` posting segments of one checkpoint.
type SegmentWrites = Vec<(String, Vec<u8>)>;

/// Serialize FTS state into write ops (no I/O, safe to call while holding a
/// mutex guard). Returns `(kv_ops, segment_writes)`. The caller writes
/// `segment_writes` through `FtsSegmentExt::write_fts_segment` when the
/// backend has one.
///
/// A flush writes through [`plan_fts`]; this full form serves tests.
#[cfg(test)]
pub(crate) fn serialize_fts(
    indices: &HashMap<String, FtsIndex<MemoryBackend>>,
    id_to_surrogate: &HashMap<String, u32>,
    next_surrogate: u32,
) -> NodeDbResult<(Vec<WriteOp>, SegmentWrites)> {
    let mut ops: Vec<WriteOp> = Vec::new();
    let mut segment_writes: SegmentWrites = Vec::new();

    // ── Collection list, layout version, surrogate maps ──────────────────────
    let index_keys: Vec<String> = indices.keys().cloned().collect();
    ops.push(put(COLLECTIONS_KEY.to_vec(), to_msgpack(&index_keys)?));
    ops.push(put(LAYOUT_KEY.to_vec(), vec![LAYOUT_PER_FIELD]));
    let surrogate_state = FtsSurrogateState {
        id_to_surrogate: id_to_surrogate
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
        next_surrogate,
    };
    ops.push(put(SURROGATES_KEY.to_vec(), to_msgpack(&surrogate_state)?));

    // ── Per-index data ────────────────────────────────────────────────────────
    for (key, idx) in indices {
        // Always collect metadata ops (doc-lengths, meta blobs) onto B+ tree.
        metadata_ops_for_index(key, idx, &mut ops)?;

        // Collect posting data: dispatched to pagedb segments or unpacked
        // into per-term KV entries at write time. Indices with no terms
        // produce no segment write.
        if let Some(blob) = serialize_postings_blob(key, idx)? {
            segment_writes.push((key.clone(), blob));
        }
    }

    Ok((ops, segment_writes))
}

// ── Dirty-aware planning ───────────────────────────────────────────────────

/// One index write a flush plans.
pub(crate) struct FtsIndexFlush {
    /// Mark flushed once this index's postings write and the batch both
    /// succeed.
    pub(super) planned: ArtifactFlush,
    /// Doc-length and meta ops, or the deletes of a dropped index's.
    pub(super) ops: Vec<WriteOp>,
    /// The posting blob, possibly of zero terms. `None` deletes the stored
    /// postings of a dropped index.
    pub(super) postings: Option<Vec<u8>>,
}

/// The FTS writes one flush plans.
pub(crate) struct FtsFlush {
    /// Puts for the index list, layout and surrogate map, each when planned.
    pub(super) ops: Vec<WriteOp>,
    /// Dirty and dropped indexes, each in index key order.
    pub(super) indexes: Vec<FtsIndexFlush>,
    /// The surrogate map put in `ops`, if any.
    pub(super) surrogates: Option<ArtifactFlush>,
    /// The index list put in `ops`, if any. Recorded as written once `ops`
    /// commits. A changed list also sweeps stored keys and segments no
    /// current index owns.
    pub(super) catalog: Option<Vec<u8>>,
    /// Every current index key, for that sweep.
    pub(super) current: HashSet<String>,
}

/// The encoded index list: the index keys, sorted, so the bytes are a
/// function of the set of keys and compare equal across ticks while the set
/// is unchanged.
fn catalog_bytes(indices: &HashMap<String, FtsIndex<MemoryBackend>>) -> NodeDbResult<Vec<u8>> {
    let mut index_keys: Vec<&String> = indices.keys().collect();
    index_keys.sort();
    to_msgpack(&index_keys)
}

/// Whether the index list differs from the one this handle last wrote.
pub(crate) fn catalog_changed(
    indices: &HashMap<String, FtsIndex<MemoryBackend>>,
    gens: &FlushGens,
) -> bool {
    catalog_bytes(indices).map_or(true, |bytes| gens.meta_changed(COLLECTIONS_KEY, &bytes))
}

/// Doc-length and meta ops for a dirty index.
///
/// The doc-length list is put even when empty, so it replaces a stored list
/// that still names removed documents. A meta blob the index does not hold is
/// deleted, so a stale one from an earlier index under the same key does not
/// come back on restore.
fn index_ops(index_key: &str, idx: &FtsIndex<MemoryBackend>) -> NodeDbResult<Vec<WriteOp>> {
    let mut ops: Vec<WriteOp> = Vec::new();
    metadata_ops_for_index(index_key, idx, &mut ops)?;
    let doclens = doclens_key(index_key).into_bytes();
    if !ops
        .iter()
        .any(|op| matches!(op, WriteOp::Put { key, .. } if *key == doclens))
    {
        let empty: Vec<(u32, u32)> = Vec::new();
        ops.push(put(doclens, to_msgpack(&empty)?));
    }
    for &subkey in META_SUBKEYS {
        let key = meta_key(index_key, subkey).into_bytes();
        if !ops
            .iter()
            .any(|op| matches!(op, WriteOp::Put { key: k, .. } if *k == key))
        {
            ops.push(WriteOp::Delete {
                ns: Namespace::Fts,
                key,
            });
        }
    }
    Ok(ops)
}

/// The deletes of a dropped index's doc lengths and meta blobs.
fn removal_ops(index_key: &str) -> Vec<WriteOp> {
    let mut ops = Vec::with_capacity(1 + META_SUBKEYS.len());
    ops.push(WriteOp::Delete {
        ns: Namespace::Fts,
        key: doclens_key(index_key).into_bytes(),
    });
    for &subkey in META_SUBKEYS {
        ops.push(WriteOp::Delete {
            ns: Namespace::Fts,
            key: meta_key(index_key, subkey).into_bytes(),
        });
    }
    ops
}

/// Plan the FTS state a flush must write (no I/O, safe to call while
/// holding the manager lock).
///
/// An index is written only when it is dirty or `full` is set, and so is the
/// surrogate map. Each generation is captured by `FlushGens::plan`, so call
/// this under the manager lock the state is read under. The index list and
/// layout byte are written only when the list differs from the one this
/// handle last wrote, or `full` is set. An index in `dropped` that is not in
/// `indices` and is dirty has its stored form deleted.
pub(crate) fn plan_fts(
    indices: &HashMap<String, FtsIndex<MemoryBackend>>,
    dropped: &HashSet<String>,
    id_to_surrogate: &HashMap<String, u32>,
    next_surrogate: u32,
    full: bool,
    gens: &FlushGens,
) -> NodeDbResult<FtsFlush> {
    let mut ops: Vec<WriteOp> = Vec::new();

    let keys_bytes = catalog_bytes(indices)?;
    let catalog = if full || gens.meta_changed(COLLECTIONS_KEY, &keys_bytes) {
        ops.push(put(COLLECTIONS_KEY.to_vec(), keys_bytes.clone()));
        ops.push(put(LAYOUT_KEY.to_vec(), vec![LAYOUT_PER_FIELD]));
        Some(keys_bytes)
    } else {
        None
    };

    let surrogates = gens.plan(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY, full);
    if surrogates.is_some() {
        let surrogate_state = FtsSurrogateState {
            id_to_surrogate: id_to_surrogate
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            next_surrogate,
        };
        ops.push(put(SURROGATES_KEY.to_vec(), to_msgpack(&surrogate_state)?));
    }

    let mut index_keys: Vec<&String> = indices.keys().collect();
    index_keys.sort();
    let mut indexes: Vec<FtsIndexFlush> = Vec::new();
    for index_key in index_keys {
        let Some(idx) = indices.get(index_key) else {
            continue;
        };
        let Some(planned) = gens.plan(FlushArtifact::FtsIndex, index_key, full) else {
            continue;
        };
        // An index with no terms still writes an empty blob, so restore
        // reads zero postings rather than the ones stored before its last
        // document was removed.
        let postings = match serialize_postings_blob(index_key, idx)? {
            Some(blob) => blob,
            None => to_msgpack(&PostingsBlob::new())?,
        };
        indexes.push(FtsIndexFlush {
            planned,
            ops: index_ops(index_key, idx)?,
            postings: Some(postings),
        });
    }

    let mut dropped_keys: Vec<&String> = dropped
        .iter()
        .filter(|key| !indices.contains_key(*key))
        .collect();
    dropped_keys.sort();
    for index_key in dropped_keys {
        let Some(planned) = gens.plan(FlushArtifact::FtsIndex, index_key, full) else {
            continue;
        };
        indexes.push(FtsIndexFlush {
            planned,
            ops: removal_ops(index_key),
            postings: None,
        });
    }

    Ok(FtsFlush {
        ops,
        indexes,
        surrogates,
        catalog,
        current: indices.keys().cloned().collect(),
    })
}
