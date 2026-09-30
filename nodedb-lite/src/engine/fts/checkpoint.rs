//! Checkpoint serialization and restoration for [`FtsCollectionManager`].
//!
//! Persists the in-memory FTS state so that a cold open can load the index
//! without re-tokenizing source documents. A flush writes only the indexes
//! that changed since this handle last wrote them, the surrogate map only
//! when a surrogate was allocated, and the index list only when the set of
//! indexes changed.
//!
//! A changed index is written whole. An index with no postings writes an
//! empty segment and an empty doc-length list, so restore reads zero postings
//! rather than the ones stored before its last document was removed. An index
//! `drop_collection` removed has its segment, doc lengths, and meta blobs
//! deleted.
//!
//! ## Storage layout
//!
//! ### B+ tree (`Namespace::Fts`) — always used
//!
//! | Key                             | Value                                      |
//! |---------------------------------|--------------------------------------------|
//! | `fts:_collections`              | MessagePack `Vec<String>` — sorted index key list |
//! | `fts:_surrogates`               | MessagePack `FtsSurrogateState`            |
//! | `fts:{index_key}:doclens`       | MessagePack `Vec<(u32,u32)>` — surrogate/len |
//! | `fts:{index_key}:meta:{subkey}` | raw bytes (fieldnorms/analyzer/language)   |
//!
//! ### pagedb segments — used when `as_fts_segment_ext()` returns `Some`
//!
//! | Segment name          | Value                                            |
//! |-----------------------|--------------------------------------------------|
//! | `fts/seg/{index_key}` | MessagePack `Vec<(String, Vec<SerPosting>)>`     |
//!
//! When pagedb segments are unavailable (WASM / legacy backends), posting data
//! falls back to the legacy KV path:
//!
//! | Key                               | Value                              |
//! |-----------------------------------|------------------------------------|
//! | `fts:{index_key}:mt:{scoped_term}`| MessagePack `Vec<SerPosting>`      |
//! | `fts:{index_key}:mtstat`          | MessagePack `(u32, u64)` (unused)  |
//!
//! ## Rationale: memtable vs segment storage
//!
//! `nodedb-fts` on Lite uses `MemoryBackend` exclusively.  All postings live in
//! a `Memtable`; the backend's LSM segment layer is unused — and that is now
//! *enforced* by [`super::LITE_MEMTABLE_CONFIG`] rather than assumed. It was
//! only ever an assumption, and it silently stopped holding at 100k unique
//! terms, when the memtable spilled into segments this serializer does not
//! read (NDB-AQL-37). Serializing the memtable alone is correct only while
//! nothing can drain it.  The pagedb segment
//! path bundles all per-term posting entries for one index key into a single
//! segment blob, reducing B+ tree pressure from O(vocab_size) entries to O(1)
//! per index key.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nodedb_fts::FtsIndex;
use nodedb_fts::backend::FtsBackend;
use nodedb_fts::backend::memory::MemoryBackend;
use nodedb_fts::block::CompactPosting;
use nodedb_mem::MemoryGovernor;
use nodedb_types::Namespace;
use nodedb_types::Surrogate;
use nodedb_types::error::{NodeDbError, NodeDbResult};
use serde::{Deserialize, Serialize};

use crate::nodedb::flush_gens::{ArtifactFlush, FTS_SURROGATES_KEY, FlushArtifact, FlushGens};
use crate::storage::engine::{StorageEngine, WriteOp};

/// Catalog key: the sorted index key list.
const COLLECTIONS_KEY: &[u8] = b"fts:_collections";

/// Surrogate map key.
const SURROGATES_KEY: &[u8] = b"fts:_surrogates";

/// Surrogate maps persisted alongside posting data.
#[derive(Serialize, Deserialize, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub(super) struct FtsSurrogateState {
    /// `doc_id` string → dense u32 surrogate.
    pub id_to_surrogate: Vec<(String, u32)>,
    /// Next surrogate to assign.
    pub next_surrogate: u32,
}

/// Known meta subkeys written by `nodedb-fts`.
const META_SUBKEYS: &[&str] = &["fieldnorms", "analyzer", "language"];

/// A single memtable posting entry serialized as a flat tuple.
///
/// Matches `CompactPosting` fields: `(doc_id, term_freq, fieldnorm, positions)`.
type SerPosting = (u32, u32, u8, Vec<u32>);

fn compact_to_ser(p: &CompactPosting) -> SerPosting {
    (p.doc_id.0, p.term_freq, p.fieldnorm, p.positions.clone())
}

fn ser_to_compact(s: SerPosting) -> CompactPosting {
    CompactPosting {
        doc_id: Surrogate(s.0),
        term_freq: s.1,
        fieldnorm: s.2,
        positions: s.3,
    }
}

/// Serialize all term postings for `index_key` into a single msgpack blob.
///
/// An index with no terms serializes to an empty list. It is still written:
/// the index is listed in `fts:_collections`, so restore reads its stored
/// postings, and an empty blob replaces the ones stored before its last
/// document was removed.
fn serialize_postings_blob(
    _index_key: &str,
    idx: &FtsIndex<MemoryBackend>,
) -> NodeDbResult<Vec<u8>> {
    let mt = idx.memtable();
    let mut entries: Vec<(String, Vec<SerPosting>)> = Vec::new();

    for scoped_term in mt.terms() {
        let postings = mt.get_postings(&scoped_term);
        if postings.is_empty() {
            continue;
        }
        let ser: Vec<SerPosting> = postings.iter().map(compact_to_ser).collect();
        entries.push((scoped_term, ser));
    }

    zerompk::to_msgpack_vec(&entries).map_err(|e| NodeDbError::serialization("msgpack", e))
}

/// Collect KV `WriteOp`s for doc-lengths and meta blobs (always on B+ tree).
///
/// The doc-length list is put even when empty, so it replaces a stored list
/// that still names removed documents. A meta blob the index does not hold is
/// deleted, so a stale one from an earlier index under the same key does not
/// come back on restore.
fn metadata_ops_for_index(
    index_key: &str,
    idx: &FtsIndex<MemoryBackend>,
    ops: &mut Vec<WriteOp>,
) -> NodeDbResult<()> {
    const TID: u64 = 0;
    const DB: u64 = 0;
    let mt = idx.memtable();

    // ── Doc lengths (per-doc lengths needed by BM25 scoring) ─────────────────
    let mut surrogates: Vec<u32> = mt
        .terms()
        .iter()
        .flat_map(|t| mt.get_postings(t).into_iter().map(|p| p.doc_id.0))
        .collect();
    surrogates.sort_unstable();
    surrogates.dedup();

    let mut doclens: Vec<(u32, u32)> = Vec::with_capacity(surrogates.len());
    for &s in &surrogates {
        if let Some(len) = idx
            .backend()
            .read_doc_length(DB, TID, index_key, Surrogate(s))
            .map_err(|e| NodeDbError::storage(format!("fts doc_len: {e}")))?
        {
            doclens.push((s, len));
        }
    }
    let doclens_key = format!("fts:{index_key}:doclens");
    let bytes =
        zerompk::to_msgpack_vec(&doclens).map_err(|e| NodeDbError::serialization("msgpack", e))?;
    ops.push(WriteOp::Put {
        ns: Namespace::Fts,
        key: doclens_key.into_bytes(),
        value: bytes,
    });

    // ── Meta blobs (fieldnorms, analyzer, language) ───────────────────────────
    for &subkey in META_SUBKEYS {
        let meta_key = format!("fts:{index_key}:meta:{subkey}").into_bytes();
        match idx
            .backend()
            .read_meta(DB, TID, index_key, subkey)
            .map_err(|e| NodeDbError::storage(format!("fts meta read: {e}")))?
        {
            Some(data) => ops.push(WriteOp::Put {
                ns: Namespace::Fts,
                key: meta_key,
                value: data,
            }),
            None => ops.push(WriteOp::Delete {
                ns: Namespace::Fts,
                key: meta_key,
            }),
        }
    }

    Ok(())
}

/// KV `WriteOp`s that delete the doc lengths and meta blobs of an index
/// `drop_collection` removed.
fn removal_ops(index_key: &str) -> Vec<WriteOp> {
    let mut ops = Vec::with_capacity(1 + META_SUBKEYS.len());
    ops.push(WriteOp::Delete {
        ns: Namespace::Fts,
        key: format!("fts:{index_key}:doclens").into_bytes(),
    });
    for &subkey in META_SUBKEYS {
        ops.push(WriteOp::Delete {
            ns: Namespace::Fts,
            key: format!("fts:{index_key}:meta:{subkey}").into_bytes(),
        });
    }
    ops
}

/// One index write a flush plans.
pub(crate) struct FtsIndexFlush {
    /// Mark flushed once this index's postings write and the batch both
    /// succeed.
    planned: ArtifactFlush,
    /// Doc-length and meta ops. The batch carries them only when the
    /// postings write succeeded.
    ops: Vec<WriteOp>,
    /// The posting blob. `None` deletes the stored postings of a dropped
    /// index.
    postings: Option<Vec<u8>>,
}

/// The FTS writes one flush plans.
pub(crate) struct FtsFlush {
    /// Puts for the index list and the surrogate map, each when planned.
    ops: Vec<WriteOp>,
    /// Dirty and dropped indexes, each in index key order.
    indexes: Vec<FtsIndexFlush>,
    /// The surrogate map put in `ops`, if any.
    surrogates: Option<ArtifactFlush>,
    /// The index list put in `ops`, if any. Record it as written once `ops`
    /// commits.
    meta: Option<(&'static [u8], Vec<u8>)>,
}

/// FTS state restored from storage.
pub(crate) struct RestoredFts {
    pub(crate) indices: HashMap<String, FtsIndex<MemoryBackend>>,
    pub(crate) id_to_surrogate: HashMap<String, u32>,
    pub(crate) surrogate_to_id: HashMap<u32, String>,
    pub(crate) next_surrogate: u32,
    /// Index keys whose postings, doc lengths, and meta blobs all decoded
    /// from the form a flush on this storage writes. Only these match their
    /// stored form.
    pub(crate) decoded: HashSet<String>,
    /// Whether the surrogate map decoded.
    pub(crate) surrogates_decoded: bool,
    /// The stored index list, recorded as written so an unchanged list is not
    /// written again.
    pub(crate) stored_catalog: Option<(&'static [u8], Vec<u8>)>,
}

impl RestoredFts {
    /// No usable checkpoint.
    fn empty() -> Self {
        Self {
            indices: HashMap::new(),
            id_to_surrogate: HashMap::new(),
            surrogate_to_id: HashMap::new(),
            next_surrogate: 0,
            decoded: HashSet::new(),
            surrogates_decoded: false,
            stored_catalog: None,
        }
    }
}

/// Serialize the FTS state a flush must write (no I/O, safe to call while
/// holding a mutex guard).
///
/// An index is written only when it is dirty or `full` is set, and so is the
/// surrogate map. Each generation is captured by `FlushGens::plan`, so call
/// this under the manager lock the state is read under. The index list is
/// written only when it differs from the value this handle last wrote, or
/// `full` is set. An index in `dropped` that is not in `indices` and is dirty
/// has its stored form deleted.
pub(crate) fn serialize_fts(
    indices: &HashMap<String, FtsIndex<MemoryBackend>>,
    dropped: &HashSet<String>,
    id_to_surrogate: &HashMap<String, u32>,
    next_surrogate: u32,
    full: bool,
    gens: &FlushGens,
) -> NodeDbResult<FtsFlush> {
    let mut ops: Vec<WriteOp> = Vec::new();
    let mut meta = None;

    // ── Collection list ───────────────────────────────────────────────────────
    // Sorted so the encoded list is a function of the set of keys, not of the
    // map's iteration order, and compares equal across ticks while the set is
    // unchanged.
    let mut index_keys: Vec<String> = indices.keys().cloned().collect();
    index_keys.sort();
    let keys_bytes = zerompk::to_msgpack_vec(&index_keys)
        .map_err(|e| NodeDbError::serialization("msgpack", e))?;
    if full || gens.meta_changed(COLLECTIONS_KEY, &keys_bytes) {
        ops.push(WriteOp::Put {
            ns: Namespace::Fts,
            key: COLLECTIONS_KEY.to_vec(),
            value: keys_bytes.clone(),
        });
        meta = Some((COLLECTIONS_KEY, keys_bytes));
    }

    // ── Surrogate maps ────────────────────────────────────────────────────────
    let surrogates = gens.plan(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY, full);
    if surrogates.is_some() {
        let surrogate_state = FtsSurrogateState {
            id_to_surrogate: id_to_surrogate
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            next_surrogate,
        };
        let surrogate_bytes = zerompk::to_msgpack_vec(&surrogate_state)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?;
        ops.push(WriteOp::Put {
            ns: Namespace::Fts,
            key: SURROGATES_KEY.to_vec(),
            value: surrogate_bytes,
        });
    }

    // ── Per-index data ────────────────────────────────────────────────────────
    let mut indexes: Vec<FtsIndexFlush> = Vec::new();
    for index_key in &index_keys {
        let Some(idx) = indices.get(index_key) else {
            continue;
        };
        let Some(planned) = gens.plan(FlushArtifact::FtsIndex, index_key, full) else {
            continue;
        };
        // Doc-lengths and meta blobs go on the B+ tree. Posting data is
        // dispatched to a pagedb segment or unpacked into per-term KV entries
        // at write time.
        let mut index_ops: Vec<WriteOp> = Vec::new();
        metadata_ops_for_index(index_key, idx, &mut index_ops)?;
        let postings = serialize_postings_blob(index_key, idx)?;
        indexes.push(FtsIndexFlush {
            planned,
            ops: index_ops,
            postings: Some(postings),
        });
    }

    // ── Dropped indexes ───────────────────────────────────────────────────────
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
        meta,
    })
}

/// Write the FTS state `serialize_fts` planned.
///
/// Each index's postings are written first: a pagedb segment when the storage
/// has FTS segments, per-term KV entries otherwise. A failed postings write is
/// logged, its doc-length and meta ops stay out of the batch, and the index
/// stays dirty, so the next flush retries it. The batch then commits the index
/// list, the surrogate map, and the doc-length and meta ops of every index
/// whose postings landed. It is skipped when empty. A failed batch returns the
/// error with nothing marked.
///
/// Callers serialize inside the FTS mutex (sync, no I/O) and call this
/// function after releasing the lock to perform async I/O. Returns the index
/// writes made durable.
pub(crate) async fn write_serialized_fts<S>(
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
        meta,
    } = flush;

    let mut written: Vec<ArtifactFlush> = Vec::with_capacity(indexes.len());
    for index in indexes {
        let FtsIndexFlush {
            planned,
            ops: index_ops,
            postings,
        } = index;
        match write_postings(storage, planned.key(), postings.as_deref()).await {
            Ok(posting_ops) => {
                ops.extend(index_ops);
                ops.extend(posting_ops);
                written.push(planned);
            }
            Err(e) => {
                tracing::error!(
                    index_key = %planned.key(),
                    error = %e,
                    "fts postings write failed; \
                     the index stays dirty and the next flush retries it"
                );
            }
        }
    }

    if !ops.is_empty() {
        storage
            .batch_write(&ops)
            .await
            .map_err(|e| NodeDbError::storage(format!("fts checkpoint batch_write: {e}")))?;
    }

    // Every write above is durable now. Record the captured generations,
    // never the current ones.
    for planned in written.iter().chain(surrogates.iter()) {
        gens.mark_flushed(planned);
        gens.record_write(planned);
    }
    if let Some((key, value)) = meta {
        gens.mark_meta_written(key, value);
    }

    Ok(written)
}

/// Write the postings of the index under `index_key` and return the KV ops
/// the batch must carry for them.
///
/// With FTS segments, `Some` writes the blob as the index's segment and `None`
/// deletes the segment. Nothing is added to the batch. Without them the
/// postings live as one KV entry per term: the returned ops put every term of
/// the blob and delete every stored term of this index the blob no longer
/// holds, so a removed term cannot come back on restore.
async fn write_postings<S>(
    storage: &S,
    index_key: &str,
    postings: Option<&[u8]>,
) -> NodeDbResult<Vec<WriteOp>>
where
    S: StorageEngine,
{
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

    // KV fallback path (WASM / legacy backends / test doubles): unpack the
    // posting blob back into per-term KV entries.
    let entries: Vec<(String, Vec<SerPosting>)> = match postings {
        Some(blob) => zerompk::from_msgpack::<Vec<(String, Vec<SerPosting>)>>(blob)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?,
        None => Vec::new(),
    };
    let mt_prefix = format!("fts:{index_key}:mt:");
    let mut ops: Vec<WriteOp> = Vec::with_capacity(entries.len());
    let mut kept: HashSet<Vec<u8>> = HashSet::with_capacity(entries.len());
    for (scoped_term, term_postings) in entries {
        let bytes = zerompk::to_msgpack_vec(&term_postings)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?;
        let mt_key = format!("{mt_prefix}{scoped_term}").into_bytes();
        kept.insert(mt_key.clone());
        ops.push(WriteOp::Put {
            ns: Namespace::Fts,
            key: mt_key,
            value: bytes,
        });
    }

    // A stored term is this index's only when its scoped term names this
    // index. The prefix also matches the terms of an index whose key extends
    // this one's, e.g. `a:b` and `a:b:mt:c`, and those are left alone.
    let owned_scope = format!("0:0:{index_key}:");
    let stored = storage
        .scan_prefix(Namespace::Fts, mt_prefix.as_bytes())
        .await
        .map_err(NodeDbError::storage)?;
    for (raw_key, _) in stored {
        let owned = raw_key
            .strip_prefix(mt_prefix.as_bytes())
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

/// Restore FTS state from storage on cold open.
///
/// `governor` is bound into every restored [`FtsIndex`] for memory accounting.
///
/// Returns an empty state if no checkpoint is found. An index whose postings,
/// doc lengths, or meta blobs fail to decode is restored with what did
/// decode, and is left out of [`RestoredFts::decoded`].
pub(crate) async fn restore_fts<S>(
    storage: &S,
    governor: Arc<MemoryGovernor>,
) -> NodeDbResult<RestoredFts>
where
    S: StorageEngine,
{
    const TID: u64 = 0;
    const DB: u64 = 0;

    // ── Read collection list ──────────────────────────────────────────────────
    let Some(keys_bytes) = storage.get(Namespace::Fts, COLLECTIONS_KEY).await? else {
        return Ok(RestoredFts::empty());
    };
    let Ok(index_keys) = zerompk::from_msgpack::<Vec<String>>(&keys_bytes) else {
        tracing::warn!("fts checkpoint: failed to decode collection list — starting fresh");
        return Ok(RestoredFts::empty());
    };

    if index_keys.is_empty() {
        return Ok(RestoredFts::empty());
    }

    // ── Read surrogate maps ───────────────────────────────────────────────────
    let surrogate_bytes = storage
        .get(Namespace::Fts, SURROGATES_KEY)
        .await?
        .unwrap_or_default();
    let (id_to_surrogate, surrogate_to_id, next_surrogate) =
        if let Ok(state) = zerompk::from_msgpack::<FtsSurrogateState>(&surrogate_bytes) {
            let mut i2s: HashMap<String, u32> = HashMap::with_capacity(state.id_to_surrogate.len());
            let mut s2i: HashMap<u32, String> = HashMap::with_capacity(state.id_to_surrogate.len());
            for (id, s) in state.id_to_surrogate {
                s2i.insert(s, id.clone());
                i2s.insert(id, s);
            }
            (i2s, s2i, state.next_surrogate)
        } else {
            tracing::warn!("fts checkpoint: failed to decode surrogate maps — starting fresh");
            return Ok(RestoredFts::empty());
        };

    #[cfg(not(target_arch = "wasm32"))]
    let seg_ext = storage.as_fts_segment_ext();
    // A flush on storage with FTS segments writes a segment, so per-term KV
    // postings there never match what it would write.
    #[cfg(not(target_arch = "wasm32"))]
    let segment_capable = seg_ext.is_some();
    #[cfg(target_arch = "wasm32")]
    let segment_capable = false;

    // ── Restore per-index data ────────────────────────────────────────────────
    let mut indices: HashMap<String, FtsIndex<MemoryBackend>> =
        HashMap::with_capacity(index_keys.len());
    let mut decoded: HashSet<String> = HashSet::with_capacity(index_keys.len());

    for index_key in &index_keys {
        let backend = MemoryBackend::new();
        let idx = FtsIndex::with_memtable_config(
            backend,
            super::LITE_MEMTABLE_CONFIG,
            Arc::clone(&governor),
        );

        // ── Posting data: try pagedb segment path first, fall back to KV ─────
        // Flipped only by the native segment path, compiled out on wasm32.
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut restored_from_segment = false;

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(seg_ext) = seg_ext {
            match seg_ext.open_fts_segment(index_key).await {
                Ok(Some(blob)) => {
                    if let Ok(entries) =
                        zerompk::from_msgpack::<Vec<(String, Vec<SerPosting>)>>(&blob)
                    {
                        for (scoped_term, postings) in entries {
                            for sp in postings {
                                idx.memtable().insert(&scoped_term, ser_to_compact(sp));
                            }
                        }
                        restored_from_segment = true;
                    } else {
                        tracing::warn!(
                            index_key,
                            "fts segment blob corrupt — falling back to KV postings"
                        );
                    }
                }
                Ok(None) => {
                    // No segment yet (first open after migration, or empty index).
                    // Will fall through to KV scan below.
                }
                Err(e) => {
                    tracing::warn!(
                        index_key,
                        error = %e,
                        "fts segment open failed — falling back to KV postings"
                    );
                }
            }
        }

        // KV fallback: legacy per-term posting entries.
        let mut postings_decoded = restored_from_segment;
        if !restored_from_segment {
            let mut all_decoded = true;
            let mt_prefix = format!("fts:{index_key}:mt:").into_bytes();
            let mt_entries = storage.scan_prefix(Namespace::Fts, &mt_prefix).await?;
            let mt_prefix_str = format!("fts:{index_key}:mt:");
            for (raw_key, value) in &mt_entries {
                let key_str = String::from_utf8_lossy(raw_key);
                let scoped_term = key_str
                    .strip_prefix(&mt_prefix_str)
                    .unwrap_or("")
                    .to_string();
                if scoped_term.is_empty() {
                    continue;
                }
                if let Ok(ser) = zerompk::from_msgpack::<Vec<SerPosting>>(value) {
                    for sp in ser {
                        idx.memtable().insert(&scoped_term, ser_to_compact(sp));
                    }
                } else {
                    all_decoded = false;
                }
            }
            if !all_decoded {
                tracing::warn!(
                    index_key,
                    "fts KV postings corrupt — restoring what decoded; the index starts dirty"
                );
            }
            postings_decoded = all_decoded && !segment_capable;
        }

        // ── Doc lengths (always on B+ tree) ──────────────────────────────────
        // Absent means the stored form predates the doc-length list a flush
        // now always writes, so the index starts dirty.
        let doclens_key = format!("fts:{index_key}:doclens");
        let doclens_decoded = match storage.get(Namespace::Fts, doclens_key.as_bytes()).await? {
            Some(data) => match zerompk::from_msgpack::<Vec<(u32, u32)>>(&data) {
                Ok(pairs) => {
                    let mut applied = true;
                    for (s, len) in pairs {
                        applied &= idx
                            .backend()
                            .write_doc_length(DB, TID, index_key, Surrogate(s), len)
                            .is_ok();
                        applied &= idx
                            .backend()
                            .increment_stats(DB, TID, index_key, len)
                            .is_ok();
                    }
                    applied
                }
                Err(_) => {
                    tracing::warn!(
                        index_key,
                        "fts doc lengths corrupt — the index starts dirty"
                    );
                    false
                }
            },
            None => false,
        };

        // ── Meta blobs (always on B+ tree) ────────────────────────────────────
        let mut meta_decoded = true;
        for &subkey in META_SUBKEYS {
            let meta_key = format!("fts:{index_key}:meta:{subkey}");
            if let Some(data) = storage.get(Namespace::Fts, meta_key.as_bytes()).await?
                && idx
                    .backend()
                    .write_meta(DB, TID, index_key, subkey, &data)
                    .is_err()
            {
                meta_decoded = false;
            }
        }

        if postings_decoded && doclens_decoded && meta_decoded {
            decoded.insert(index_key.clone());
        }
        indices.insert(index_key.clone(), idx);
    }

    tracing::debug!(
        index_count = indices.len(),
        decoded_count = decoded.len(),
        surrogate_count = id_to_surrogate.len(),
        "fts checkpoint restored"
    );

    Ok(RestoredFts {
        indices,
        id_to_surrogate,
        surrogate_to_id,
        next_surrogate,
        decoded,
        surrogates_decoded: true,
        stored_catalog: Some((COLLECTIONS_KEY, keys_bytes)),
    })
}
