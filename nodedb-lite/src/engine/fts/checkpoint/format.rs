// SPDX-License-Identifier: Apache-2.0

//! On-storage shapes and key names of the FTS checkpoint.
//!
//! ## Storage layout
//!
//! ### B+ tree (`Namespace::Fts`) — always used
//!
//! | Key                             | Value                                      |
//! |---------------------------------|--------------------------------------------|
//! | `fts:_collections`              | MessagePack `Vec<String>` — index key list |
//! | `fts:_surrogates`               | MessagePack `FtsSurrogateState`            |
//! | `fts:_layout`                   | one byte — checkpoint layout version       |
//! | `fts:{index_key}:doclens`       | MessagePack `Vec<(u32,u32)>` — surrogate/len |
//! | `fts:{index_key}:meta:{subkey}` | raw bytes (fieldnorms/analyzer/language/fuzzy)   |
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
//!
//! A flush writes only the indexes that changed since this handle last wrote
//! them, the surrogate map only when a surrogate was allocated, and the index
//! list only when the set of indexes changed. A changed index is written
//! whole: an empty one writes an empty segment and doc-length list, and a
//! rewritten index on the KV path deletes the terms it no longer holds. A
//! dropped index has its segment, doc lengths and meta blobs deleted, and a
//! changed index list also removes every key and segment no current index
//! owns, so a dropped index or a retracted term does not come back on restore.
//!
//! ## Rationale: memtable vs segment storage
//!
//! `nodedb-fts` on Lite uses `MemoryBackend` exclusively, with memtable spill
//! thresholds out of reach, so all postings live in the `Memtable`; the
//! backend's LSM segment layer is unused. The pagedb segment path bundles all
//! per-term posting entries for one index key into a single segment blob,
//! reducing B+ tree pressure from O(vocab_size) entries to O(1) per index key.

use nodedb_fts::block::CompactPosting;
use nodedb_types::Surrogate;
use serde::{Deserialize, Serialize};

/// Key of the index key list.
pub(crate) const COLLECTIONS_KEY: &[u8] = b"fts:_collections";

/// Key of the surrogate maps.
pub(super) const SURROGATES_KEY: &[u8] = b"fts:_surrogates";

/// Key of the checkpoint layout version byte.
pub(super) const LAYOUT_KEY: &[u8] = b"fts:_layout";

/// Layout version whose schemaless documents carry per-field indexes next to
/// the whole-document index. A checkpoint without it holds only
/// whole-document indexes for schemaless documents.
pub(super) const LAYOUT_PER_FIELD: u8 = 2;

/// Prefix of every key the checkpoint owns in `Namespace::Fts`.
pub(super) const CHECKPOINT_PREFIX: &[u8] = b"fts:";

/// Prefix of the pagedb segment-index sentinels, owned by the segment store.
pub(super) const SEGMENT_INDEX_PREFIX: &[u8] = b"fts:_seg_idx:";

/// Known meta subkeys written by `nodedb-fts`.
pub(super) const META_SUBKEYS: &[&str] = &["fieldnorms", "analyzer", "language", "fuzzy"];

/// Lite is single-database/single-tenant: both scope ids are 0.
pub(super) const DB: u64 = 0;
pub(super) const TID: u64 = 0;

/// Surrogate maps persisted alongside posting data.
#[derive(Serialize, Deserialize, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub(super) struct FtsSurrogateState {
    /// `doc_id` string → dense u32 surrogate.
    pub id_to_surrogate: Vec<(String, u32)>,
    /// Next surrogate to assign.
    pub next_surrogate: u32,
}

/// A single memtable posting entry serialized as a flat tuple.
///
/// Matches `CompactPosting` fields: `(doc_id, term_freq, fieldnorm, positions)`.
pub(super) type SerPosting = (u32, u32, u8, Vec<u32>);

/// All postings of one index: `(scoped_term, postings)` pairs.
pub(super) type PostingsBlob = Vec<(String, Vec<SerPosting>)>;

pub(super) fn compact_to_ser(p: &CompactPosting) -> SerPosting {
    (p.doc_id.0, p.term_freq, p.fieldnorm, p.positions.clone())
}

pub(super) fn ser_to_compact(s: SerPosting) -> CompactPosting {
    CompactPosting {
        doc_id: Surrogate(s.0),
        term_freq: s.1,
        fieldnorm: s.2,
        positions: s.3,
    }
}

/// Key of the doc-length list of `index_key`.
pub(super) fn doclens_key(index_key: &str) -> String {
    format!("fts:{index_key}:doclens")
}

/// Key of the `subkey` meta blob of `index_key`.
pub(super) fn meta_key(index_key: &str, subkey: &str) -> String {
    format!("fts:{index_key}:meta:{subkey}")
}

/// Prefix of the legacy per-term posting keys of `index_key`.
pub(super) fn postings_prefix(index_key: &str) -> String {
    format!("fts:{index_key}:mt:")
}

/// The memtable scope of `index_key`. Each Lite index is its own
/// [`nodedb_fts::FtsIndex`] holding one whole-document scope named by the key.
pub(crate) fn index_scope(index_key: &str) -> nodedb_fts::IndexScope<'_> {
    nodedb_fts::IndexScope::document(index_key)
}

/// The persisted name of `term` in `index_key`: `"{DB}:{TID}:{index_key}:{term}"`.
///
/// Checkpoints written before nodedb main's per-scope memtable stored its
/// flat memtable keys verbatim. The format keeps that shape, so a
/// checkpoint reads the same across the upgrade in either direction.
pub(super) fn scoped_term(index_key: &str, term: &str) -> String {
    format!("{DB}:{TID}:{index_key}:{term}")
}

/// The bare term of a persisted `scoped_term`. `None` when the name belongs
/// to another index.
pub(super) fn bare_term<'t>(index_key: &str, scoped: &'t str) -> Option<&'t str> {
    scoped
        .strip_prefix(&format!("{DB}:{TID}:{index_key}:"))
        .filter(|t| !t.is_empty())
}
