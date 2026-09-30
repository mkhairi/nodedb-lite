//! Per-collection FTS manager for Lite.
//!
//! Wraps `nodedb_fts::FtsIndex<MemoryBackend>` with per-collection management:
//! - Incremental insert/remove on document put/delete
//! - Multi-field per-collection keying (`collection:field`)
//! - BM25 search delegated directly to nodedb-fts (BMW, analyzers, fuzzy)
//! - Persistent: checkpoint serialized to `Namespace::Fts` on `flush()`,
//!   restored on `NodeDbLite::open` without re-tokenizing source documents.
//!
//! This is the canonical FTS implementation for Lite.
//!
//! Every mutation goes through a `&mut self` method here, and each one marks
//! the index it changed dirty for flush. A call that changes nothing marks
//! nothing, so an idle index is not rewritten. The surrogate map is marked
//! dirty only when a new surrogate is allocated.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tracing;

use nodedb_fts::FtsIndex;
use nodedb_fts::FtsSearchParams;
use nodedb_fts::backend::FtsBackend;
use nodedb_fts::backend::memory::MemoryBackend;
use nodedb_fts::posting::QueryMode as FtsQueryMode;
use nodedb_mem::MemoryGovernor;
use nodedb_types::Surrogate;
use nodedb_types::error::NodeDbResult;
use nodedb_types::text_search::{QueryMode, TextSearchParams};

use crate::error::LiteError;
use crate::nodedb::flush_gens::{ArtifactFlush, FTS_SURROGATES_KEY, FlushArtifact, FlushGens};

use super::checkpoint::{FtsFlush, RestoredFts};

/// A resolved FTS result with the original string doc_id restored.
pub struct FtsResult {
    pub doc_id: String,
    pub score: f32,
    pub fuzzy: bool,
}

/// Wrap an index-layer failure as the typed error the write path propagates.
fn fts_err(collection: &str, e: impl std::fmt::Display) -> LiteError {
    LiteError::FtsIndex {
        collection: collection.to_owned(),
        detail: e.to_string(),
    }
}

/// Whether `idx` stores a document length for `surrogate`. A read error
/// counts as stored, so a caller that marks on it over-marks rather than
/// missing a change.
fn holds_length(idx: &FtsIndex<MemoryBackend>, key: &str, surrogate: Surrogate) -> bool {
    !matches!(
        idx.backend().read_doc_length(0, 0, key, surrogate),
        Ok(None)
    )
}

/// Manages per-collection (and per-field) in-memory full-text search indexes.
///
/// Each `(collection, field)` pair gets its own `FtsIndex<MemoryBackend>`.
/// A special `collection:_doc` key is used for whole-document text indexing
/// (all string fields concatenated) used by the `text_search` API.
pub struct FtsCollectionManager {
    /// Key: `"{collection}:{field}"` → FTS index.
    /// Whole-document index uses key `"{collection}:_doc"`.
    ///
    /// Private: every mutation must go through a method that marks the index
    /// dirty for flush.
    indices: HashMap<String, FtsIndex<MemoryBackend>>,
    /// Index keys `drop_collection` removed whose stored segment, doc lengths,
    /// and meta blobs no flush has deleted yet. A key leaves the set once its
    /// deletion is durable, or when the index is created again.
    dropped: HashSet<String>,
    /// Forward map: original string doc_id → dense u32 surrogate.
    ///
    /// Surrogates **must** be dense (0, 1, 2, …) because `nodedb_fts::Memtable`
    /// uses them as direct indices into a `Vec<u8>` fieldnorm array
    /// (`record_doc` calls `vec.resize(surrogate + 1, 0)`). Hashing strings
    /// into the u32 space produced sparse surrogates near `u32::MAX`, which
    /// allocated multi-gigabyte zero-filled vectors per insert and made
    /// indexing effectively hang.
    id_to_surrogate: HashMap<String, u32>,
    /// Reverse map: surrogate u32 → original string doc_id.
    surrogate_to_id: HashMap<u32, String>,
    /// Next surrogate to assign on first sighting of a doc_id.
    next_surrogate: u32,
    /// Reverse map: Origin global surrogate → Lite string doc_id.
    ///
    /// Populated when `FtsIndexDoc` frames arrive from Origin via the sync path.
    /// Needed by `FtsDeleteDoc` to translate the Origin surrogate back to the
    /// Lite string doc_id without dropping the whole collection.
    origin_surrogate_to_doc_id: HashMap<u32, String>,
    /// Collection name → bound analyzer name, from `TextOp::SetTextConfig`.
    ///
    /// Retained so indexes created after the analyzer was bound inherit it —
    /// DDL normally binds the analyzer before the first document is written,
    /// when none of the collection's indexes exist yet. See
    /// `super::analyzer` for the binding logic.
    pub(super) collection_analyzers: HashMap<String, String>,
    /// Collection name → default fuzzy matching, from `TextOp::SetTextConfig`.
    ///
    /// Retained for the same reason as `collection_analyzers`: the DDL that
    /// sets it usually runs before any of the collection's indexes exist.
    pub(super) collection_fuzzy_defaults: HashMap<String, bool>,
    /// Memory governor bound into every `FtsIndex` this manager creates.
    pub(super) governor: Arc<MemoryGovernor>,
    /// Flush dirty tracking for every index, under its index key, and for
    /// the surrogate map.
    gens: Arc<FlushGens>,
}

impl FtsCollectionManager {
    /// Create an empty manager with its own, unshared flush tracking.
    pub fn new(governor: Arc<MemoryGovernor>) -> Self {
        Self::with_gens(governor, Arc::new(FlushGens::default()))
    }

    /// Create an empty manager that records its mutations in `gens`.
    ///
    /// The store passes its own `FlushGens`, so its flush sees which indexes
    /// changed.
    pub(crate) fn with_gens(governor: Arc<MemoryGovernor>, gens: Arc<FlushGens>) -> Self {
        Self {
            indices: HashMap::new(),
            dropped: HashSet::new(),
            id_to_surrogate: HashMap::new(),
            surrogate_to_id: HashMap::new(),
            // Start at 1: Surrogate(0) is the unassigned sentinel and is
            // rejected by FtsIndex::index_document with SurrogateOutOfRange.
            next_surrogate: 1,
            origin_surrogate_to_doc_id: HashMap::new(),
            collection_analyzers: HashMap::new(),
            collection_fuzzy_defaults: HashMap::new(),
            governor,
            gens,
        }
    }

    /// Mark the index under `key` dirty.
    fn mark_index(&self, key: &str) {
        self.gens.bump(FlushArtifact::FtsIndex, key);
    }

    /// Look up or allocate a dense surrogate for a string `doc_id`.
    ///
    /// Returns the existing surrogate if `doc_id` has been indexed before,
    /// otherwise assigns the next sequential u32 and records the mapping
    /// in both directions.
    fn surrogate_for(&mut self, doc_id: &str) -> Surrogate {
        if let Some(&s) = self.id_to_surrogate.get(doc_id) {
            return Surrogate(s);
        }
        let s = self.next_surrogate;
        self.next_surrogate = self
            .next_surrogate
            .checked_add(1)
            .expect("FTS surrogate counter overflowed u32");
        self.id_to_surrogate.insert(doc_id.to_owned(), s);
        self.surrogate_to_id.insert(s, doc_id.to_owned());
        self.gens
            .bump(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY);
        Surrogate(s)
    }

    /// Look up an existing surrogate without allocating one.
    fn lookup_surrogate(&self, doc_id: &str) -> Option<Surrogate> {
        self.id_to_surrogate.get(doc_id).copied().map(Surrogate)
    }

    /// Returns true if no collections are indexed.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    // ── Whole-document indexing (used by `document_put` / `text_search`) ─────

    /// Index all string field values from a document as a single text blob.
    ///
    /// The document is stored under the `"{collection}:_doc"` key.
    /// Calling again with the same `doc_id` replaces the previous entry.
    ///
    /// Empty text is a removal, not a no-op: a document updated until it has
    /// no indexable words must stop matching the words it used to contain.
    pub fn index_document(
        &mut self,
        collection: &str,
        doc_id: &str,
        text: &str,
    ) -> Result<(), LiteError> {
        if text.is_empty() {
            return self.remove_document(collection, doc_id);
        }
        let surrogate = self.surrogate_for(doc_id);
        let key = format!("{collection}:_doc");
        self.upsert_text(collection, &key, surrogate, text)
    }

    /// Remove a document from the whole-document index.
    pub fn remove_document(&mut self, collection: &str, doc_id: &str) -> Result<(), LiteError> {
        let key = format!("{collection}:_doc");
        self.remove_text(collection, &key, doc_id)
    }

    /// Replace the entry for `surrogate` in the index under `key` with
    /// `text`, creating the index when absent (upsert semantics).
    ///
    /// Marks the index dirty when it was created, when the document was in
    /// it before or is in it after, when its posting count changed, or when
    /// the write failed partway. Re-indexing identical text over-marks; a
    /// call that touches nothing does not mark.
    fn upsert_text(
        &mut self,
        collection: &str,
        key: &str,
        surrogate: Surrogate,
        text: &str,
    ) -> Result<(), LiteError> {
        if !self.indices.contains_key(key) {
            let fresh = self.new_index_for(key);
            self.indices.insert(key.to_owned(), fresh);
            // The write of the new index replaces whatever the dropped one
            // left in storage.
            self.dropped.remove(key);
            self.mark_index(key);
        }
        let Some(idx) = self.indices.get(key) else {
            return Ok(());
        };
        let held = holds_length(idx, key, surrogate);
        let postings_before = idx.memtable().posting_count();
        // Remove old entry first (upsert semantics).
        let result = idx
            .remove_document(0, 0, key, surrogate)
            .map_err(|e| fts_err(collection, e))
            .and_then(|()| {
                idx.index_document(0, 0, key, surrogate, text)
                    .map_err(|e| fts_err(collection, e))
            });
        if held
            || holds_length(idx, key, surrogate)
            || idx.memtable().posting_count() != postings_before
            || result.is_err()
        {
            self.mark_index(key);
        }
        result
    }

    /// Remove `doc_id` from the index under `key`.
    ///
    /// Marks the index dirty only when the document held a length or
    /// postings there, or when the removal failed partway.
    fn remove_text(&mut self, collection: &str, key: &str, doc_id: &str) -> Result<(), LiteError> {
        let Some(surrogate) = self.lookup_surrogate(doc_id) else {
            return Ok(());
        };
        let Some(idx) = self.indices.get(key) else {
            return Ok(());
        };
        let held = holds_length(idx, key, surrogate);
        let postings_before = idx.memtable().posting_count();
        let result = idx
            .remove_document(0, 0, key, surrogate)
            .map_err(|e| fts_err(collection, e));
        if held || idx.memtable().posting_count() != postings_before || result.is_err() {
            self.mark_index(key);
        }
        result
    }

    /// Search the whole-document index for a collection.
    ///
    /// All query knobs are passed via [`TextSearchParams`]: boolean mode (OR/AND),
    /// fuzzy matching, and BM25 scoring parameters (k1, b).
    pub fn search(
        &self,
        collection: &str,
        query: &str,
        top_k: usize,
        params: &TextSearchParams,
    ) -> Vec<FtsResult> {
        let key = format!("{collection}:_doc");
        let Some(idx) = self.indices.get(&key) else {
            return Vec::new();
        };
        let mode = match params.mode {
            QueryMode::Or => FtsQueryMode::Or,
            QueryMode::And => FtsQueryMode::And,
            _ => FtsQueryMode::Or,
        };
        let raw = idx
            .search(
                0,
                0,
                &key,
                FtsSearchParams {
                    query,
                    top_k,
                    fuzzy_enabled: params.fuzzy,
                    mode,
                    prefilter: None,
                },
            )
            .inspect_err(|e| tracing::warn!(collection, error = %e, "fts search failed"))
            .unwrap_or_default();
        raw.into_iter()
            .filter_map(|r| {
                let doc_id = self.surrogate_to_id.get(&r.doc_id.0)?.clone();
                Some(FtsResult {
                    doc_id,
                    score: r.score,
                    fuzzy: r.fuzzy,
                })
            })
            .collect()
    }

    /// Like [`Self::search`] but restricts results to documents whose string
    /// doc_id is in `allowed`. Fetches `top_k * 8` candidates from BM25 to
    /// account for haystack documents that rank below non-haystack documents.
    pub(crate) fn search_with_allowed(
        &self,
        collection: &str,
        query: &str,
        top_k: usize,
        params: &TextSearchParams,
        allowed: &std::collections::HashSet<String>,
    ) -> Vec<FtsResult> {
        let fetch_k = top_k.saturating_mul(8).max(top_k);
        self.search(collection, query, fetch_k, params)
            .into_iter()
            .filter(|r| allowed.contains(&r.doc_id))
            .take(top_k)
            .collect()
    }

    // ── BM25ScoreScan: all docs with injected score (0.0 for non-matches) ────

    /// Return every known document in `collection` together with its BM25 score
    /// against `query`. Documents that are not in the BM25 hit set receive
    /// score `0.0`. This powers `TextOp::BM25ScoreScan`.
    pub fn scan_all_with_scores(
        &self,
        collection: &str,
        query: &str,
        params: &TextSearchParams,
    ) -> Vec<(String, f32)> {
        let key = format!("{collection}:_doc");
        let Some(idx) = self.indices.get(&key) else {
            return Vec::new();
        };
        let mode = match params.mode {
            QueryMode::Or => FtsQueryMode::Or,
            QueryMode::And => FtsQueryMode::And,
            _ => FtsQueryMode::Or,
        };
        // Fetch BM25 hits for the query (all matching docs).
        // Use the total known-surrogate count as top_k; this is a safe upper
        // bound and avoids passing usize::MAX which causes a heap allocation overflow.
        let total_known = self.surrogate_to_id.len().max(1);
        let hits: HashMap<u32, f32> = idx
            .search(
                0,
                0,
                &key,
                FtsSearchParams {
                    query,
                    top_k: total_known,
                    fuzzy_enabled: params.fuzzy,
                    mode,
                    prefilter: None,
                },
            )
            .inspect_err(|e| tracing::warn!(collection, error = %e, "bm25 scan failed"))
            .unwrap_or_default()
            .into_iter()
            .map(|r| (r.doc_id.0, r.score))
            .collect();

        // Emit every known doc_id in this collection with its score (0.0 if absent).
        self.surrogate_to_id
            .iter()
            .filter_map(|(&sur, doc_id)| {
                // Only include surrogates that belong to this collection by checking
                // whether this surrogate appears in the index at all (has a doc_len).
                // We use id_to_surrogate presence as the membership test.
                if self.id_to_surrogate.contains_key(doc_id) {
                    let score = hits.get(&sur).copied().unwrap_or(0.0);
                    Some((doc_id.clone(), score))
                } else {
                    None
                }
            })
            .collect()
    }

    // ── PhraseSearch: exact consecutive-term matching ─────────────────────────

    /// Search for documents where `terms` appear as an exact consecutive phrase.
    ///
    /// Algorithm: fetch OR results from BM25 (any term present), then filter
    /// to candidates that contain all terms with consecutive positions
    /// (term_0 at position p, term_1 at p+1, …). Scoring is BM25 score with
    /// an earlier-position bonus (higher score for phrases closer to doc start).
    pub fn phrase_search(
        &self,
        collection: &str,
        terms: &[String],
        top_k: usize,
        params: &TextSearchParams,
    ) -> Vec<FtsResult> {
        if terms.is_empty() {
            return Vec::new();
        }
        let key = format!("{collection}:_doc");
        let Some(idx) = self.indices.get(&key) else {
            return Vec::new();
        };

        // Gather OR results for all terms to get candidates with position data.
        // Use a generous multiplier over top_k; phrase filter will further reduce
        // the set. Capped at the total known-doc count to avoid heap overflow.
        let query = terms.join(" ");
        let candidate_limit = (top_k * 10).max(100).min(self.surrogate_to_id.len().max(1));
        let or_hits = idx
            .search(
                0,
                0,
                &key,
                FtsSearchParams {
                    query: &query,
                    top_k: candidate_limit,
                    fuzzy_enabled: params.fuzzy,
                    mode: FtsQueryMode::Or,
                    prefilter: None,
                },
            )
            .inspect_err(|e| tracing::warn!(collection, error = %e, "phrase search or-pass failed"))
            .unwrap_or_default();

        if or_hits.is_empty() {
            return Vec::new();
        }

        // For each candidate doc, retrieve per-term position lists and check
        // for a consecutive sequence: term[i] at pos p, term[i+1] at p+1, etc.
        let mut phrase_hits: Vec<FtsResult> = or_hits
            .into_iter()
            .filter_map(|hit| {
                let sur = hit.doc_id;
                // Retrieve postings for each term from the memtable.
                let term_positions: Vec<Vec<u32>> = terms
                    .iter()
                    .map(|term| {
                        // Memtable key scope is `{database_id}:{tenant}:{collection}:{term}`;
                        // Lite is single-database/single-tenant, so both ids are 0.
                        let scoped = format!("0:0:{key}:{term}");
                        idx.memtable()
                            .get_postings(&scoped)
                            .into_iter()
                            .find(|p| p.doc_id == sur)
                            .map(|p| p.positions.clone())
                            .unwrap_or_default()
                    })
                    .collect();

                // Check that every term has at least one position.
                if term_positions.iter().any(|p| p.is_empty()) {
                    return None;
                }

                // Find any anchor position p in term_positions[0] such that
                // term_positions[i] contains p+i for all i.
                let anchors = &term_positions[0];
                let found = anchors.iter().any(|&p| {
                    term_positions
                        .iter()
                        .enumerate()
                        .skip(1)
                        .all(|(i, positions)| positions.binary_search(&(p + i as u32)).is_ok())
                });

                if !found {
                    return None;
                }

                // Score: BM25 score with earlier-position bonus.
                let earliest = anchors.iter().copied().min().unwrap_or(u32::MAX);
                let position_bonus = 1.0 / (1.0 + earliest as f32 * 0.01);
                let score = hit.score * position_bonus;

                let doc_id = self.surrogate_to_id.get(&sur.0)?.clone();
                Some(FtsResult {
                    doc_id,
                    score,
                    fuzzy: hit.fuzzy,
                })
            })
            .collect();

        phrase_hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        phrase_hits.truncate(top_k);
        phrase_hits
    }

    // ── Origin-surrogate reverse map (for FtsIndexDoc / FtsDeleteDoc sync) ────

    /// Register an association between an Origin global surrogate and the
    /// Lite string `doc_id`. Called from the `FtsIndexDoc` execution arm so
    /// `FtsDeleteDoc` can later resolve the Origin surrogate to a string doc_id
    /// and call the proper single-doc removal instead of dropping the collection.
    pub fn register_origin_surrogate(&mut self, origin_surrogate: Surrogate, doc_id: &str) {
        self.origin_surrogate_to_doc_id
            .insert(origin_surrogate.0, doc_id.to_owned());
    }

    /// Remove a single document identified by its Origin-assigned surrogate.
    ///
    /// Returns `true` if the document was found and removed, `false` if the
    /// surrogate has no known Lite mapping (e.g. it was never indexed via
    /// this Lite instance).
    pub fn remove_by_origin_surrogate(
        &mut self,
        collection: &str,
        origin_surrogate: Surrogate,
    ) -> Result<Option<String>, LiteError> {
        let Some(doc_id) = self.origin_surrogate_to_doc_id.remove(&origin_surrogate.0) else {
            tracing::debug!(
                collection,
                sur = origin_surrogate.0,
                "FtsDeleteDoc: no Lite mapping for Origin surrogate — document was never indexed here"
            );
            return Ok(None);
        };
        self.remove_document(collection, &doc_id)?;
        Ok(Some(doc_id))
    }

    // ── Per-field indexing (used by strict collections via index_integration) ─

    /// Index a single field value for a document.
    ///
    /// Key is `"{collection}:{field}"`. Calling again with the same `doc_id`
    /// replaces the previous entry (upsert semantics).
    ///
    /// Empty text is a removal, not a no-op: clearing a field must stop the
    /// document matching the words that field used to contain.
    pub fn index_field(
        &mut self,
        collection: &str,
        field: &str,
        doc_id: &str,
        text: &str,
    ) -> Result<(), LiteError> {
        if text.is_empty() {
            return self.remove_field(collection, field, doc_id);
        }
        let surrogate = self.surrogate_for(doc_id);
        let key = format!("{collection}:{field}");
        self.upsert_text(collection, &key, surrogate, text)
    }

    /// Remove all field entries for a document across all fields in a collection.
    pub fn remove_field(
        &mut self,
        collection: &str,
        field: &str,
        doc_id: &str,
    ) -> Result<(), LiteError> {
        let key = format!("{collection}:{field}");
        self.remove_text(collection, &key, doc_id)
    }

    /// Number of distinct collection prefixes with active indexes.
    pub fn collection_count(&self) -> usize {
        self.indices
            .keys()
            .map(|k| k.split(':').next().unwrap_or(k.as_str()))
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Drop all FTS indexes for a collection (called on collection drop/truncate).
    ///
    /// Each removed index is marked dirty and remembered, so the next flush
    /// deletes its stored segment, doc lengths, and meta blobs.
    pub fn drop_collection(&mut self, collection: &str) {
        let prefix = format!("{collection}:");
        let removed: Vec<String> = self
            .indices
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        for key in removed {
            self.indices.remove(&key);
            self.mark_index(&key);
            self.dropped.insert(key);
        }
    }

    /// Apply `update` to every index belonging to `collection`.
    ///
    /// `update` returns whether it changed the index. Each index it changed
    /// is marked dirty.
    pub(super) fn update_collection_indexes(
        &mut self,
        collection: &str,
        mut update: impl FnMut(&str, &FtsIndex<MemoryBackend>) -> bool,
    ) {
        let prefix = format!("{collection}:");
        for (key, idx) in &self.indices {
            if key.starts_with(&prefix) && update(key.as_str(), idx) {
                self.gens.bump(FlushArtifact::FtsIndex, key);
            }
        }
    }

    // ── Checkpoint helpers (used by core.rs flush/restore) ────────────────────

    /// Serialize the indexes, surrogate map, and index list the next flush
    /// must write, and the deletions of dropped indexes.
    ///
    /// Plans each write under this manager's lock, which the caller holds
    /// through `&self`, so every captured generation describes exactly the
    /// bytes serialized. `full` writes all of them.
    pub(crate) fn checkpoint_dirty(&self, full: bool) -> NodeDbResult<FtsFlush> {
        super::checkpoint::serialize_fts(
            &self.indices,
            &self.dropped,
            &self.id_to_surrogate,
            self.next_surrogate,
            full,
            &self.gens,
        )
    }

    /// Forget the dropped indexes whose stored forms a flush deleted.
    ///
    /// `written` lists the index writes that flush made durable. A key that
    /// was created again, or dropped again since, keeps its current state.
    pub(crate) fn forget_dropped(&mut self, written: &[ArtifactFlush]) {
        for planned in written {
            let key = planned.key();
            if !self.indices.contains_key(key) && !self.gens.is_dirty(FlushArtifact::FtsIndex, key)
            {
                self.dropped.remove(key);
            }
        }
    }

    /// Replace internal state from a restored checkpoint.  Called by
    /// `restore_fts_indices` in `core.rs` when a valid checkpoint is found.
    ///
    /// An index in `restored.decoded` matches its stored form and starts
    /// clean. Any other index starts dirty, so the next flush writes it. The
    /// surrogate map starts clean only when it decoded. The stored index list
    /// is recorded as written, so an unchanged list is not written again.
    pub(crate) fn load_checkpoint(&mut self, restored: RestoredFts) {
        let RestoredFts {
            indices,
            id_to_surrogate,
            surrogate_to_id,
            next_surrogate,
            decoded,
            surrogates_decoded,
            stored_catalog,
        } = restored;
        for key in self.indices.keys().chain(indices.keys()) {
            self.mark_index(key);
        }
        for key in indices.keys().filter(|key| decoded.contains(*key)) {
            self.gens.mark_clean(FlushArtifact::FtsIndex, key);
        }
        self.gens
            .bump(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY);
        if surrogates_decoded {
            self.gens
                .mark_clean(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY);
        }
        if let Some((key, value)) = stored_catalog {
            self.gens.mark_meta_written(key, value);
        }
        self.indices = indices;
        self.id_to_surrogate = id_to_surrogate;
        self.surrogate_to_id = surrogate_to_id;
        self.next_surrogate = next_surrogate;
        // origin_surrogate_to_doc_id is not persisted across restarts because
        // origin surrogates are only relevant for the lifetime of a sync session;
        // FtsIndexDoc frames re-register the mapping on re-sync.
    }

    /// The index under `key`, for tests that inspect one index directly.
    #[cfg(test)]
    pub(super) fn index(&self, key: &str) -> Option<&FtsIndex<MemoryBackend>> {
        self.indices.get(key)
    }
}

/// Build a real, uncapped governor for FTS tests across `engine::fts`.
#[cfg(test)]
pub(crate) fn test_governor() -> Arc<MemoryGovernor> {
    use nodedb_mem::{EngineLimits, GovernorConfig};

    let per_engine = usize::MAX / nodedb_mem::EngineId::ALL.len();
    Arc::new(
        MemoryGovernor::new(GovernorConfig {
            global_ceiling: per_engine * nodedb_mem::EngineId::ALL.len(),
            engine_limits: EngineLimits::uniform(per_engine),
        })
        .expect("test governor"),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nodedb_types::Surrogate;
    use nodedb_types::text_search::{QueryMode, TextSearchParams};

    use super::{FtsCollectionManager, test_governor};
    use crate::nodedb::flush_gens::{FTS_SURROGATES_KEY, FlushArtifact, FlushGens};

    fn default_params() -> TextSearchParams {
        TextSearchParams {
            fuzzy: false,
            mode: QueryMode::Or,
        }
    }

    // ── Stale-posting retraction ──────────────────────────────────────────────

    #[test]
    fn clearing_a_document_removes_it_from_the_index() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");
        assert_eq!(mgr.search("col", "quick", 10, &default_params()).len(), 1);

        // An update that strips every indexable word is a removal, not a no-op:
        // the document must stop matching the words it used to contain.
        mgr.index_document("col", "doc1", "")
            .expect("index update must succeed");
        assert!(
            mgr.search("col", "quick", 10, &default_params()).is_empty(),
            "cleared document must not keep matching its prior terms"
        );
    }

    #[test]
    fn clearing_a_field_removes_it_from_the_field_index() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_field("col", "title", "doc1", "the quick brown fox")
            .expect("index update must succeed");
        assert!(
            !mgr.indices
                .get("col:title")
                .expect("field index exists")
                .memtable()
                .is_empty(),
            "field index must hold the document's postings"
        );

        mgr.index_field("col", "title", "doc1", "")
            .expect("index update must succeed");
        assert!(
            mgr.indices
                .get("col:title")
                .expect("field index exists")
                .memtable()
                .is_empty(),
            "cleared field must not keep its prior postings"
        );
    }

    // ── Flush dirty tracking ──────────────────────────────────────────────────

    #[test]
    fn only_a_real_change_marks_an_index_or_the_surrogates_dirty() {
        let gens = Arc::new(FlushGens::default());
        let mut mgr = FtsCollectionManager::with_gens(test_governor(), Arc::clone(&gens));
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");
        mgr.index_document("col", "doc2", "a lazy dog")
            .expect("index update must succeed");
        mgr.index_field("col", "title", "doc1", "quick")
            .expect("index update must succeed");
        let clean = |key: &str| gens.mark_clean(FlushArtifact::FtsIndex, key);
        clean("col:_doc");
        clean("col:title");
        gens.mark_clean(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY);

        mgr.remove_document("col", "never_indexed")
            .expect("removal must succeed");
        mgr.remove_field("col", "body", "doc1")
            .expect("removal must succeed");
        mgr.remove_field("col", "title", "doc2")
            .expect("removal must succeed");
        mgr.drop_collection("nothing_here");
        for key in ["col:_doc", "col:title"] {
            assert!(
                !gens.is_dirty(FlushArtifact::FtsIndex, key),
                "a call that changes nothing must not mark {key} dirty"
            );
        }

        mgr.index_document("col", "doc1", "a slow turtle")
            .expect("index update must succeed");
        assert!(gens.is_dirty(FlushArtifact::FtsIndex, "col:_doc"));
        assert!(!gens.is_dirty(FlushArtifact::FtsIndex, "col:title"));
        assert!(
            !gens.is_dirty(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY),
            "re-indexing a known document allocates no surrogate"
        );

        mgr.index_document("col", "doc3", "brand new words")
            .expect("index update must succeed");
        assert!(gens.is_dirty(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY));

        clean("col:title");
        mgr.drop_collection("col");
        assert!(
            gens.is_dirty(FlushArtifact::FtsIndex, "col:title"),
            "a dropped index must be flushed so its stored form is deleted"
        );
    }

    // ── BM25ScoreScan ─────────────────────────────────────────────────────────

    #[test]
    fn bm25_score_scan_nonmatching_docs_get_zero_score() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");
        mgr.index_document("col", "doc2", "unrelated content about databases")
            .expect("index update must succeed");

        let scored = mgr.scan_all_with_scores("col", "quick", &default_params());
        let doc1_score = scored.iter().find(|(id, _)| id == "doc1").map(|(_, s)| *s);
        let doc2_score = scored.iter().find(|(id, _)| id == "doc2").map(|(_, s)| *s);

        assert!(
            doc1_score.is_some(),
            "doc1 must appear in scan_all_with_scores"
        );
        assert!(
            doc2_score.is_some(),
            "doc2 must appear in scan_all_with_scores"
        );
        assert!(
            doc1_score.unwrap() > 0.0,
            "doc1 matches 'quick' — score must be positive"
        );
        assert!(
            (doc2_score.unwrap() - 0.0).abs() < f32::EPSILON,
            "doc2 does not match 'quick' — score must be 0.0"
        );
    }

    #[test]
    fn bm25_score_scan_empty_collection_returns_empty() {
        let mgr = FtsCollectionManager::new(test_governor());
        let scored = mgr.scan_all_with_scores("nonexistent", "query", &default_params());
        assert!(scored.is_empty());
    }

    // ── PhraseSearch ──────────────────────────────────────────────────────────

    #[test]
    fn phrase_search_finds_exact_phrase() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox jumps over")
            .expect("index update must succeed");
        mgr.index_document("col", "doc2", "the brown quick fox")
            .expect("index update must succeed");

        let terms: Vec<String> = vec!["quick".into(), "brown".into()];
        let results = mgr.phrase_search("col", &terms, 10, &default_params());

        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        // "the quick brown fox" has quick at pos N, brown at pos N+1 — match
        // "the brown quick fox" has brown then quick — not a forward phrase match
        assert!(
            ids.contains(&"doc1"),
            "doc1 contains 'quick brown' consecutively"
        );
        assert!(
            !ids.contains(&"doc2"),
            "doc2 has 'brown quick' (reversed) — must not match"
        );
    }

    #[test]
    fn phrase_search_no_results_for_nonexistent_phrase() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");

        let terms: Vec<String> = vec!["fox".into(), "jumps".into()];
        let results = mgr.phrase_search("col", &terms, 10, &default_params());
        assert!(
            results.is_empty(),
            "phrase 'fox jumps' not in doc — no results"
        );
    }

    // ── FtsDeleteDoc / origin surrogate reverse map ───────────────────────────

    #[test]
    fn fts_delete_doc_removes_only_targeted_doc() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "rust programming language")
            .expect("index update must succeed");
        mgr.index_document("col", "doc2", "rust is fast and safe")
            .expect("index update must succeed");
        mgr.index_document("col", "doc3", "python is also great")
            .expect("index update must succeed");

        // Register origin surrogate for doc2 (as if FtsIndexDoc was dispatched).
        mgr.register_origin_surrogate(Surrogate(42), "doc2");

        // Delete via origin surrogate.
        let removed = mgr
            .remove_by_origin_surrogate("col", Surrogate(42))
            .expect("removal must succeed");
        assert!(removed.is_some(), "doc2 must be found and removed");

        // doc1 and doc3 still searchable, doc2 not.
        let results = mgr.search("col", "rust", 10, &default_params());
        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(ids.contains(&"doc1"), "doc1 must still be present");
        assert!(
            !ids.contains(&"doc2"),
            "doc2 must be removed from the index"
        );
    }

    #[test]
    fn search_with_allowed_ids_excludes_non_members() {
        use nodedb_types::text_search::{QueryMode, TextSearchParams};
        use std::collections::HashSet;

        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc-a", "rust programming language memory safe")
            .expect("index update must succeed");
        mgr.index_document("col", "doc-b", "rust is fast and compiled")
            .expect("index update must succeed");
        mgr.index_document("col", "doc-c", "python is also a language")
            .expect("index update must succeed");

        let allowed: HashSet<String> = ["doc-a".to_string()].into_iter().collect();
        let params = TextSearchParams {
            fuzzy: false,
            mode: QueryMode::Or,
        };

        let results = mgr.search_with_allowed("col", "rust", 10, &params, &allowed);
        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(
            ids.contains(&"doc-a"),
            "doc-a must appear (in allowed set and matches query), got: {ids:?}"
        );
        assert!(
            !ids.contains(&"doc-b"),
            "doc-b must be excluded (not in allowed set), got: {ids:?}"
        );
        assert!(
            !ids.contains(&"doc-c"),
            "doc-c must be excluded (not in allowed set and does not match rust), got: {ids:?}"
        );
    }

    #[test]
    fn fts_delete_doc_unknown_surrogate_returns_false() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "hello world")
            .expect("index update must succeed");

        let removed = mgr
            .remove_by_origin_surrogate("col", Surrogate(99))
            .expect("removal must succeed");
        assert!(removed.is_none(), "unknown surrogate must return None");

        // doc1 unaffected.
        let results = mgr.search("col", "hello", 10, &default_params());
        assert_eq!(results.len(), 1);
    }

    // ── HybridSearchTriple (unit-level RRF logic) ─────────────────────────────

    #[test]
    fn hybrid_triple_rrf_score_ordering() {
        // Verify that a document appearing in all three sources ranks above
        // one appearing in only one source — purely testing RRF math.
        use nodedb_query::fusion::{RankedResult, reciprocal_rank_fusion_weighted};

        let vector_ranked: Vec<RankedResult<String>> = vec![
            RankedResult {
                document_id: "A".into(),
                rank: 0,
                score: 0.9,
                source: "vector",
            },
            RankedResult {
                document_id: "B".into(),
                rank: 1,
                score: 0.5,
                source: "vector",
            },
        ];
        let text_ranked: Vec<RankedResult<String>> = vec![RankedResult {
            document_id: "A".into(),
            rank: 0,
            score: 0.8,
            source: "text",
        }];
        let graph_ranked: Vec<RankedResult<String>> = vec![RankedResult {
            document_id: "A".into(),
            rank: 0,
            score: 0.0,
            source: "graph",
        }];

        let fused = reciprocal_rank_fusion_weighted(
            &[vector_ranked, text_ranked, graph_ranked],
            &[60.0, 60.0, 60.0],
            10,
        );

        assert!(!fused.is_empty());
        assert_eq!(
            fused[0].document_id, "A",
            "A appears in all three sources — must rank first"
        );
        if fused.len() > 1 {
            assert!(
                fused[0].rrf_score > fused[1].rrf_score,
                "A's score must exceed B's"
            );
        }
    }
}
