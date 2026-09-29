// SPDX-License-Identifier: Apache-2.0

//! Per-`(collection, field)` sparse inverted index manager for Lite.
//!
//! Mirrors the FTS manager's shape: one index per named field, keyed
//! `"{collection}:{field}"`, maintained incrementally on document write and
//! delete, and checkpointed to storage on flush so a reopen is free.
//!
//! Every mutation goes through a `&mut self` method here, and each one marks
//! the index it changed dirty for flush. A call that changes nothing marks
//! nothing, so an idle index is not rewritten.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nodedb_types::SparseVector;
use nodedb_types::Value;
use nodedb_types::error::NodeDbResult;

use crate::nodedb::flush_gens::{FlushArtifact, FlushGens};

use super::checkpoint::SparseFlush;
use super::index::{SparseHit, SparseInvertedIndex};

/// Index key used when a caller does not name a field.
const DEFAULT_FIELD: &str = "_sparse";

/// Manages sparse inverted indexes for every `(collection, field)` pair.
pub struct SparseVectorManager {
    /// Key: `"{collection}:{field}"` → inverted index.
    indices: HashMap<String, SparseInvertedIndex>,
    /// Flush dirty tracking for each index, under its index key.
    gens: Arc<FlushGens>,
}

impl Default for SparseVectorManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseVectorManager {
    /// Create an empty manager with its own, unshared flush tracking.
    pub fn new() -> Self {
        Self::with_gens(Arc::new(FlushGens::default()))
    }

    /// Create an empty manager that records its mutations in `gens`.
    ///
    /// The store passes its own `FlushGens`, so its flush sees which indexes
    /// changed.
    pub(crate) fn with_gens(gens: Arc<FlushGens>) -> Self {
        Self {
            indices: HashMap::new(),
            gens,
        }
    }

    /// Build the `"{collection}:{field}"` index key.
    ///
    /// An empty `field` maps to the default field name so callers that do not
    /// name a field still address a stable index.
    pub fn index_key(collection: &str, field: &str) -> String {
        if field.is_empty() {
            format!("{collection}:{DEFAULT_FIELD}")
        } else {
            format!("{collection}:{field}")
        }
    }

    /// Whether no collection has any sparse index.
    pub fn is_empty(&self) -> bool {
        self.indices.values().all(SparseInvertedIndex::is_empty)
    }

    /// Number of live indexes.
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Insert or replace `doc_id`'s sparse vector for `(collection, field)`.
    pub fn index_document(
        &mut self,
        collection: &str,
        field: &str,
        doc_id: &str,
        vector: &SparseVector,
    ) {
        let key = Self::index_key(collection, field);
        if self
            .indices
            .entry(key.clone())
            .or_default()
            .insert(doc_id, vector)
        {
            self.gens.bump(FlushArtifact::SparseIndex, &key);
        }
    }

    /// Remove `doc_id` from one `(collection, field)` index.
    ///
    /// Returns `true` when the document was present.
    pub fn remove_document(&mut self, collection: &str, field: &str, doc_id: &str) -> bool {
        let key = Self::index_key(collection, field);
        let removed = match self.indices.get_mut(&key) {
            Some(index) => index.delete(doc_id),
            None => false,
        };
        if removed {
            self.gens.bump(FlushArtifact::SparseIndex, &key);
        }
        removed
    }

    /// Remove `doc_id` from every sparse index belonging to `collection`.
    ///
    /// Used by the document-delete path, which knows the collection and the
    /// document ID but not which fields carried sparse vectors.
    pub fn remove_document_all_fields(&mut self, collection: &str, doc_id: &str) -> usize {
        let prefix = format!("{collection}:");
        let mut removed = 0usize;
        for (key, index) in self.indices.iter_mut() {
            if key.starts_with(&prefix) && index.delete(doc_id) {
                self.gens.bump(FlushArtifact::SparseIndex, key);
                removed += 1;
            }
        }
        removed
    }

    /// Reconcile every sparse index of `collection` against a document's fields.
    ///
    /// String fields that parse as a sparse-vector literal (`'{12: 0.5}'`) are
    /// indexed under their own field name. A field that does not parse is not
    /// an error — it is simply not a sparse column — but the document is then
    /// removed from that field's index so a column that stops holding a sparse
    /// vector cannot leave stale postings behind.
    pub fn index_document_fields(
        &mut self,
        collection: &str,
        doc_id: &str,
        fields: &HashMap<String, Value>,
    ) {
        let mut indexed_fields: Vec<String> = Vec::new();

        for (field, value) in fields {
            let Value::String(literal) = value else {
                continue;
            };
            let Ok(vector) = SparseVector::parse_literal(literal) else {
                continue;
            };
            self.index_document(collection, field, doc_id, &vector);
            indexed_fields.push(Self::index_key(collection, field));
        }

        // Drop the document from any other sparse index of this collection.
        // Runs on every document write, so only an index that actually held
        // the document is marked dirty.
        let prefix = format!("{collection}:");
        for (key, index) in self.indices.iter_mut() {
            if key.starts_with(&prefix)
                && !indexed_fields.iter().any(|k| k == key)
                && index.delete(doc_id)
            {
                self.gens.bump(FlushArtifact::SparseIndex, key);
            }
        }
    }

    /// Top-`k` documents for `(collection, field)` by dot product, descending.
    ///
    /// An absent index yields no hits rather than an error — a collection that
    /// has never been written simply has nothing to match.
    pub fn search(
        &self,
        collection: &str,
        field: &str,
        query: &SparseVector,
        top_k: usize,
    ) -> Vec<SparseHit> {
        let key = Self::index_key(collection, field);
        match self.indices.get(&key) {
            Some(index) => index.search(query, top_k),
            None => Vec::new(),
        }
    }

    /// Serialize the indexes the next flush must write.
    ///
    /// Plans each write under this manager's lock, which the caller holds
    /// through `&self`, so every captured generation describes exactly the
    /// bytes serialized. `full` writes every index and the index list.
    pub(crate) fn checkpoint_dirty(&self, full: bool) -> NodeDbResult<SparseFlush> {
        super::checkpoint::serialize_sparse(&self.indices, full, &self.gens)
    }

    /// Install indexes restored from a checkpoint, replacing current state.
    ///
    /// An index in `decoded` matches its stored form and starts clean. Any
    /// other index starts dirty, so the next flush writes it.
    pub fn load_checkpoint(
        &mut self,
        indices: HashMap<String, SparseInvertedIndex>,
        decoded: &HashSet<String>,
    ) {
        for key in self.indices.keys().chain(indices.keys()) {
            self.gens.bump(FlushArtifact::SparseIndex, key);
        }
        for key in indices.keys().filter(|key| decoded.contains(*key)) {
            self.gens.mark_clean(FlushArtifact::SparseIndex, key);
        }
        self.indices = indices;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(entries: &[(u32, f32)]) -> SparseVector {
        SparseVector::from_entries(entries.to_vec()).expect("valid sparse vector")
    }

    #[test]
    fn fields_are_isolated_per_index() {
        let mut mgr = SparseVectorManager::new();
        mgr.index_document("docs", "title_sparse", "d1", &sv(&[(1, 1.0)]));
        mgr.index_document("docs", "body_sparse", "d1", &sv(&[(2, 1.0)]));

        assert_eq!(mgr.index_count(), 2);
        assert!(
            mgr.search("docs", "title_sparse", &sv(&[(2, 1.0)]), 10)
                .is_empty()
        );
        assert_eq!(
            mgr.search("docs", "body_sparse", &sv(&[(2, 1.0)]), 10)
                .len(),
            1
        );
    }

    #[test]
    fn collections_are_isolated() {
        let mut mgr = SparseVectorManager::new();
        mgr.index_document("a", "f", "d1", &sv(&[(1, 1.0)]));
        mgr.index_document("b", "f", "d1", &sv(&[(1, 1.0)]));

        mgr.remove_document_all_fields("a", "d1");
        assert!(mgr.search("a", "f", &sv(&[(1, 1.0)]), 10).is_empty());
        assert_eq!(mgr.search("b", "f", &sv(&[(1, 1.0)]), 10).len(), 1);
    }

    #[test]
    fn missing_index_returns_no_hits() {
        let mgr = SparseVectorManager::new();
        assert!(mgr.search("nope", "f", &sv(&[(1, 1.0)]), 10).is_empty());
    }

    #[test]
    fn non_sparse_string_fields_are_skipped() {
        let mut mgr = SparseVectorManager::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), Value::String("hello world".into()));
        fields.insert("n".to_string(), Value::Integer(3));
        fields.insert("emb".to_string(), Value::String("{1: 0.5}".into()));

        mgr.index_document_fields("docs", "d1", &fields);

        assert_eq!(mgr.index_count(), 1);
        assert_eq!(mgr.search("docs", "emb", &sv(&[(1, 1.0)]), 10).len(), 1);
    }

    #[test]
    fn field_that_stops_being_sparse_is_unindexed() {
        let mut mgr = SparseVectorManager::new();
        let mut fields = HashMap::new();
        fields.insert("emb".to_string(), Value::String("{1: 0.5}".into()));
        mgr.index_document_fields("docs", "d1", &fields);

        fields.insert("emb".to_string(), Value::String("plain text".into()));
        mgr.index_document_fields("docs", "d1", &fields);

        assert!(mgr.search("docs", "emb", &sv(&[(1, 1.0)]), 10).is_empty());
    }

    #[test]
    fn only_a_real_change_marks_an_index_dirty() {
        let mut mgr = SparseVectorManager::new();
        mgr.index_document("docs", "emb", "d1", &sv(&[(1, 1.0)]));
        let key = SparseVectorManager::index_key("docs", "emb");
        mgr.gens.mark_clean(FlushArtifact::SparseIndex, &key);

        mgr.index_document("docs", "emb", "d1", &sv(&[(1, 1.0)]));
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), Value::String("plain".into()));
        mgr.index_document_fields("docs", "d2", &fields);
        assert!(!mgr.remove_document("docs", "emb", "d2"));
        assert!(
            !mgr.gens.is_dirty(FlushArtifact::SparseIndex, &key),
            "a call that changes nothing must not mark the index dirty"
        );

        mgr.index_document("docs", "emb", "d1", &sv(&[(1, 2.0)]));
        assert!(mgr.gens.is_dirty(FlushArtifact::SparseIndex, &key));
    }

    #[test]
    fn remove_document_reports_presence() {
        let mut mgr = SparseVectorManager::new();
        mgr.index_document("docs", "emb", "d1", &sv(&[(1, 1.0)]));
        assert!(mgr.remove_document("docs", "emb", "d1"));
        assert!(!mgr.remove_document("docs", "emb", "d1"));
    }
}
