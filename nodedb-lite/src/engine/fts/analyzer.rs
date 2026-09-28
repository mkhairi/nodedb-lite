// SPDX-License-Identifier: Apache-2.0

//! Per-collection text configuration for Lite's FTS indexes.
//!
//! Mirrors Origin's `TextOp::SetTextConfig`: the analyzer name and the default
//! fuzzy-matching flag are persisted into each index's backend metadata via
//! `FtsIndex::set_collection_analyzer` / `FtsIndex::set_collection_fuzzy`, so
//! `analyze_for_collection` resolves the analyzer for every later tokenization
//! of that collection's text — indexing and query-time scoring alike — and
//! `FtsIndex::search` falls back to fuzzy matching for the collection even when
//! the query did not ask for it.
//!
//! Lite shards one collection across several `FtsIndex` instances — a
//! whole-document index keyed `"{collection}"` plus one per indexed field
//! keyed `"{collection}:{field}"` — and passes that composite key as the
//! `collection` argument to nodedb-fts. Each setting therefore has to be bound
//! on every one of a collection's indexes under its own key, including indexes
//! that do not exist yet: DDL normally runs before any document is written, so
//! the values are also retained in `collection_analyzers` /
//! `collection_fuzzy_defaults` and applied to each index at creation time.

use super::manager::registry::fts_err;
use super::manager::{FtsCollectionManager, resident_index};
use crate::engine::fts::LiteFtsIndex;
use crate::error::LiteError;

impl FtsCollectionManager {
    /// Bind `analyzer_name` to every index belonging to `collection`, and
    /// retain it so indexes created later inherit the same analyzer.
    ///
    /// Unrecognized names fall back to the standard analyzer inside
    /// nodedb-fts at resolve time, matching Origin's behavior. Fails when an
    /// index cannot record the binding.
    pub fn set_collection_analyzer(
        &mut self,
        collection: &str,
        analyzer_name: &str,
    ) -> Result<(), LiteError> {
        self.collection_analyzers
            .insert(collection.to_string(), analyzer_name.to_string());
        for key in self.collection_keys(collection) {
            if let Some(idx) = self.indices.get(&key) {
                idx.set_collection_analyzer(0, 0, &key, analyzer_name)
                    .map_err(|e| fts_err(collection, e))?;
            }
        }
        Ok(())
    }

    /// Bind the default fuzzy-matching flag to every index belonging to
    /// `collection`, and retain it so indexes created later inherit it.
    /// Fails when an index cannot record the binding.
    pub fn set_collection_fuzzy(&mut self, collection: &str, fuzzy: bool) -> Result<(), LiteError> {
        self.collection_fuzzy_defaults
            .insert(collection.to_string(), fuzzy);
        for key in self.collection_keys(collection) {
            if let Some(idx) = self.indices.get(&key) {
                idx.set_collection_fuzzy(0, 0, &key, fuzzy)
                    .map_err(|e| fts_err(collection, e))?;
            }
        }
        Ok(())
    }

    /// Analyzer bound to `collection`, if any.
    pub(crate) fn analyzer_for(&self, collection: &str) -> Option<&str> {
        self.collection_analyzers
            .get(collection)
            .map(String::as_str)
    }

    /// Default fuzzy-matching flag bound to `collection`, if any.
    pub(crate) fn fuzzy_for(&self, collection: &str) -> Option<bool> {
        self.collection_fuzzy_defaults.get(collection).copied()
    }

    /// Create the index for `key` of `collection`, applying the collection's
    /// bound text config.
    ///
    /// Used at every index-creation site so an analyzer or fuzzy default bound
    /// before the first write is not silently lost for indexes materialized
    /// afterwards. Fails when the new index cannot record the binding.
    pub(crate) fn new_index_for(
        &self,
        collection: &str,
        key: &str,
    ) -> Result<LiteFtsIndex, LiteError> {
        let idx = resident_index(std::sync::Arc::clone(&self.governor));
        if let Some(name) = self.analyzer_for(collection) {
            idx.set_collection_analyzer(0, 0, key, name)
                .map_err(|e| fts_err(collection, e))?;
        }
        if let Some(fuzzy) = self.fuzzy_for(collection) {
            idx.set_collection_fuzzy(0, 0, key, fuzzy)
                .map_err(|e| fts_err(collection, e))?;
        }
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::FtsCollectionManager;
    use crate::engine::fts::manager::test_governor;

    const DOC_KEY: &str = "col";

    /// Read back the analyzer and fuzzy default persisted on the whole-document
    /// index of `col`.
    fn doc_index_config(mgr: &FtsCollectionManager) -> (Option<String>, bool) {
        let idx = mgr
            .indices
            .get(DOC_KEY)
            .expect("whole-document index must exist");
        (
            idx.get_collection_analyzer(0, 0, DOC_KEY)
                .expect("meta read must succeed"),
            idx.get_collection_fuzzy(0, 0, DOC_KEY)
                .expect("meta read must succeed"),
        )
    }

    #[test]
    fn setting_analyzer_leaves_fuzzy_default_unchanged() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");

        mgr.set_collection_fuzzy("col", true)
            .expect("config binding must succeed");
        mgr.set_collection_analyzer("col", "german")
            .expect("config binding must succeed");

        let (analyzer, fuzzy) = doc_index_config(&mgr);
        assert_eq!(analyzer.as_deref(), Some("german"));
        assert!(
            fuzzy,
            "binding the analyzer must not clear the fuzzy default"
        );
        assert_eq!(mgr.fuzzy_for("col"), Some(true));
    }

    #[test]
    fn setting_fuzzy_default_leaves_analyzer_unchanged() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");

        mgr.set_collection_analyzer("col", "german")
            .expect("config binding must succeed");
        mgr.set_collection_fuzzy("col", true)
            .expect("config binding must succeed");

        let (analyzer, fuzzy) = doc_index_config(&mgr);
        assert_eq!(
            analyzer.as_deref(),
            Some("german"),
            "binding the fuzzy default must not clear the analyzer"
        );
        assert!(fuzzy);
        assert_eq!(mgr.analyzer_for("col"), Some("german"));
    }

    #[test]
    fn config_bound_before_any_index_is_inherited_by_later_indexes() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        // DDL order: config first, documents afterwards — no index exists yet.
        mgr.set_collection_analyzer("col", "german")
            .expect("config binding must succeed");
        mgr.set_collection_fuzzy("col", true)
            .expect("config binding must succeed");
        assert!(mgr.indices.is_empty());

        mgr.index_document("col", "doc1", "der schnelle braune fuchs")
            .expect("index update must succeed");

        let (analyzer, fuzzy) = doc_index_config(&mgr);
        assert_eq!(analyzer.as_deref(), Some("german"));
        assert!(fuzzy);
    }

    #[test]
    fn config_bound_before_any_index_is_inherited_by_later_field_indexes() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.set_collection_analyzer("col", "german")
            .expect("config binding must succeed");
        mgr.set_collection_fuzzy("col", true)
            .expect("config binding must succeed");

        mgr.index_field("col", "title", "doc1", "der schnelle braune fuchs")
            .expect("index update must succeed");

        let key = "col:title";
        let idx = mgr.indices.get(key).expect("field index must exist");
        assert_eq!(
            idx.get_collection_analyzer(0, 0, key)
                .expect("meta read must succeed")
                .as_deref(),
            Some("german")
        );
        assert!(
            idx.get_collection_fuzzy(0, 0, key)
                .expect("meta read must succeed")
        );
    }

    #[test]
    fn binding_nothing_leaves_the_collection_at_its_defaults() {
        // The `SetTextConfig { analyzer_name: None, fuzzy_default: None }`
        // case: neither setter runs, so nothing is retained or persisted.
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.index_document("col", "doc1", "the quick brown fox")
            .expect("index update must succeed");

        let (analyzer, fuzzy) = doc_index_config(&mgr);
        assert_eq!(analyzer, None);
        assert!(!fuzzy);
        assert_eq!(mgr.analyzer_for("col"), None);
        assert_eq!(mgr.fuzzy_for("col"), None);
    }

    #[test]
    fn config_is_bound_to_the_exact_collection_name() {
        let mut mgr = FtsCollectionManager::new(test_governor());
        mgr.set_collection_analyzer("a:b", "german")
            .expect("config binding must succeed");
        mgr.set_collection_fuzzy("a:b", true)
            .expect("config binding must succeed");

        assert_eq!(mgr.analyzer_for("a:b"), Some("german"));
        assert_eq!(mgr.fuzzy_for("a:b"), Some(true));
        assert_eq!(mgr.analyzer_for("a"), None);
        assert_eq!(mgr.fuzzy_for("a"), None);
    }
}
