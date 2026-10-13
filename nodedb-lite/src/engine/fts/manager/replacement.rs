// SPDX-License-Identifier: Apache-2.0

//! Private collection candidates reserve IDs from the live manager.

use nodedb_mem::MemoryGovernor;
use nodedb_types::{Surrogate, Value};
use std::collections::HashMap;
use std::sync::Arc;

use super::registry::{
    FtsCollectionManager, fts_err, index_key, key_of_collection, resident_index,
};
use crate::engine::fts::{LiteFtsIndex, catalog::SearchDeclarationRecord};
use crate::error::LiteError;

pub(crate) struct CollectionReplacement {
    collection: String,
    record: SearchDeclarationRecord,
    indices: HashMap<String, LiteFtsIndex>,
    governor: Arc<MemoryGovernor>,
    retained: HashMap<String, super::retained::RetainedIndex>,
}

impl FtsCollectionManager {
    pub(crate) fn begin_replacement(
        &self,
        collection: &str,
        record: SearchDeclarationRecord,
    ) -> Result<CollectionReplacement, LiteError> {
        if record.revision != 0 || record.declaration.is_some() {
            record.check(collection)?;
        }
        let mut replacement = CollectionReplacement {
            collection: collection.into(),
            record,
            indices: HashMap::new(),
            governor: Arc::clone(&self.governor),
            retained: HashMap::new(),
        };
        let fields = replacement
            .record
            .declaration
            .as_ref()
            .map(|declaration| declaration.fields.clone());
        if let Some(fields) = fields {
            replacement.ensure_index("")?;
            for field in fields {
                replacement.ensure_index(&field)?;
            }
        }
        Ok(replacement)
    }

    pub(crate) fn reserve_surrogate(
        &mut self,
        collection: &str,
        id: &str,
    ) -> Result<Surrogate, LiteError> {
        self.surrogate_for(collection, id)
    }

    /// Drop `collection`'s resident postings and leases before an open-time
    /// rebuild replaces them. Nothing serves the collection until the
    /// replacement publishes, so the old leases must not stack on the new ones.
    pub(crate) fn release_for_rebuild(&mut self, collection: &str) {
        let system = index_key(collection, "_geohash");
        self.indices
            .retain(|key, _| !key_of_collection(key, collection) || key == &system);
        self.retained
            .retain(|key, _| !key_of_collection(key, collection) || key == &system);
    }

    pub(crate) fn publish_replacement(&mut self, replacement: CollectionReplacement) {
        let CollectionReplacement {
            collection,
            record,
            indices,
            retained,
            ..
        } = replacement;
        let system = index_key(&collection, "_geohash");
        let removed: Vec<String> = self
            .indices
            .keys()
            .filter(|key| key_of_collection(key, &collection) && *key != &system)
            .filter(|key| !indices.contains_key(*key))
            .cloned()
            .collect();
        let published: Vec<String> = indices.keys().cloned().collect();
        self.indices
            .retain(|key, _| !key_of_collection(key, &collection) || key == &system);
        self.retained
            .retain(|key, _| !key_of_collection(key, &collection) || key == &system);
        self.indices.extend(indices);
        self.retained.extend(retained);
        // A published index replaces the stored form under its key; one the
        // replacement left out is deleted by the next flush.
        for key in published {
            self.mark_created(&key);
        }
        for key in removed {
            self.mark_removed(&key);
        }
        self.remove_declaration_policy(&collection);
        if let Some(declaration) = &record.declaration {
            self.collection_analyzers
                .insert(collection.clone(), declaration.analyzer.clone());
            self.collection_fuzzy_defaults
                .insert(collection.clone(), declaration.fuzzy);
        }
        if record.revision != 0 {
            self.declarations.insert(collection, record);
        }
    }
}

impl CollectionReplacement {
    pub(crate) fn index_document_fields(
        &mut self,
        _id: &str,
        surrogate: Surrogate,
        fields: &HashMap<String, Value>,
    ) -> Result<(), LiteError> {
        let texts = super::fields::selected_texts(fields, self.record.declaration.as_ref());
        let whole = super::fields::joined_text(&texts);
        self.index_text("", surrogate, &whole)?;
        let per_field = self
            .record
            .declaration
            .as_ref()
            .is_none_or(|declaration| declaration.per_field);
        for (field, text) in texts {
            if per_field && !field.is_empty() {
                self.index_text(field, surrogate, text)?;
            }
        }
        Ok(())
    }

    fn index_text(
        &mut self,
        field: &str,
        surrogate: Surrogate,
        text: &str,
    ) -> Result<(), LiteError> {
        if text.is_empty() {
            return Ok(());
        }
        self.ensure_index(field)?;
        let key = index_key(&self.collection, field);
        let tokens = self
            .indices
            .get(&key)
            .ok_or_else(|| {
                fts_err(
                    &self.collection,
                    format!("candidate index '{key}' is absent"),
                )
            })?
            .analyze_for_collection(0, 0, &key, text)
            .map_err(|error| fts_err(&self.collection, error))?;
        self.retained
            .get_mut(&key)
            .ok_or_else(|| {
                fts_err(
                    &self.collection,
                    format!("candidate lease '{key}' is absent"),
                )
            })?
            .reserve_write(Arc::clone(&self.governor), &key, surrogate, &tokens)?;
        self.indices
            .get(&key)
            .ok_or_else(|| {
                fts_err(
                    &self.collection,
                    format!("candidate index '{key}' is absent"),
                )
            })?
            .index_analyzed_document(0, 0, &key, surrogate, &tokens)
            .map_err(|error| fts_err(&self.collection, error))
    }

    fn ensure_index(&mut self, field: &str) -> Result<(), LiteError> {
        let key = index_key(&self.collection, field);
        if !self.indices.contains_key(&key) {
            let mut retained = super::retained::RetainedIndex::default();
            retained.reserve_base(Arc::clone(&self.governor), &key)?;
            let index = resident_index(Arc::clone(&self.governor));
            let (analyzer, fuzzy) = self
                .record
                .declaration
                .as_ref()
                .map_or(("standard", false), |declaration| {
                    (declaration.analyzer.as_str(), declaration.fuzzy)
                });
            index
                .set_collection_analyzer(0, 0, &key, analyzer)
                .map_err(|error| fts_err(&self.collection, error))?;
            index
                .set_collection_fuzzy(0, 0, &key, fuzzy)
                .map_err(|error| fts_err(&self.collection, error))?;
            self.indices.insert(key.clone(), index);
            self.retained.insert(key, retained);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::fts::manager::test_governor;
    use nodedb_types::text_search::TextSearchParams;

    #[test]
    fn unpublished_candidates_leave_prior_postings_and_surrogate_reservations_intact() {
        let mut manager = FtsCollectionManager::new(test_governor());
        manager.index_document("docs", "old", "alpha").unwrap();
        manager.index_document("other", "other", "gamma").unwrap();
        let record = manager.next_declaration_record("docs", &None).unwrap();
        let mut candidate = manager.begin_replacement("docs", record).unwrap();
        let reserved = manager.reserve_surrogate("docs", "new").unwrap();
        candidate
            .index_document_fields(
                "new",
                reserved,
                &HashMap::from([("title".into(), Value::String("beta".into()))]),
            )
            .unwrap();
        assert_eq!(manager.reserve_surrogate("docs", "new").unwrap(), reserved);
        assert_eq!(
            manager
                .search("docs", "", "alpha", 10, &TextSearchParams::default())
                .unwrap()
                .len(),
            1
        );
        manager.publish_replacement(candidate);
        assert!(
            manager
                .search("docs", "", "alpha", 10, &TextSearchParams::default())
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            manager
                .search("docs", "", "beta", 10, &TextSearchParams::default())
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            manager
                .search("other", "", "gamma", 10, &TextSearchParams::default())
                .unwrap()
                .len(),
            1
        );
    }
    fn allocated(governor: &MemoryGovernor) -> usize {
        governor
            .snapshot()
            .into_iter()
            .find(|entry| entry.engine == nodedb_mem::EngineId::Fts)
            .unwrap()
            .allocated
    }

    /// The memtable charges the governor for the postings it holds, so the FTS
    /// engine's total must track them. A second, estimated lease on the same
    /// postings double-counts them.
    #[test]
    fn fts_charge_tracks_resident_postings_without_a_second_lease() {
        let governor = test_governor();
        let mut manager = FtsCollectionManager::new(Arc::clone(&governor));
        for i in 0..20_000 {
            manager
                .index_document(
                    "docs",
                    &format!("d{i}"),
                    &format!("alpha beta gamma delta epsilon zeta eta theta word{i}"),
                )
                .unwrap();
        }
        let resident: usize = manager
            .indices
            .values()
            .map(|index| index.memtable().resident_bytes())
            .sum();
        // The memtable moves its charge in 1 MiB steps, once per index.
        let slack = (manager.indices.len() + 1) << 20;
        assert!(
            allocated(&governor) <= resident + slack,
            "FTS charged {} B for {} B of resident postings",
            allocated(&governor),
            resident
        );
    }

    /// Postings leave the memtable when their document is removed, and so
    /// must their charge. A lease that survives removal keeps it forever.
    #[test]
    fn removing_every_document_releases_its_fts_charge() {
        let governor = test_governor();
        let mut manager = FtsCollectionManager::new(Arc::clone(&governor));
        manager.index_document("docs", "seed", "seed").unwrap();
        let base = allocated(&governor);
        for i in 0..20_000 {
            manager
                .index_document(
                    "docs",
                    &format!("d{i}"),
                    &format!("alpha beta gamma delta epsilon zeta eta theta word{i}"),
                )
                .unwrap();
        }
        let full = allocated(&governor);
        assert!(full > base);
        for i in 0..20_000 {
            manager.remove_document("docs", &format!("d{i}")).unwrap();
        }
        let after = allocated(&governor);
        assert!(
            after.saturating_sub(base) < (full - base) / 10,
            "charge after removing every document: {after} B (base {base} B, full {full} B)"
        );
    }

    #[test]
    fn candidate_charge_releases_on_drop_and_moves_into_resident_indices() {
        let governor = test_governor();
        let mut manager = FtsCollectionManager::new(Arc::clone(&governor));
        manager.index_document("docs", "id", "alpha").unwrap();
        let prior = allocated(&governor);
        // The memtable charges in 1 MiB steps, so the title carries enough
        // distinct terms to move the charge.
        let title: Vec<String> = (0..40_000).map(|i| format!("w{i}")).collect();
        let fields = HashMap::from([("title".into(), Value::String(title.join(" ")))]);
        let mut candidate = manager
            .begin_replacement("docs", manager.current_declaration_record("docs"))
            .unwrap();
        let surrogate = manager.reserve_surrogate("docs", "id").unwrap();
        candidate
            .index_document_fields("id", surrogate, &fields)
            .unwrap();
        assert!(allocated(&governor) > prior);
        drop(candidate);
        assert_eq!(allocated(&governor), prior);
        let mut candidate = manager
            .begin_replacement("docs", manager.current_declaration_record("docs"))
            .unwrap();
        candidate
            .index_document_fields("id", surrogate, &fields)
            .unwrap();
        manager.publish_replacement(candidate);
        let resident = allocated(&governor);
        for _ in 0..100 {
            manager
                .index_document_fields("docs", "id", &fields)
                .unwrap();
        }
        assert_eq!(allocated(&governor), resident);
        manager.drop_collection("docs");
        assert_eq!(allocated(&governor), 0);
    }

    #[test]
    fn candidate_and_live_whole_document_phrases_share_field_order() {
        let mut manager = FtsCollectionManager::new(test_governor());
        let fields = HashMap::from([
            ("z".into(), Value::String("beta".into())),
            ("a".into(), Value::String("alpha".into())),
        ]);
        let phrase = vec!["alpha".into(), "beta".into()];
        let params = TextSearchParams::default();
        manager
            .index_document_fields("docs", "id", &fields)
            .unwrap();
        assert_eq!(
            manager
                .phrase_search("docs", "", &phrase, 10, &params)
                .unwrap()
                .len(),
            1
        );
        for declaration in [
            Some(crate::engine::fts::catalog::SearchDeclaration {
                name: "fts_docs".into(),
                fields: vec!["z".into(), "a".into()],
                analyzer: "standard".into(),
                fuzzy: false,
                per_field: true,
            }),
            None,
        ] {
            let record = manager
                .next_declaration_record("docs", &declaration)
                .unwrap();
            let mut candidate = manager.begin_replacement("docs", record).unwrap();
            let surrogate = manager.reserve_surrogate("docs", "id").unwrap();
            candidate
                .index_document_fields("id", surrogate, &fields)
                .unwrap();
            manager.publish_replacement(candidate);
            assert_eq!(
                manager
                    .phrase_search("docs", "", &phrase, 10, &params)
                    .unwrap()
                    .len(),
                1
            );
            manager
                .index_document_fields("docs", "id", &fields)
                .unwrap();
            assert_eq!(
                manager
                    .phrase_search("docs", "", &phrase, 10, &params)
                    .unwrap()
                    .len(),
                1
            );
        }
    }
}
