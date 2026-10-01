// SPDX-License-Identifier: Apache-2.0

//! Declaration selection, revision bookkeeping, and default restoration.

use super::FtsCollectionManager;
use crate::engine::fts::catalog::{DECLARATION_FORMAT, SearchDeclaration, SearchDeclarationRecord};
use crate::error::LiteError;
use std::collections::BTreeMap;

impl FtsCollectionManager {
    pub(crate) fn declaration_for(&self, collection: &str) -> Option<&SearchDeclaration> {
        self.declarations.get(collection)?.declaration.as_ref()
    }

    pub(crate) fn declaration_revisions(&self) -> BTreeMap<String, u64> {
        self.declarations
            .iter()
            .map(|(collection, record)| (collection.clone(), record.revision))
            .collect()
    }

    pub(crate) fn next_declaration_record(
        &self,
        collection: &str,
        declaration: &Option<SearchDeclaration>,
    ) -> Result<SearchDeclarationRecord, LiteError> {
        let revision = self
            .declarations
            .get(collection)
            .map_or(0, |record| record.revision)
            .checked_add(1)
            .ok_or_else(|| LiteError::Backpressure {
                detail: format!(
                    "search revision space exhausted for '{collection}': recreate the store"
                ),
            })?;
        let record = SearchDeclarationRecord {
            format_version: DECLARATION_FORMAT,
            revision,
            declaration: declaration.clone(),
        };
        record.check(collection)?;
        Ok(record)
    }

    pub(crate) fn current_declaration_record(&self, collection: &str) -> SearchDeclarationRecord {
        self.declarations
            .get(collection)
            .cloned()
            .unwrap_or(SearchDeclarationRecord {
                format_version: DECLARATION_FORMAT,
                revision: 0,
                declaration: None,
            })
    }

    pub(crate) fn load_declarations(&mut self, records: BTreeMap<String, SearchDeclarationRecord>) {
        for (collection, record) in &records {
            if let Some(declaration) = &record.declaration {
                self.collection_analyzers
                    .insert(collection.clone(), declaration.analyzer.clone());
                self.collection_fuzzy_defaults
                    .insert(collection.clone(), declaration.fuzzy);
            } else {
                self.collection_analyzers.remove(collection);
                self.collection_fuzzy_defaults.remove(collection);
            }
        }
        self.declarations = records;
    }

    pub(crate) fn ensure_declaration_indices(&mut self) -> Result<(), LiteError> {
        for (collection, record) in &self.declarations {
            let Some(declaration) = &record.declaration else {
                continue;
            };
            for field in std::iter::once("").chain(declaration.fields.iter().map(String::as_str)) {
                let key = super::index_key(collection, field);
                if !self.indices.contains_key(&key) {
                    let mut retained = super::retained::RetainedIndex::default();
                    retained.reserve_base(std::sync::Arc::clone(&self.governor), &key)?;
                    let index = self.new_index_for(collection, &key)?;
                    self.indices.insert(key.clone(), index);
                    self.retained.insert(key.clone(), retained);
                }
                let index = self.indices.get(&key).ok_or_else(|| {
                    super::registry::fts_err(
                        collection,
                        format!("declared index '{key}' is absent"),
                    )
                })?;
                index
                    .set_collection_analyzer(0, 0, &key, &declaration.analyzer)
                    .map_err(|error| super::registry::fts_err(collection, error))?;
                index
                    .set_collection_fuzzy(0, 0, &key, declaration.fuzzy)
                    .map_err(|error| super::registry::fts_err(collection, error))?;
            }
        }
        Ok(())
    }

    pub(crate) fn remove_collection_postings(&mut self, collection: &str) {
        self.indices
            .retain(|key, _| !super::registry::key_of_collection(key, collection));
        self.retained
            .retain(|key, _| !super::registry::key_of_collection(key, collection));
    }

    pub(crate) fn remove_declaration_policy(&mut self, collection: &str) {
        self.collection_analyzers.remove(collection);
        self.collection_fuzzy_defaults.remove(collection);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::fts::manager::test_governor;
    use nodedb_types::{Value, text_search::TextSearchParams};
    use std::collections::HashMap;

    #[test]
    fn declared_fields_replace_whole_document_and_preserve_system_postings() {
        let mut manager = FtsCollectionManager::new(test_governor());
        let fields = HashMap::from([
            ("title".into(), Value::String("alpha".into())),
            ("body".into(), Value::String("beta".into())),
        ]);
        manager
            .index_document_fields("docs", "id", &fields)
            .unwrap();
        manager
            .index_field("docs", "_geohash", "id", "u4pruy")
            .unwrap();
        let record = manager
            .next_declaration_record(
                "docs",
                &Some(SearchDeclaration {
                    name: "fts_docs".into(),
                    fields: vec!["title".into()],
                    analyzer: "standard".into(),
                    fuzzy: false,
                }),
            )
            .unwrap();
        manager.load_declarations(BTreeMap::from([("docs".into(), record)]));
        manager
            .index_document_fields("docs", "id", &fields)
            .unwrap();
        assert!(
            manager
                .search("docs", "", "beta", 10, &TextSearchParams::default())
                .unwrap()
                .is_empty()
        );
        assert!(
            manager
                .search("docs", "body", "beta", 10, &TextSearchParams::default())
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            manager
                .search("docs", "title", "alpha", 10, &TextSearchParams::default())
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            manager
                .search(
                    "docs",
                    "_geohash",
                    "u4pruy",
                    10,
                    &TextSearchParams::default()
                )
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            manager
                .next_declaration_record("docs", &None)
                .unwrap()
                .revision,
            2
        );
    }
    #[tokio::test]
    async fn empty_checkpoint_restores_declared_fuzzy_config_for_future_documents() {
        use crate::engine::fts::{catalog, checkpoint};
        use crate::storage::{engine::StorageEngine, pagedb_storage::PagedbStorageMem};
        use std::sync::Arc;
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let governor = test_governor();
        let mut manager = FtsCollectionManager::new(Arc::clone(&governor));
        let record = manager
            .next_declaration_record(
                "docs",
                &Some(SearchDeclaration {
                    name: "fts_docs".into(),
                    fields: vec!["title".into()],
                    analyzer: "standard".into(),
                    fuzzy: true,
                }),
            )
            .unwrap();
        catalog::persist_declaration(&storage, "docs", &record)
            .await
            .unwrap();
        manager.load_declarations(BTreeMap::from([("docs".into(), record)]));
        manager.ensure_declaration_indices().unwrap();
        let (indices, ids, next) = manager.checkpoint_data();
        let (operations, segments) = checkpoint::serialize_fts(indices, ids, next).unwrap();
        assert!(segments.is_empty());
        storage.batch_write(&operations).await.unwrap();
        drop(manager);
        let checkpoint = checkpoint::restore_fts(&storage, Arc::clone(&governor))
            .await
            .unwrap();
        for (key, index) in &checkpoint.indices {
            assert!(index.get_collection_fuzzy(0, 0, key).unwrap());
            index.set_collection_fuzzy(0, 0, key, false).unwrap();
            index.set_collection_analyzer(0, 0, key, "german").unwrap();
        }
        let mut restored = FtsCollectionManager::new(governor);
        restored.load_declarations(catalog::load_declarations(&storage).await.unwrap());
        restored.load_checkpoint(
            checkpoint.indices,
            checkpoint.id_to_surrogate,
            checkpoint.surrogate_to_id,
            checkpoint.next_surrogate,
        );
        restored.ensure_declaration_indices().unwrap();
        for (key, index) in &restored.indices {
            assert!(index.get_collection_fuzzy(0, 0, key).unwrap());
        }
        restored
            .index_document_fields(
                "docs",
                "future",
                &HashMap::from([("title".into(), Value::String("database".into()))]),
            )
            .unwrap();
        for field in ["", "title"] {
            let hits = restored
                .search("docs", field, "databse", 10, &TextSearchParams::default())
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "future");
            assert!(hits[0].fuzzy);
        }
    }
}
