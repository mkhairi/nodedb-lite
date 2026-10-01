// SPDX-License-Identifier: BUSL-1.1

//! In-memory field indexes over schemaless document collections.
//!
//! `CREATE INDEX` on a schemaless collection persists only the index spec.
//! The postings — value key to the ids of the documents holding it — are
//! derived from the Loro documents. They are built in one pass when the index
//! is registered: at open after the CRDT state is restored, and on
//! `CREATE INDEX`. Every write the engine applies then moves its document
//! from the old value's posting to the new one. A path that rewrites a
//! collection's state wholesale (import, compaction, peer-id rotation)
//! rebuilds that collection's postings.
//!
//! Value keys come from [`loro_value_to_index_key`], the stringification the
//! SQL index lookup also uses. A document whose field is absent, null, binary
//! or a container is not indexed under that field.
//!
//! Memory per (document, field) pair: one `String` doc id in a `BTreeSet`,
//! about 24 bytes inline plus the id's heap allocation plus B-tree node slack,
//! roughly 80–100 bytes. Each distinct value adds its key string, a map slot
//! and at least one B-tree leaf node of about 300 bytes. A field with mostly
//! unique values therefore costs about 400 bytes per document.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use loro::LoroValue;
use nodedb_crdt::CrdtState;

use crate::query::document_ops::indexes::bare_path;
use crate::query::value_utils::loro_value_to_index_key;

use super::types::CrdtEngine;

/// One indexed field: value key → ids of the documents holding it.
#[derive(Debug)]
struct FieldPostings {
    /// Keys are lowercased, for a `case_insensitive` index.
    case_insensitive: bool,
    ids_by_key: HashMap<String, BTreeSet<String>>,
}

impl FieldPostings {
    fn insert(&mut self, key: String, doc_id: &str) {
        self.ids_by_key
            .entry(key)
            .or_default()
            .insert(doc_id.to_string());
    }

    fn remove(&mut self, key: &str, doc_id: &str) {
        if let Some(ids) = self.ids_by_key.get_mut(key) {
            ids.remove(doc_id);
            if ids.is_empty() {
                self.ids_by_key.remove(key);
            }
        }
    }
}

/// The value keys one document holds, one slot per indexed field of its
/// collection, in field order. `None` means the field is not indexed for it.
pub(in crate::engine::crdt) type IndexedKeys = Vec<Option<String>>;

/// Every field index, per collection, per canonical field (`$.scope`).
#[derive(Debug, Default)]
pub(in crate::engine::crdt) struct FieldIndexes {
    by_collection: HashMap<String, BTreeMap<String, FieldPostings>>,
}

impl FieldIndexes {
    /// Keys `doc_id` holds in `state`, or `None` when `collection` has no
    /// field index.
    fn doc_keys(
        &self,
        state: Option<&CrdtState>,
        collection: &str,
        doc_id: &str,
    ) -> Option<IndexedKeys> {
        let fields = self.by_collection.get(collection)?;
        let row = state.and_then(|s| s.read_row(collection, doc_id));
        Some(
            fields
                .iter()
                .map(|(field, postings)| field_key(row.as_ref(), field, postings.case_insensitive))
                .collect(),
        )
    }

    /// Move `doc_id` from the keys it held (`before`) to the keys it holds
    /// in `state` now.
    fn move_doc(
        &mut self,
        state: Option<&CrdtState>,
        collection: &str,
        doc_id: &str,
        before: IndexedKeys,
    ) {
        let Some(after) = self.doc_keys(state, collection, doc_id) else {
            return;
        };
        let Some(fields) = self.by_collection.get_mut(collection) else {
            return;
        };
        for ((postings, old), new) in fields.values_mut().zip(before).zip(after) {
            if old == new {
                continue;
            }
            if let Some(old) = old {
                postings.remove(&old, doc_id);
            }
            if let Some(new) = new {
                postings.insert(new, doc_id);
            }
        }
    }

    /// Replace every field index of `collection` with one pass over the
    /// documents in `state`.
    fn rebuild(&mut self, state: Option<&CrdtState>, collection: &str) {
        let Some(fields) = self.by_collection.get_mut(collection) else {
            return;
        };
        for postings in fields.values_mut() {
            postings.ids_by_key = HashMap::new();
        }
        let Some(state) = state else {
            return;
        };
        for doc_id in state.row_ids(collection) {
            let row = state.read_row(collection, &doc_id);
            for (field, postings) in fields.iter_mut() {
                if let Some(key) = field_key(row.as_ref(), field, postings.case_insensitive) {
                    postings.insert(key, &doc_id);
                }
            }
        }
    }
}

/// The key `row` holds for `field`, read as the collection scan reads it: a
/// top-level key of the document map.
fn field_key(row: Option<&LoroValue>, field: &str, case_insensitive: bool) -> Option<String> {
    let LoroValue::Map(map) = row? else {
        return None;
    };
    let key = loro_value_to_index_key(map.get(bare_path(field))?)?;
    Some(if case_insensitive {
        key.to_lowercase()
    } else {
        key
    })
}

impl CrdtEngine {
    /// Register a field index on `collection` and build its postings from the
    /// collection's current documents. `field` is in canonical JSON-path form
    /// (`$.scope`). Registering an index again rebuilds it.
    pub fn register_field_index(&mut self, collection: &str, field: &str, case_insensitive: bool) {
        self.add_field_index(collection, field, case_insensitive);
        self.rebuild_field_indexes(collection);
    }

    /// Register an empty field index on `collection`. The caller rebuilds the
    /// collection's postings once after registering all its fields.
    pub(crate) fn add_field_index(
        &mut self,
        collection: &str,
        field: &str,
        case_insensitive: bool,
    ) {
        self.field_indexes
            .by_collection
            .entry(collection.to_string())
            .or_default()
            .insert(
                field.to_string(),
                FieldPostings {
                    case_insensitive,
                    ids_by_key: HashMap::new(),
                },
            );
    }

    /// Remove the field index on `(collection, field)`, if registered.
    pub fn drop_field_index(&mut self, collection: &str, field: &str) {
        if let Some(fields) = self.field_indexes.by_collection.get_mut(collection) {
            fields.remove(field);
            if fields.is_empty() {
                self.field_indexes.by_collection.remove(collection);
            }
        }
    }

    /// Remove every field index on `collection`.
    pub fn drop_field_indexes(&mut self, collection: &str) {
        self.field_indexes.by_collection.remove(collection);
    }

    /// Whether a field index with built postings covers `(collection, field)`.
    pub fn has_field_index(&self, collection: &str, field: &str) -> bool {
        self.field_indexes
            .by_collection
            .get(collection)
            .is_some_and(|fields| fields.contains_key(field))
    }

    /// Ids of the documents whose `field` holds `key`, ascending.
    ///
    /// `None` when no field index covers `(collection, field)`. `key` is the
    /// [`loro_value_to_index_key`] form of the value looked up.
    pub fn field_index_lookup(
        &self,
        collection: &str,
        field: &str,
        key: &str,
    ) -> Option<Vec<String>> {
        let postings = self
            .field_indexes
            .by_collection
            .get(collection)?
            .get(field)?;
        let key = if postings.case_insensitive {
            Cow::Owned(key.to_lowercase())
        } else {
            Cow::Borrowed(key)
        };
        Some(
            postings
                .ids_by_key
                .get(key.as_ref())
                .map(|ids| ids.iter().cloned().collect())
                .unwrap_or_default(),
        )
    }

    /// Keys `doc_id` holds now, read before a write so [`Self::reindex_doc`]
    /// can move it afterwards. `None` when the collection has no field index.
    pub(in crate::engine::crdt) fn indexed_keys(
        &self,
        collection: &str,
        doc_id: &str,
    ) -> Option<IndexedKeys> {
        self.field_indexes
            .doc_keys(self.states.get(collection), collection, doc_id)
    }

    /// Move `doc_id` from the keys read by [`Self::indexed_keys`] before a
    /// write to the keys it holds now.
    pub(in crate::engine::crdt) fn reindex_doc(
        &mut self,
        collection: &str,
        doc_id: &str,
        before: Option<IndexedKeys>,
    ) {
        let Some(before) = before else {
            return;
        };
        self.field_indexes
            .move_doc(self.states.get(collection), collection, doc_id, before);
    }

    /// Rebuild every field index of `collection` from its documents. A no-op
    /// for a collection with no field index.
    pub(crate) fn rebuild_field_indexes(&mut self, collection: &str) {
        self.field_indexes
            .rebuild(self.states.get(collection), collection);
    }

    /// Rebuild the field indexes of every collection.
    pub(in crate::engine::crdt) fn rebuild_all_field_indexes(&mut self) {
        let collections: Vec<String> = self.field_indexes.by_collection.keys().cloned().collect();
        for collection in &collections {
            self.rebuild_field_indexes(collection);
        }
    }
}

#[cfg(test)]
impl CrdtEngine {
    /// Panic unless every field index equals a fresh full-scan derivation of
    /// its collection's documents.
    pub(crate) fn assert_field_indexes_consistent(&self) {
        for (collection, fields) in &self.field_indexes.by_collection {
            let mut fresh = FieldIndexes::default();
            let fresh_fields = fresh.by_collection.entry(collection.clone()).or_default();
            for (field, postings) in fields {
                fresh_fields.insert(
                    field.clone(),
                    FieldPostings {
                        case_insensitive: postings.case_insensitive,
                        ids_by_key: HashMap::new(),
                    },
                );
            }
            fresh.rebuild(self.states.get(collection), collection);
            for (field, postings) in fields {
                let expected = &fresh.by_collection[collection.as_str()][field.as_str()];
                assert_eq!(
                    postings.ids_by_key, expected.ids_by_key,
                    "postings of {collection} {field} differ from a full scan"
                );
            }
        }
    }
}
