// SPDX-License-Identifier: Apache-2.0

//! Secondary-index upkeep on every row write.
//!
//! Every path that changes a row — a local upsert, merge or delete, a batch, a
//! deferred write, a list edit, an imported remote delta, a rolled-back
//! rejection — passes through this engine. Hooking the index here keeps it
//! current for all of them, under the same borrow that makes the row change,
//! so no write path can skip it and a flush never sees one without the other.

use std::collections::HashMap;
use std::sync::Arc;

use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::index::IndexCatalog;
use crate::index::document::{predicted_row, row_value};

use super::types::{CrdtEngine, CrdtRowOp};

/// Predicted rows of one collection: `(doc_id, row after the write)`.
type PredictedRows<'a> = Vec<(&'a str, Option<Value>)>;

impl CrdtEngine {
    /// Maintain `catalog`'s document indexes from every row write from now on.
    pub fn set_index_catalog(&mut self, catalog: Arc<IndexCatalog>) {
        self.indexes = Some(catalog);
    }

    /// A document's current contents.
    fn current_row(&self, collection: &str, doc_id: &str) -> Option<Value> {
        self.read(collection, doc_id).map(|row| row_value(&row))
    }

    /// Refuse the row writes `rows`, before any is applied, when one would
    /// give a unique index a value another document holds. Rows are predicted
    /// in order, so a merge onto a row written earlier in the same batch sees
    /// that write.
    pub fn check_unique_writes(&self, rows: &[CrdtRowOp<'_>]) -> Result<(), LiteError> {
        let Some(catalog) = &self.indexes else {
            return Ok(());
        };
        let mut predicted: HashMap<(&str, &str), Value> = HashMap::new();
        let mut by_collection: Vec<(&str, PredictedRows<'_>)> = Vec::new();
        for &(mode, collection, doc_id, fields) in rows {
            if !catalog.is_indexed(collection) {
                continue;
            }
            let base = predicted
                .get(&(collection, doc_id))
                .cloned()
                .or_else(|| self.current_row(collection, doc_id));
            let row = predicted_row(base, mode, fields);
            predicted.insert((collection, doc_id), row.clone());
            match by_collection.iter_mut().find(|(c, _)| *c == collection) {
                Some((_, list)) => list.push((doc_id, Some(row))),
                None => by_collection.push((collection, vec![(doc_id, Some(row))])),
            }
        }
        for (collection, list) in by_collection {
            catalog.check_unique(collection, &list, |id| self.current_row(collection, id))?;
        }
        Ok(())
    }

    /// Bring the index entries of `doc_ids` in line with their current rows.
    pub(in crate::engine::crdt) fn sync_indexes<'a>(
        &mut self,
        collection: &str,
        doc_ids: impl IntoIterator<Item = &'a str>,
    ) {
        let ids: Vec<&str> = doc_ids.into_iter().collect();
        for id in &ids {
            self.reconcile_live_id(collection, id);
        }
        if let Some(catalog) = &self.indexes {
            catalog.resync(collection, ids, |id| self.current_row(collection, id));
        }
    }

    /// Remove every index entry of `collection`, keeping the definitions.
    pub(in crate::engine::crdt) fn clear_index_entries(&self, collection: &str) {
        if let Some(catalog) = &self.indexes {
            catalog.clear_document_entries(collection);
        }
    }
}
