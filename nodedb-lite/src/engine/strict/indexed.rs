// SPDX-License-Identifier: Apache-2.0

//! Strict rows and the secondary-index catalog.
//!
//! Every strict row write commits through [`StrictEngine::commit`], which
//! adds the index entries the written rows imply to the same storage batch.
//! An index entry names a row by the hex of its storage key after
//! `{collection}:`, which reads the row back whatever its primary key type.

use std::collections::HashMap;
use std::sync::Arc;

use nodedb_types::Namespace;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::index::IndexCatalog;
use crate::index::IndexEngine;
use crate::index::durable::RowImage;
use crate::index::key::{doc_id_of_key, key_of_doc_id};
use crate::storage::engine::{StorageEngine, WriteOp};

use super::crud::decode_tuple;
use super::engine::{CollectionState, StrictEngine};

/// A strict row as a document value: column name to value.
pub(crate) fn row_object(state: &CollectionState, values: &[Value]) -> Value {
    Value::Object(
        state
            .schema
            .columns
            .iter()
            .zip(values)
            .map(|(col, v)| (col.name.clone(), v.clone()))
            .collect::<HashMap<_, _>>(),
    )
}

impl<S: StorageEngine> StrictEngine<S> {
    /// Maintain `catalog`'s strict indexes from every row write from now on.
    pub fn set_index_catalog(&self, catalog: Arc<IndexCatalog>) {
        // Set once, when the query engine is built; a second call keeps the
        // first catalog.
        let _ = self.indexes.set(catalog);
    }

    /// The document id an index entry names the row stored at `key` by.
    fn doc_id_of(collection: &str, key: &[u8]) -> Option<String> {
        key.strip_prefix(collection.as_bytes())
            .and_then(|rest| rest.strip_prefix(b":"))
            .map(doc_id_of_key)
    }

    /// The row an index entry names, as a document value.
    pub(crate) async fn row_by_doc_id(
        &self,
        collection: &str,
        doc_id: &str,
    ) -> Result<Option<Vec<Value>>, LiteError> {
        let state = self.get_state(collection)?;
        let Some(suffix) = key_of_doc_id(doc_id) else {
            return Ok(None);
        };
        let mut key = collection.as_bytes().to_vec();
        key.push(b':');
        key.extend_from_slice(&suffix);
        match self.storage.get(Namespace::Strict, &key).await? {
            Some(bytes) => decode_tuple(&state, &bytes).map(Some),
            None => Ok(None),
        }
    }

    /// Every row of `collection` with the document id its entries carry.
    pub(crate) async fn rows_with_ids(
        &self,
        collection: &str,
    ) -> Result<Vec<(String, Value)>, LiteError> {
        let state = self.get_state(collection)?;
        let prefix = format!("{collection}:");
        let mut rows = Vec::new();
        for (key, bytes) in self
            .storage
            .scan_prefix(Namespace::Strict, prefix.as_bytes())
            .await?
        {
            if let Some(id) = Self::doc_id_of(collection, &key) {
                let values = decode_tuple(&state, &bytes)?;
                rows.push((id, row_object(&state, &values)));
            }
        }
        Ok(rows)
    }

    /// Commit strict row writes `ops` of `collection` — puts of encoded
    /// tuples and deletes — with the index entries they imply, in one storage
    /// batch. A write a unique index refuses commits nothing.
    pub(crate) async fn commit(
        &self,
        collection: &str,
        ops: Vec<WriteOp>,
    ) -> Result<(), LiteError> {
        let Some(catalog) = self.indexes.get() else {
            return self.storage.batch_write(&ops).await;
        };
        let state = self.get_state(collection)?;
        let mut images = Vec::new();
        for op in &ops {
            let (key, row) = match op {
                WriteOp::Put { ns, key, value } if *ns == Namespace::Strict => {
                    let values = decode_tuple(&state, value)?;
                    (key, Some(row_object(&state, &values)))
                }
                WriteOp::Delete { ns, key } if *ns == Namespace::Strict => (key, None),
                _ => continue,
            };
            if let Some(doc_id) = Self::doc_id_of(collection, key) {
                images.push(RowImage {
                    collection: collection.to_string(),
                    doc_id,
                    row,
                });
            }
        }
        catalog
            .commit_rows(
                &*self.storage,
                IndexEngine::Strict,
                ops,
                images,
                |coll, id| async move {
                    let state = self.get_state(&coll)?;
                    Ok(self
                        .row_by_doc_id(&coll, &id)
                        .await?
                        .map(|values| row_object(&state, &values)))
                },
            )
            .await
    }
}
