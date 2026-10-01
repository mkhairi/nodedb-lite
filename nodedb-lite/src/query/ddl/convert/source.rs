// SPDX-License-Identifier: Apache-2.0
use nodedb_types::columnar::{ColumnDef, StrictSchema};
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::index::IndexEngine;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::schema::document_to_row;

impl<S: StorageEngine> LiteQueryEngine<S> {
    /// Re-declare the indexes of `collection` over `engine` rows and build
    /// them from those rows: a conversion that failed leaves its source
    /// indexed as it was.
    pub(in crate::query::ddl::convert) async fn restore_indexes(
        &self,
        collection: &str,
        engine: IndexEngine,
    ) -> Result<(), LiteError> {
        for def in self
            .indexes
            .move_collection(&*self.storage, collection, engine)
            .await?
        {
            crate::query::document_ops::indexes::rebuild_index(self, def).await?;
        }
        Ok(())
    }

    /// Decode every stored tuple of the strict collection `collection`.
    /// A tuple that does not decode fails the read: skipping it would lose
    /// the row.
    pub(in crate::query::ddl::convert) async fn decode_strict_rows(
        &self,
        collection: &str,
        schema: &StrictSchema,
    ) -> Result<Vec<Vec<Value>>, LiteError> {
        let raw = self.strict.scan_raw(collection).await?;
        let decoder = nodedb_strict::TupleDecoder::new(schema);
        raw.iter()
            .map(|tuple_bytes| {
                decoder
                    .extract_all(tuple_bytes)
                    .map_err(|e| LiteError::Corrupted {
                        detail: format!("strict tuple of '{collection}' does not decode: {e}"),
                    })
            })
            .collect()
    }

    /// Read rows from any source (CRDT or strict) as Vec<Value>.
    pub(in crate::query::ddl::convert) async fn read_source_rows(
        &self,
        collection: &str,
        target_columns: &[ColumnDef],
    ) -> Result<Vec<Vec<Value>>, LiteError> {
        // Try CRDT first.
        {
            let crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
            let ids = crdt.list_ids(collection);
            if !ids.is_empty() {
                return Ok(ids
                    .iter()
                    .filter_map(|id| {
                        crdt.read(collection, id).map(|loro_val| {
                            let doc = crate::nodedb::convert::loro_value_to_document(id, &loro_val);
                            document_to_row(&doc.fields, target_columns)
                        })
                    })
                    .collect());
            }
        }

        // Try strict.
        if let Some(schema) = self.strict.schema(collection) {
            return self.decode_strict_rows(collection, &schema).await;
        }

        Err(LiteError::Query(format!(
            "collection '{collection}' not found in any storage mode"
        )))
    }
}
