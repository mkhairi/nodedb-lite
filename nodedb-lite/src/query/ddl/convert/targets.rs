// SPDX-License-Identifier: Apache-2.0
//! DDL handlers for CONVERT COLLECTION between storage modes.
//!
//! - CONVERT COLLECTION <name> TO strict (<col_defs>)
//! - CONVERT COLLECTION <name> TO columnar (<col_defs>)
//! - CONVERT COLLECTION <name> TO document

use nodedb_types::columnar::{ColumnarProfile, ColumnarSchema, StrictSchema};
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::index::IndexEngine;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::schema::{conversion_refused, document_to_row, parse_convert_sql};
use crate::nodedb::convert::value_to_loro;

impl<S: StorageEngine> LiteQueryEngine<S> {
    /// Handle: CONVERT COLLECTION <name> TO strict (<col_defs>)
    ///
    /// Reads all schemaless documents from the CRDT engine, validates each
    /// against the target schema, and writes as Binary Tuples in the strict engine.
    /// The original schemaless collection is dropped after successful conversion.
    pub(in crate::query) async fn handle_convert_to_strict(
        &self,
        sql: &str,
    ) -> Result<QueryResult, LiteError> {
        let (source_name, target_schema) = parse_convert_sql(sql, "strict")?;
        self.convert_to_strict(&source_name, target_schema).await
    }

    /// Convert the schemaless collection `source_name` to strict under
    /// `target_schema`. Rows come from the CRDT store, the only schemaless
    /// store Lite keeps, so the source format needs no plan-level hint.
    ///
    /// All or nothing: a document the strict engine refuses rolls the new
    /// strict collection back and leaves the schemaless collection as it was.
    pub(in crate::query) async fn convert_to_strict(
        &self,
        source_name: &str,
        target_schema: StrictSchema,
    ) -> Result<QueryResult, LiteError> {
        let guard = self.fts_state.admit_mutation().await;
        let result = self
            .convert_to_strict_admitted(guard.permit(), source_name, target_schema)
            .await;
        guard.finish(result)
    }

    pub(in crate::query) async fn convert_to_strict_admitted(
        &self,
        permit: &crate::engine::fts::coordinator::TextMutationPermit,
        source_name: &str,
        target_schema: StrictSchema,
    ) -> Result<QueryResult, LiteError> {
        let _permit = permit;
        crate::engine::fts::checkpoint::persist_checkpoint_incomplete(&*self.storage).await?;
        // Read all documents from the source (CRDT/schemaless).
        let docs: Vec<nodedb_types::document::Document> = {
            let crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
            crdt.list_ids(source_name)
                .iter()
                .filter_map(|id| {
                    crdt.read(source_name, id).map(|loro_val| {
                        crate::nodedb::convert::loro_value_to_document(id, &loro_val)
                    })
                })
                .collect()
        };

        if docs.is_empty() {
            return Err(LiteError::Query(format!(
                "collection '{source_name}' is empty or does not exist"
            )));
        }

        // Create the strict collection.
        self.strict
            .create_collection(source_name, target_schema.clone())
            .await?;

        // The collection's indexes cover the strict rows from here on, so
        // each row inserted below enters them, unique checks included.
        self.indexes
            .move_collection(&*self.storage, source_name, IndexEngine::Strict)
            .await?;

        // Convert each document to a row and insert.
        let mut inserted: Vec<Vec<Value>> = Vec::with_capacity(docs.len());
        for doc in &docs {
            let values = document_to_row(&doc.fields, &target_schema.columns);
            if let Err(e) = self.strict.insert(source_name, &values).await {
                self.strict.drop_collection(source_name).await?;
                self.restore_indexes(source_name, IndexEngine::Document)
                    .await?;
                return Err(conversion_refused(source_name, "strict", &doc.id, e));
            }
            inserted.push(values);
        }

        // Drop the old schemaless collection from CRDT in one step, so the
        // documents are either all rows now or all still documents.
        let cleared = self
            .crdt
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .clear_collection(source_name);
        if let Err(e) = cleared {
            self.strict.drop_collection(source_name).await?;
            self.restore_indexes(source_name, IndexEngine::Document)
                .await?;
            return Err(e);
        }

        // The documents are rows now: their text is indexed under row ids.
        self.fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .remove_collection_postings(source_name);
        crate::query::text_index::index_strict_rows(
            self,
            source_name,
            inserted.iter().map(Vec::as_slice),
        )?;

        self.register_strict_collection(source_name);

        let converted = inserted.len() as u64;
        Ok(QueryResult {
            columns: vec!["result".into()],
            rows: vec![vec![Value::String(format!(
                "converted {converted} documents to strict '{source_name}'"
            ))]],
            rows_affected: converted,
            command: None,
        })
    }

    /// Handle: CONVERT COLLECTION <name> TO columnar (<col_defs>)
    pub(in crate::query) async fn handle_convert_to_columnar(
        &self,
        sql: &str,
    ) -> Result<QueryResult, LiteError> {
        let (source_name, target_schema) = parse_convert_sql(sql, "columnar")?;
        self.convert_to_columnar(&source_name, target_schema).await
    }

    /// Convert `source_name` (CRDT or strict) to a plain columnar collection
    /// under `target_schema`. The source store is detected by probing the
    /// CRDT store first, then the strict store.
    ///
    /// All or nothing: a row the columnar engine refuses rolls the new
    /// columnar collection back.
    pub(in crate::query) async fn convert_to_columnar(
        &self,
        source_name: &str,
        target_schema: StrictSchema,
    ) -> Result<QueryResult, LiteError> {
        let guard = self.fts_state.admit_mutation().await;
        let result = self
            .convert_to_columnar_admitted(guard.permit(), source_name, target_schema)
            .await;
        guard.finish(result)
    }

    pub(in crate::query) async fn convert_to_columnar_admitted(
        &self,
        permit: &crate::engine::fts::coordinator::TextMutationPermit,
        source_name: &str,
        target_schema: StrictSchema,
    ) -> Result<QueryResult, LiteError> {
        let _permit = permit;
        crate::engine::fts::checkpoint::persist_checkpoint_incomplete(&*self.storage).await?;
        if self
            .fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .declaration_for(source_name)
            .is_some()
        {
            return Err(LiteError::Unsupported {
                detail: format!(
                    "CONVERT COLLECTION '{source_name}' TO columnar retains an active SEARCH INDEX: DROP SEARCH INDEX first"
                ),
            });
        }

        let columnar_schema = ColumnarSchema::new(target_schema.columns)
            .map_err(|e| LiteError::Query(e.to_string()))?;

        // Read from CRDT or strict.
        let rows = self
            .read_source_rows(source_name, &columnar_schema.columns)
            .await?;

        // Create columnar collection.
        self.columnar
            .create_collection(source_name, columnar_schema, ColumnarProfile::Plain, false)
            .await?;

        // Insert rows.
        for (n, row) in rows.iter().enumerate() {
            if let Err(e) = self.columnar.insert(source_name, row) {
                self.columnar.drop_collection(source_name).await?;
                return Err(conversion_refused(
                    source_name,
                    "columnar",
                    &format!("#{n}"),
                    e,
                ));
            }
        }

        self.register_columnar_collection(source_name);

        let converted = rows.len() as u64;
        Ok(QueryResult {
            columns: vec!["result".into()],
            rows: vec![vec![Value::String(format!(
                "converted {converted} rows to columnar '{source_name}'"
            ))]],
            rows_affected: converted,
            command: None,
        })
    }

    /// Handle: CONVERT COLLECTION <name> TO document
    ///
    /// Reads from strict or columnar, writes as schemaless MessagePack documents.
    pub(in crate::query) async fn handle_convert_to_document(
        &self,
        sql: &str,
    ) -> Result<QueryResult, LiteError> {
        let parts: Vec<&str> = sql.split_whitespace().collect();
        let source_name = parts
            .get(2)
            .ok_or(LiteError::Query("expected collection name".into()))?
            .to_lowercase();
        self.convert_to_document(&source_name).await
    }

    /// Convert the strict collection `source_name` to schemaless documents.
    /// The strict store's own schema drives tuple decoding, so the source
    /// format needs no plan-level hint.
    ///
    /// All or nothing: every tuple is decoded before any document is
    /// written, and a document write that fails removes the documents
    /// already written and leaves the strict collection in place.
    pub(in crate::query) async fn convert_to_document(
        &self,
        source_name: &str,
    ) -> Result<QueryResult, LiteError> {
        let guard = self.fts_state.admit_mutation().await;
        let result = self
            .convert_to_document_admitted(guard.permit(), source_name)
            .await;
        guard.finish(result)
    }

    pub(in crate::query) async fn convert_to_document_admitted(
        &self,
        permit: &crate::engine::fts::coordinator::TextMutationPermit,
        source_name: &str,
    ) -> Result<QueryResult, LiteError> {
        let _permit = permit;
        crate::engine::fts::checkpoint::persist_checkpoint_incomplete(&*self.storage).await?;
        let mut written: Vec<String> = Vec::new();

        if let Some(schema) = self.strict.schema(source_name) {
            let rows = self.decode_strict_rows(source_name, &schema).await?;
            // The collection's indexes cover documents from here on, so each
            // document written below enters them, unique checks included.
            self.indexes
                .move_collection(&*self.storage, source_name, IndexEngine::Document)
                .await?;
            let upserted = {
                let mut crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
                let mut result = Ok(());
                for values in &rows {
                    let doc_id = nodedb_types::id_gen::uuid_v7();
                    let fields: Vec<(&str, loro::LoroValue)> = schema
                        .columns
                        .iter()
                        .zip(values.iter())
                        .map(|(col, val)| (col.name.as_str(), value_to_loro(val)))
                        .collect();
                    if let Err(e) = crdt.upsert(source_name, &doc_id, &fields) {
                        result = Err(conversion_refused(source_name, "document", &doc_id, e));
                        break;
                    }
                    written.push(doc_id);
                }
                if result.is_err() {
                    for doc_id in &written {
                        crdt.delete(source_name, doc_id)?;
                    }
                }
                result
            };
            if let Err(e) = upserted {
                self.restore_indexes(source_name, IndexEngine::Strict)
                    .await?;
                return Err(e);
            }

            // Drop the strict collection.
            self.strict.drop_collection(source_name).await?;

            // The rows are documents now: their text is indexed under the
            // new document ids.
            self.fts_state
                .manager
                .lock()
                .map_err(|_| LiteError::LockPoisoned)?
                .remove_collection_postings(source_name);
            crate::query::text_index::reindex_documents(
                self,
                source_name,
                written.iter().map(String::as_str),
            )?;
        }

        self.register_collection(source_name);

        let converted = written.len() as u64;
        Ok(QueryResult {
            columns: vec!["result".into()],
            rows: vec![vec![Value::String(format!(
                "converted {converted} rows to document '{source_name}'"
            ))]],
            rows_affected: converted,
            command: None,
        })
    }
}
