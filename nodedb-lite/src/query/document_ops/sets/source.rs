// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::reads::loro_value_to_ndb_value;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::value_utils::value_to_string;
use crate::storage::engine::StorageEngine;
use nodedb_types::value::Value;
use std::collections::HashMap;
/// Collection and column bindings for document joins.
#[derive(Clone, Copy)]
pub(crate) struct DocumentJoin<'a> {
    pub target_collection: &'a str,
    pub source_collection: &'a str,
    pub source_alias: &'a str,
    pub target_join_col: &'a str,
    pub source_join_col: &'a str,
}

/// Scan a collection and return all document IDs.
pub(in crate::query) async fn collect_ids_pub<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<Vec<String>, LiteError> {
    collect_ids(engine, collection).await
}

/// Fetch a document as a field map — public for query-layer callers.
pub(in crate::query) async fn fetch_document_value_pub<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    doc_id: &str,
) -> Result<HashMap<String, Value>, LiteError> {
    fetch_document_value(engine, collection, doc_id).await
}

pub(super) async fn collect_ids<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<Vec<String>, LiteError> {
    if is_strict(engine, collection) {
        let schema = engine
            .strict
            .schema(collection)
            .ok_or_else(|| LiteError::collection_not_found("strict", collection))?;
        let pk_idx = schema
            .columns
            .iter()
            .position(|c| c.primary_key)
            .ok_or_else(|| LiteError::BadRequest {
                detail: format!("strict collection '{collection}' has no primary key"),
            })?;
        let all_rows = engine.strict.list_rows(collection).await?;
        Ok(all_rows
            .iter()
            .map(|row| value_to_string(&row[pk_idx]))
            .collect())
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        Ok(crdt.list_ids(collection))
    }
}

/// Fetch a document as a field map (String → Value).
pub(super) async fn fetch_document_value<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    doc_id: &str,
) -> Result<HashMap<String, Value>, LiteError> {
    if is_strict(engine, collection) {
        let schema = engine
            .strict
            .schema(collection)
            .ok_or_else(|| LiteError::collection_not_found("strict", collection))?;
        let pk = Value::String(doc_id.to_string());
        match engine.strict.get(collection, &pk).await? {
            Some(row) => {
                let map = schema
                    .columns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, col)| row.get(i).map(|v| (col.name.clone(), v.clone())))
                    .collect();
                Ok(map)
            }
            None => Ok(HashMap::new()),
        }
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        match crdt.read(collection, doc_id) {
            Some(val) => match loro_value_to_ndb_value(&val) {
                Value::Object(map) => Ok(map),
                _ => Ok(HashMap::new()),
            },
            None => Ok(HashMap::new()),
        }
    }
}

/// Build a join map: join_col_value → document field map.
pub(super) async fn build_join_map<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    join_col: &str,
) -> Result<HashMap<String, HashMap<String, Value>>, LiteError> {
    let ids = collect_ids(engine, collection).await?;
    let mut map: HashMap<String, HashMap<String, Value>> = HashMap::with_capacity(ids.len());
    for id in &ids {
        let doc = fetch_document_value(engine, collection, id).await?;
        if let Some(key_val) = doc.get(join_col) {
            let key = value_to_string(key_val);
            map.insert(key, doc);
        }
    }
    Ok(map)
}

/// Extract a field value from a document map as a String, returning None if absent.
pub(super) fn extract_field_str(doc: &HashMap<String, Value>, field: &str) -> Option<String> {
    doc.get(field).map(value_to_string)
}
