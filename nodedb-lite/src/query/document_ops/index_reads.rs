// SPDX-License-Identifier: Apache-2.0
//! Reads through a secondary index: the rows an equality or a range selects.
//!
//! The index yields candidates; each candidate's current row is read and, for
//! an equality, confirmed against the comparison a scan would evaluate. The
//! answer is therefore the scan's answer, in the scan's row shape.

use std::collections::HashMap;
use std::sync::Arc;

use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::engine::document::history::ops::{is_bitemporal, versioned_get_current};
use crate::error::LiteError;
use crate::index::document::{matches_probe, row_value};
use crate::index::key::key_of_doc_id;
use crate::index::{IndexDef, IndexEngine, canonical_field, field_spec};
use crate::query::document_rows;
use crate::query::engine::LiteQueryEngine;
use crate::query::kv_ops::sql_read::kv_select_keys;
use crate::storage::engine::StorageEngine;

/// The document or strict index on `field` (as the planner or a physical
/// op names it) of `collection`: the engines whose equality lookups the
/// planner rewrites.
fn lookup_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    field: &str,
) -> Result<Arc<IndexDef>, LiteError> {
    let (path, is_array) = canonical_field(field);
    let def = engine
        .indexes
        .def_on_field(collection, &field_spec(&path, is_array))
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("collection '{collection}' has no index on {path}"),
        })?;
    if def.engine == IndexEngine::KeyValue {
        return Err(LiteError::Unsupported {
            detail: format!(
                "index '{}' on key-value collection '{collection}' answers scans, \
                 not document index lookups",
                def.name
            ),
        });
    }
    Ok(def)
}

/// The current strict rows of `ids`, in order, that `keep` accepts, as
/// `(id, row values in schema order)`, with the schema's column names.
async fn strict_candidate_rows<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    ids: Vec<String>,
    keep: impl Fn(&Value) -> bool,
) -> Result<(Vec<String>, Vec<(String, Vec<Value>)>), LiteError> {
    let schema = engine
        .strict
        .schema(collection)
        .ok_or_else(|| LiteError::collection_not_found("strict", collection))?;
    let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let mut rows = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(values) = engine.strict.row_by_doc_id(collection, &id).await? else {
            continue;
        };
        let doc = Value::Object(
            columns
                .iter()
                .cloned()
                .zip(values.iter().cloned())
                .collect::<HashMap<_, _>>(),
        );
        if keep(&doc) {
            rows.push((id, values));
        }
    }
    Ok((columns, rows))
}

/// The current rows of `ids`, in order, that `keep` accepts, in the shape a
/// scan of `def`'s collection answers with.
async fn engine_candidate_rows<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    def: &IndexDef,
    ids: Vec<String>,
    keep: impl Fn(&Value) -> bool,
) -> Result<(Vec<String>, Vec<(String, Vec<Value>)>), LiteError> {
    match def.engine {
        IndexEngine::Document => Ok((
            document_rows::columns(),
            candidate_rows(engine, &def.collection, ids, keep).await?,
        )),
        IndexEngine::Strict => strict_candidate_rows(engine, &def.collection, ids, keep).await,
        IndexEngine::KeyValue => {
            let keys: Vec<Vec<u8>> = ids.iter().filter_map(|id| key_of_doc_id(id)).collect();
            let result = kv_select_keys(engine, &def.collection, &keys).await?;
            let rows = result
                .rows
                .into_iter()
                .map(|row| (String::new(), row))
                .collect();
            Ok((result.columns, rows))
        }
    }
}

/// The current rows of `ids`, in order, that `keep` accepts, as
/// `(id, row in the scan shape)`. A bitemporal collection's rows come from its
/// history, as its scan's do.
async fn candidate_rows<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    ids: Vec<String>,
    keep: impl Fn(&Value) -> bool,
) -> Result<Vec<(String, Vec<Value>)>, LiteError> {
    let mut rows = Vec::with_capacity(ids.len());
    if is_bitemporal(&*engine.storage, collection).await? {
        for id in ids {
            let Some(version) = versioned_get_current(&*engine.storage, collection, &id).await?
            else {
                continue;
            };
            if !version.is_live() {
                continue;
            }
            let doc = if version.body.is_empty() {
                Value::Object(Default::default())
            } else {
                nodedb_types::json_msgpack::value_from_msgpack(&version.body).map_err(|e| {
                    LiteError::Serialization {
                        detail: format!(
                            "live version of '{collection}'/'{id}' does not decode: {e}"
                        ),
                    }
                })?
            };
            if keep(&doc) {
                let row = document_rows::value_row(&id, doc);
                rows.push((id, row));
            }
        }
        return Ok(rows);
    }
    let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
    for id in ids {
        let Some(doc) = crdt.read(collection, &id) else {
            continue;
        };
        if keep(&row_value(&doc)) {
            let row = document_rows::crdt_row(&id, &doc);
            rows.push((id, row));
        }
    }
    Ok(rows)
}

/// The documents of `collection` whose `field` equals `probe`, through the
/// index on that field, after skipping `offset` and keeping at most `limit`.
pub async fn indexed_fetch<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    field: &str,
    probe: &Value,
    limit: usize,
    offset: usize,
) -> Result<QueryResult, LiteError> {
    let def = lookup_index(engine, collection, field)?;
    let ids = engine.indexes.lookup_eq(&def, probe);
    let (columns, rows) =
        engine_candidate_rows(engine, &def, ids, |doc| matches_probe(&def, doc, probe)).await?;
    let rows = rows
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|(_, row)| row)
        .collect();
    Ok(QueryResult {
        columns,
        rows,
        rows_affected: 0,
        command: None,
    })
}

/// The ids of the documents of `collection` whose `field` equals `probe`.
pub async fn index_lookup<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    field: &str,
    probe: &Value,
) -> Result<QueryResult, LiteError> {
    let def = lookup_index(engine, collection, field)?;
    let ids = engine.indexes.lookup_eq(&def, probe);
    let (_, rows) =
        engine_candidate_rows(engine, &def, ids, |doc| matches_probe(&def, doc, probe)).await?;
    let rows = rows
        .into_iter()
        .map(|(id, _)| vec![Value::String(id)])
        .collect();
    Ok(QueryResult {
        columns: vec!["document_id".into()],
        rows,
        rows_affected: 0,
        command: None,
    })
}

/// The documents `def` may hold between the bounds, in index order. `None`
/// when the bounds share no value class and the caller must scan. The rows
/// are candidates: the caller applies the range predicate itself.
pub async fn index_range_fetch<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    def: &IndexDef,
    lower: Option<&Value>,
    upper: Option<&Value>,
) -> Result<Option<QueryResult>, LiteError> {
    let Some(ids) = engine.indexes.lookup_range(def, lower, upper) else {
        return Ok(None);
    };
    let (columns, rows) = engine_candidate_rows(engine, def, ids, |_| true).await?;
    Ok(Some(QueryResult {
        columns,
        rows: rows.into_iter().map(|(_, row)| row).collect(),
        rows_affected: 0,
        command: None,
    }))
}
