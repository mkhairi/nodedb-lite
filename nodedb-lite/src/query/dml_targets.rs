// SPDX-License-Identifier: Apache-2.0

//! The rows a SQL `UPDATE` or `DELETE` on a document collection targets.
//!
//! The plan's key list names the targets when the WHERE names keys. Any
//! other WHERE, or none, is resolved when the statement runs: every live row
//! is read in its SQL read shape, the WHERE is applied to it, and the
//! matching rows' keys are the targets. KV collections resolve their targets
//! the same way (`visitor::kv_dml`).

use std::collections::BTreeSet;

use nodedb_sql::types::SqlValue;
use nodedb_sql::types::filter::Filter;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::nodedb::convert::loro_value_to_document;
use crate::query::engine::{LiteQueryEngine, sql_value_to_string};
use crate::query::visitor::scan_post::filter_mask;
use crate::storage::engine::StorageEngine;

/// Ids of the schemaless documents of `collection` a statement targets: the
/// named keys, or else every document `filters` match.
pub(crate) fn document_targets<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<Vec<String>, LiteError> {
    if !target_keys.is_empty() {
        return Ok(target_keys.iter().map(sql_value_to_string).collect());
    }
    let docs: Vec<(String, std::collections::HashMap<String, Value>)> = {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        crdt.list_ids(collection)
            .into_iter()
            .filter_map(|id| {
                let value = crdt.read(collection, &id)?;
                let fields = loro_value_to_document(&id, &value).fields;
                Some((id, fields))
            })
            .collect()
    };
    // One result shaped as the documents' SQL read: a column per field any
    // document holds, NULL where a document lacks it.
    let columns: Vec<String> = docs
        .iter()
        .flat_map(|(_, fields)| fields.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let rows: Vec<Vec<Value>> = docs
        .iter()
        .map(|(_, fields)| {
            columns
                .iter()
                .map(|c| fields.get(c).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    let keep = filter_mask(
        &QueryResult {
            columns,
            rows,
            rows_affected: 0,
            command: None,
        },
        filters,
    )?;
    Ok(docs
        .into_iter()
        .zip(keep)
        .filter_map(|((id, _), keep)| keep.then_some(id))
        .collect())
}

/// Primary keys of the strict rows of `collection` a statement targets: the
/// named keys, or else every row `filters` match.
pub(crate) async fn strict_targets<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    filters: &[Filter],
    named_keys: Vec<Value>,
    pk_idx: usize,
) -> Result<Vec<Value>, LiteError> {
    if !named_keys.is_empty() {
        return Ok(named_keys);
    }
    let Some(schema) = engine.strict.schema(collection) else {
        return Ok(Vec::new());
    };
    let rows = engine.strict.list_rows(collection).await?;
    let keep = filter_mask(
        &QueryResult {
            columns: schema.columns.iter().map(|c| c.name.clone()).collect(),
            rows: rows.clone(),
            rows_affected: 0,
            command: None,
        },
        filters,
    )?;
    Ok(rows
        .into_iter()
        .zip(keep)
        .filter_map(|(row, keep)| if keep { row.get(pk_idx).cloned() } else { None })
        .collect())
}
