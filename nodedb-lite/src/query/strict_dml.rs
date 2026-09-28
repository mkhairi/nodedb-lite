//! Strict-engine DML dispatch for the Lite query layer.
//!
//! INSERT, UPDATE, and DELETE for strict collections convert SQL values to
//! `nodedb_types::Value` according to the collection schema, then delegate
//! to `StrictEngine` which validates types and encodes as Binary Tuples.
//! Every write then brings the rows' text-index entries up to date.

use std::collections::HashMap;

use nodedb_sql::types::filter::Filter;
use nodedb_sql::types::{SqlExpr, SqlValue};
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::dml_targets::strict_targets;
use crate::query::engine::LiteQueryEngine;
use crate::query::text_index::{index_strict_rows, reindex_strict_rows};
use crate::storage::engine::StorageEngine;

use super::coerce::{build_row, coerce_sql_value, sql_value_to_string, sql_value_to_value};
use super::engine_read::parse_pk_value;

/// Insert rows into a strict collection.
///
/// Each `row` is a list of `(column_name, SqlValue)` pairs. Values are
/// coerced to match the schema column type.
pub async fn insert_strict<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    rows: &[Vec<(String, SqlValue)>],
    if_absent: bool,
) -> Result<QueryResult, LiteError> {
    let strict = &engine.strict;
    let schema = strict
        .schema(collection)
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("strict collection '{collection}' does not exist"),
        })?;

    // Cache the PK column position — `if_absent` requires a real PK; without one
    // we'd silently treat column 0 as the key and corrupt rows.
    let pk_idx = if if_absent {
        Some(
            schema
                .columns
                .iter()
                .position(|c| c.primary_key)
                .ok_or_else(|| LiteError::BadRequest {
                    detail: format!(
                        "strict collection '{collection}' has no primary key column; \
                         INSERT … ON CONFLICT DO NOTHING requires one"
                    ),
                })?,
        )
    } else {
        None
    };

    let mut affected: u64 = 0;
    for row_pairs in rows {
        let values = build_row(row_pairs, &schema.columns)?;

        if let Some(idx) = pk_idx {
            let pk_val = &values[idx];
            // Check for duplicate by attempting a point read.
            if strict.get(collection, pk_val).await?.is_some() {
                continue;
            }
        }

        strict
            .insert(collection, &values)
            .await
            .map_err(|e| match e {
                LiteError::BadRequest { detail }
                    if detail.contains("duplicate primary key") && if_absent =>
                {
                    // Race: another insert won between our check and insert.
                    // if_absent semantics: skip.
                    LiteError::BadRequest { detail }
                }
                other => other,
            })?;
        index_strict_rows(engine, collection, [values.as_slice()])?;
        affected += 1;
    }
    Ok(QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: affected,
        command: Some("INSERT".into()),
    })
}

/// Update the rows of a strict collection the WHERE targets: the named
/// primary keys, or else every row the WHERE matches.
pub async fn update_strict<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    assignments: &[(String, SqlExpr)],
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<QueryResult, LiteError> {
    let strict = &engine.strict;
    let schema = strict
        .schema(collection)
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("strict collection '{collection}' does not exist"),
        })?;
    let (pk_idx, pk_col) = schema
        .columns
        .iter()
        .enumerate()
        .find(|(_, c)| c.primary_key)
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("strict collection '{collection}' has no primary key column"),
        })?;

    // Convert assignments to a HashMap<col_name, Value>.
    let mut updates: HashMap<String, Value> = HashMap::with_capacity(assignments.len());
    for (field, expr) in assignments {
        let SqlExpr::Literal(val) = expr else {
            continue;
        };
        let typed = match schema.columns.iter().find(|c| c.name == *field) {
            Some(c) => coerce_sql_value(val, &c.column_type)?,
            None => sql_value_to_value(val)?,
        };
        updates.insert(field.clone(), typed);
    }

    let named: Vec<Value> = target_keys
        .iter()
        .map(|key| parse_pk_value(&sql_value_to_string(key), &pk_col.column_type))
        .collect();
    let pks = strict_targets(engine, collection, filters, named, pk_idx).await?;
    let mut affected: u64 = 0;
    for pk_value in &pks {
        if strict.update(collection, pk_value, &updates).await? {
            affected += 1;
        }
    }
    reindex_strict_rows(engine, collection, &pks).await?;
    Ok(QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: affected,
        command: Some("UPDATE".into()),
    })
}

/// Delete the rows of a strict collection the WHERE targets: the named
/// primary keys, or else every row the WHERE matches.
pub async fn delete_strict<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<QueryResult, LiteError> {
    let strict = &engine.strict;
    let schema = strict
        .schema(collection)
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("strict collection '{collection}' does not exist"),
        })?;
    let (pk_idx, pk_col) = schema
        .columns
        .iter()
        .enumerate()
        .find(|(_, c)| c.primary_key)
        .ok_or_else(|| LiteError::BadRequest {
            detail: format!("strict collection '{collection}' has no primary key column"),
        })?;

    let named: Vec<Value> = target_keys
        .iter()
        .map(|key| parse_pk_value(&sql_value_to_string(key), &pk_col.column_type))
        .collect();
    let pks = strict_targets(engine, collection, filters, named, pk_idx).await?;
    let mut affected: u64 = 0;
    for pk_value in &pks {
        if strict.delete(collection, pk_value).await? {
            affected += 1;
        }
    }
    reindex_strict_rows(engine, collection, &pks).await?;
    Ok(QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: affected,
        command: Some("DELETE".into()),
    })
}
