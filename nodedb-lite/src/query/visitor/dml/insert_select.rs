// SPDX-License-Identifier: Apache-2.0

//! `INSERT ... SELECT` lowering.

use std::collections::HashMap;

use nodedb_sql::types::SqlPlan;
use nodedb_sql::types_expr::SqlExpr;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::document_ops::is_strict;
use crate::query::document_ops::writes::point_insert;
use crate::query::engine::LiteQueryEngine;
use crate::query::expr_convert::convert_sql_expr;
use crate::storage::engine::StorageEngine;

use crate::query::visitor::adapter::LiteFut;
use crate::query::visitor::scan_post::row_to_typed_value;

use super::rows::{declared_primary_key, extract_id, row_to_msgpack, value_to_sql_value};

/// `INSERT INTO target [(cols)] SELECT ... FROM source`.
///
/// Executes the source plan and inserts one target row per source row.
/// `column_map` is the target list: one `(target column, source
/// expression)` per declared column, evaluated against the source row.
/// Empty means copy each source row unchanged. Routing (strict vs
/// schemaless CRDT) is detected via `is_strict`.
pub(in crate::query::visitor) fn lower_insert_select<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    target: &str,
    source: &SqlPlan,
    limit: usize,
    column_map: &[(String, SqlExpr)],
) -> Result<LiteFut<'a>, LiteError> {
    let target = target.to_string();
    let source = source.clone();
    let effective_limit = if limit == 0 { usize::MAX } else { limit };

    // A declared PRIMARY KEY implies NOT NULL. A literal `SELECT NULL` into
    // the pk column is knowable before any row is scanned.
    let declared_pk = declared_primary_key(engine, &target);
    if let Some(pk) = &declared_pk
        && column_map.iter().any(|(field, expr)| {
            field == pk && matches!(expr, SqlExpr::Literal(nodedb_sql::types::SqlValue::Null))
        })
    {
        return Err(LiteError::NotNullViolation {
            collection: target,
            column: pk.clone(),
        });
    }
    let column_map: Vec<(String, nodedb_query::expr::types::SqlExpr)> = column_map
        .iter()
        .map(|(field, expr)| Ok((field.clone(), convert_sql_expr(expr)?)))
        .collect::<Result<_, LiteError>>()?;

    Ok(Box::pin(async move {
        let source_result = engine.execute_plan(&source).await?;
        let cols = source_result.columns.clone();
        let mut maps: Vec<HashMap<String, Value>> = Vec::new();
        for row in source_result.rows.into_iter().take(effective_limit) {
            if column_map.is_empty() {
                maps.push(cols.iter().cloned().zip(row).collect());
                continue;
            }
            let doc = row_to_typed_value(&cols, &row);
            let mut shaped = HashMap::with_capacity(column_map.len());
            for (field, expr) in &column_map {
                let value = expr.eval(&doc)?;
                if matches!(value, Value::Null) && declared_pk.as_deref() == Some(field.as_str()) {
                    return Err(LiteError::NotNullViolation {
                        collection: target.clone(),
                        column: field.clone(),
                    });
                }
                shaped.insert(field.clone(), value);
            }
            maps.push(shaped);
        }

        let mut affected: u64 = 0;

        if is_strict(engine, &target) {
            // Strict path: convert Value rows to SqlValue rows and call strict_dml.
            use crate::query::strict_dml;
            use nodedb_sql::types::SqlValue;

            let sql_rows: Vec<Vec<(String, SqlValue)>> = maps
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|(col, val)| (col, value_to_sql_value(val)))
                        .collect()
                })
                .collect();

            let res = strict_dml::insert_strict(engine, &target, &sql_rows, false).await?;
            affected = res.rows_affected;
        } else {
            // Schemaless CRDT path.
            for row_map in maps {
                let doc_id = extract_id(&row_map);
                let value_bytes = row_to_msgpack(&row_map)?;
                point_insert(engine, &target, &doc_id, &value_bytes, false).await?;
                affected += 1;
            }
        }

        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("INSERT".into()),
        })
    }))
}
