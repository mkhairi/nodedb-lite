// SPDX-License-Identifier: Apache-2.0

//! `INSERT ... SELECT` lowering.

use std::collections::HashMap;

use nodedb_sql::types::SqlPlan;
use nodedb_sql::types_expr::SqlExpr;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::document_ops::is_strict;
use crate::query::document_ops::writes::point_insert_admitted;
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
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
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
        let source_result = engine.execute_plan_admitted(&source, Some(permit)).await?;
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

            let res = strict_dml::insert_strict_admitted(engine, permit, &target, &sql_rows, false)
                .await?;
            affected = res.rows_affected;
        } else {
            // Schemaless CRDT path.
            for row_map in maps {
                let doc_id = extract_id(&row_map);
                let value_bytes = row_to_msgpack(row_map)?;
                point_insert_admitted(engine, permit, &target, &doc_id, &value_bytes, false)
                    .await?;
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

#[cfg(test)]
mod tests {
    use nodedb_sql::types::SqlValue;
    use nodedb_sql::types_expr::BinaryOp;
    use nodedb_types::columnar::{ColumnDef, ColumnType, StrictSchema};

    use nodedb_sql::types::SqlPlan;
    use nodedb_sql::types::query::EngineType;
    use nodedb_sql::types_expr::SqlExpr;
    use nodedb_types::value::Value;

    use crate::error::LiteError;
    use crate::query::engine::LiteQueryEngine;

    use super::lower_insert_select;
    use crate::query::engine::test_engine;

    fn schema() -> StrictSchema {
        StrictSchema {
            columns: vec![
                ColumnDef::required("id", ColumnType::String).with_primary_key(),
                ColumnDef::nullable("n", ColumnType::Int64),
            ],
            version: 1,
            dropped_columns: Vec::new(),
            bitemporal: false,
        }
    }

    fn scan(collection: &str) -> SqlPlan {
        SqlPlan::Scan {
            collection: collection.to_string(),
            alias: None,
            engine: EngineType::DocumentStrict,
            filters: Vec::new(),
            projection: Vec::new(),
            sort_keys: Vec::new(),
            limit: None,
            offset: 0,
            distinct: false,
            window_functions: Vec::new(),
            temporal: nodedb_sql::temporal::TemporalScope::default(),
        }
    }

    fn col(name: &str) -> SqlExpr {
        SqlExpr::Column {
            table: None,
            name: name.to_string(),
        }
    }

    async fn seed(engine: &LiteQueryEngine<crate::PagedbStorageMem>) {
        engine
            .strict
            .create_collection("src", schema())
            .await
            .expect("src");
        engine
            .strict
            .create_collection("dst", schema())
            .await
            .expect("dst");
        engine
            .strict
            .insert("src", &[Value::String("a".into()), Value::Integer(5)])
            .await
            .expect("row");
    }

    #[tokio::test]
    async fn insert_select_column_map_shapes_each_target_row() {
        let engine = test_engine().await;
        seed(&engine).await;
        let guard = engine.fts_state.admit_mutation().await;
        let column_map = vec![
            (
                "id".to_string(),
                SqlExpr::BinaryOp {
                    left: Box::new(col("id")),
                    op: BinaryOp::Concat,
                    right: Box::new(SqlExpr::Literal(SqlValue::String("-copy".into()))),
                },
            ),
            (
                "n".to_string(),
                SqlExpr::BinaryOp {
                    left: Box::new(col("n")),
                    op: BinaryOp::Mul,
                    right: Box::new(SqlExpr::Literal(SqlValue::Int(2))),
                },
            ),
        ];
        let r = lower_insert_select(&engine, guard.permit(), "dst", &scan("src"), 0, &column_map)
            .expect("lower")
            .await;
        let r = guard.finish(r).expect("run");
        assert_eq!(r.rows_affected, 1);
        let row = engine
            .strict
            .get("dst", &Value::String("a-copy".into()))
            .await
            .expect("get")
            .expect("row present");
        assert_eq!(row[1], Value::Integer(10));
    }

    #[tokio::test]
    async fn insert_select_literal_null_into_pk_is_rejected_before_scanning() {
        let engine = test_engine().await;
        seed(&engine).await;
        let guard = engine.fts_state.admit_mutation().await;
        let column_map = vec![
            ("id".to_string(), SqlExpr::Literal(SqlValue::Null)),
            ("n".to_string(), col("n")),
        ];
        let err =
            match lower_insert_select(&engine, guard.permit(), "dst", &scan("src"), 0, &column_map)
            {
                Ok(_) => panic!("a literal NULL into the pk must be rejected"),
                Err(e) => e,
            };
        let err = guard
            .finish::<(), _>(Err(err))
            .expect_err("refused mutation");
        assert!(matches!(err, LiteError::NotNullViolation { .. }), "{err}");
    }

    #[tokio::test]
    async fn insert_select_empty_map_copies_rows_unchanged() {
        let engine = test_engine().await;
        seed(&engine).await;
        let guard = engine.fts_state.admit_mutation().await;
        let r = lower_insert_select(&engine, guard.permit(), "dst", &scan("src"), 0, &[])
            .expect("lower")
            .await;
        let r = guard.finish(r).expect("run");
        assert_eq!(r.rows_affected, 1);
        let row = engine
            .strict
            .get("dst", &Value::String("a".into()))
            .await
            .expect("get")
            .expect("row present");
        assert_eq!(row[1], Value::Integer(5));
    }
}
