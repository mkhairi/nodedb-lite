// SPDX-License-Identifier: Apache-2.0
//! SQL-visitor lowering for query-shaped SqlPlan variants:
//! Aggregate, Join, DocumentIndexLookup, RangeScan, Cte, Subquery.

use std::collections::HashMap;

use nodedb_physical::PhysicalTaskVisitor;
use nodedb_physical::physical_plan::SortKeySpec;
use nodedb_physical::physical_plan::document::DocumentOp;
use nodedb_physical::physical_plan::query::AggregateSpec;
use nodedb_query::expr::GroupKeySpec;
use nodedb_sql::SubqueryVisitArgs;
use nodedb_sql::temporal::TemporalScope;
use nodedb_sql::types::SqlPlan;
use nodedb_sql::types::filter::{Filter, FilterExpr};
use nodedb_sql::types::query::EngineType;
use nodedb_sql::types::query::{AggregateExpr, JoinType, Projection, SortKey, WindowSpec};
use nodedb_sql::types_expr::{BinaryOp, SqlExpr, SqlValue};
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::document_ops::indexes::bare_path;
use crate::query::engine::LiteQueryEngine;
use crate::query::expr_convert::convert_sql_expr;
use crate::query::filter_convert::{sql_filters_to_metadata, sql_value_to_value};
use crate::query::physical_visitor::LiteDataPlaneVisitor;
use crate::query::query_ops::aggregate::execute_aggregate;
use crate::query::query_ops::joins::inline_hash::execute_inline_hash_join;
use crate::query::value_utils::sql_value_to_index_key;
use crate::storage::engine::StorageEngine;

use super::adapter::LiteFut;
use super::having_eval::{apply_having_result, make_agg_alias_map};
use super::projection::join_projections;
use super::scan_post::{
    RowSink, ScanPostArgs, apply_scan_post_processing, apply_window_functions, distinct_rows,
    filter_rows, project_rows, sort_rows,
};

/// Convert a `nodedb_sql` `AggregateExpr` to a physical `AggregateSpec`.
pub(super) fn sql_agg_to_spec(agg: &AggregateExpr) -> AggregateSpec {
    let field = agg
        .args
        .first()
        .and_then(|a| match a {
            SqlExpr::Column { name, .. } => Some(name.clone()),
            SqlExpr::Wildcard => Some("*".to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "*".to_string());

    AggregateSpec {
        function: agg.function.clone(),
        alias: agg.alias.clone(),
        user_alias: None,
        field,
        expr: None,
    }
}

fn convert_aggregates(aggs: &[AggregateExpr]) -> Vec<AggregateSpec> {
    aggs.iter().map(sql_agg_to_spec).collect()
}

/// Convert a SQL `SortKey` into the physical-plan [`SortKeySpec`].
///
/// A bare column becomes a pushable key. Anything computed becomes a
/// deliberately NON-column spec, so `SortKeySpec::as_column()` reports `None`
/// and every consumer skips it — which is what already happened in practice:
/// the previous code stringified the expression with `format!("{other:?}")`
/// and used that as a column name, which could never match a real column. Same
/// observable ordering, but the limitation is now explicit in the type instead
/// of hidden behind an unmatchable name.
fn sort_key_to_spec(k: &SortKey) -> SortKeySpec {
    match &k.expr {
        SqlExpr::Column { name, .. } => SortKeySpec::column(name.clone(), k.ascending),
        _ => SortKeySpec {
            expr: nodedb_query::expr::SqlExpr::Literal(nodedb_types::value::Value::Null),
            ascending: k.ascending,
            // Same PostgreSQL default `SortKeySpec::column` applies:
            // ASC → NULLS LAST, DESC → NULLS FIRST.
            nulls_first: !k.ascending,
        },
    }
}

/// Encode `Vec<Filter>` → msgpack bytes via `ScanFilter`.
fn encode_filters(filters: &[Filter]) -> Result<Vec<u8>, LiteError> {
    if filters.is_empty() {
        return Ok(Vec::new());
    }
    // Complex QExpr predicates are evaluated post-scan; only primitive conditions
    // are pushed to the physical visitor via serialized MetadataFilter.
    match sql_filters_to_metadata(filters, &[])?.meta {
        None => Ok(Vec::new()),
        Some(mf) => zerompk::to_msgpack_vec(&mf).map_err(|e| LiteError::Serialization {
            detail: format!("encode filters: {e}"),
        }),
    }
}

/// Convert `QueryResult` rows to `Vec<HashMap<String, Value>>`.
fn result_to_maps(result: QueryResult) -> Vec<HashMap<String, Value>> {
    let cols = result.columns;
    result
        .rows
        .into_iter()
        .map(|row| cols.iter().cloned().zip(row).collect())
        .collect()
}

/// Encode a `QueryResult` as msgpack bytes for inline hash join.
fn encode_result_msgpack(result: &QueryResult) -> Result<Vec<u8>, LiteError> {
    let maps: Vec<HashMap<String, Value>> = result
        .rows
        .iter()
        .map(|row| {
            result
                .columns
                .iter()
                .cloned()
                .zip(row.iter().cloned())
                .collect()
        })
        .collect();
    zerompk::to_msgpack_vec(&maps).map_err(|e| LiteError::Serialization {
        detail: format!("encode join side msgpack: {e}"),
    })
}

// ── Aggregate ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_aggregate<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    input: &SqlPlan,
    group_by: &[SqlExpr],
    aggregates: &[AggregateExpr],
    having: &[Filter],
    _limit: usize,
    grouping_sets: Option<&[Vec<usize>]>,
    sort_keys: &[SortKey],
) -> Result<LiteFut<'a>, LiteError> {
    let input = input.clone();
    // A bare column groups by field extraction; anything else is a computed key
    // (e.g. `GROUP BY date_trunc('day', ts)`) that must be evaluated per row.
    // Its output name is the SQL text the planner assigned, so HAVING/ORDER BY
    // and the result columns all agree on one label.
    let group_cols: Vec<GroupKeySpec> = group_by
        .iter()
        .enumerate()
        .map(|(i, e)| match e {
            SqlExpr::Column { name, .. } => Ok(GroupKeySpec::column(name.clone())),
            other => {
                let expr = convert_sql_expr(other)?;
                // `SqlPlan::Aggregate` carries `group_by_aliases`, but the
                // upstream dispatcher discards them before the visitor sees
                // them, so the positional label is the stable name available.
                Ok(GroupKeySpec {
                    output_name: format!("group_key_{i}"),
                    field: None,
                    expr: Some(expr),
                })
            }
        })
        .collect::<Result<Vec<_>, LiteError>>()?;
    let agg_specs = convert_aggregates(aggregates);
    // Build aggregate-function → alias lookup for HAVING post-filter.
    let agg_alias_map = make_agg_alias_map(aggregates);
    // HAVING predicates always reference aggregate results (e.g. SUM(salary) > 100)
    // which are not present as named columns until after aggregation.
    // apply_having_result handles all predicate shapes via having_eval, including
    // function-call resolution through agg_alias_map. We always do the post-filter
    // and pass empty bytes to execute_aggregate (no pushdown for HAVING).
    let having_post = having.to_vec();
    let sort_pairs: Vec<SortKeySpec> = sort_keys.iter().map(sort_key_to_spec).collect();
    let gs: Vec<Vec<u32>> = grouping_sets
        .unwrap_or(&[])
        .iter()
        .map(|s| s.iter().map(|&i| i as u32).collect())
        .collect();

    Ok(Box::pin(async move {
        let source_result = engine.execute_plan(&input).await?;
        let rows = result_to_maps(source_result);
        let result = execute_aggregate(rows, &group_cols, &agg_specs, &[], &[], &sort_pairs, &gs)?;
        Ok(apply_having_result(result, &having_post, &agg_alias_map))
    }))
}

// ── Join ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_join<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    left: &SqlPlan,
    right: &SqlPlan,
    on: &[(String, String)],
    join_type: JoinType,
    _condition: Option<&SqlExpr>,
    limit: Option<usize>,
    projection: &[Projection],
    filters: &[Filter],
) -> Result<LiteFut<'a>, LiteError> {
    let left = left.clone();
    let right = right.clone();
    let on = on.to_vec();
    let limit = limit.unwrap_or(usize::MAX);
    // JoinType debug output: Inner, Left, Right, Full — lower to string for hash join.
    let join_type_str = format!("{join_type:?}").to_lowercase();
    let proj = join_projections(projection, "JOIN")?;
    let post_filters_bytes = encode_filters(filters)?;

    Ok(Box::pin(async move {
        let left_result = engine.execute_plan(&left).await?;
        let right_result = engine.execute_plan(&right).await?;

        let left_bytes = encode_result_msgpack(&left_result)?;
        let right_bytes = encode_result_msgpack(&right_result)?;

        execute_inline_hash_join(
            &left_bytes,
            &right_bytes,
            None,
            &on,
            &join_type_str,
            limit,
            &proj,
            &post_filters_bytes,
        )
    }))
}

// ── DocumentIndexLookup ──────────────────────────────────────────────────────

/// Lower an equality on an indexed field.
///
/// The rows are the ones a full scan with the same WHERE returns, in the same
/// shape. The planner removes the indexed equality from `filters`. It is put
/// back, so the fetched rows are filtered with the scan's own equality, and
/// a key two distinct values share (`'1'` and `1`) cannot leak a row. The
/// index narrows the rows only when [`index_candidate_ids`] can name every
/// row the scan matches. Otherwise the collection is scanned.
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_document_index_lookup<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    _alias: Option<&str>,
    engine_type: EngineType,
    field: &str,
    value: &SqlValue,
    filters: &[Filter],
    projection: &[Projection],
    sort_keys: &[SortKey],
    limit: Option<usize>,
    offset: usize,
    distinct: bool,
    window_functions: &[WindowSpec],
    case_insensitive: bool,
    _temporal: &TemporalScope,
) -> Result<LiteFut<'a>, LiteError> {
    if case_insensitive {
        // Lite lists no case-insensitive index as Ready, so the planner never
        // emits this. The literal arrives lowercased, and the scan equality
        // cannot be rebuilt from it.
        return Err(LiteError::BadRequest {
            detail: format!(
                "case-insensitive index lookup on {collection}({field}) is not supported"
            ),
        });
    }
    let collection = collection.to_string();
    let field = field.to_string();
    let value = value.clone();
    let mut all_filters = Vec::with_capacity(filters.len() + 1);
    all_filters.push(Filter {
        expr: FilterExpr::Expr(SqlExpr::BinaryOp {
            // The evaluator ignores table qualifiers, so `table: None` is safe.
            left: Box::new(SqlExpr::Column {
                table: None,
                name: bare_path(&field).to_string(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(SqlExpr::Literal(value.clone())),
        }),
    });
    all_filters.extend_from_slice(filters);
    let sort_keys = sort_keys.to_vec();
    let window_functions = window_functions.to_vec();
    let projection = projection.to_vec();

    Ok(Box::pin(async move {
        // The same row budget the scan lowering applies.
        let budget = if sort_keys.is_empty() && !distinct && window_functions.is_empty() {
            limit.map(|n| n.saturating_add(offset))
        } else {
            None
        };
        let mut sink = RowSink::new(&all_filters, budget)?;
        let key = if engine_type == EngineType::DocumentSchemaless {
            index_lookup_key(engine, &collection, &value).await
        } else {
            None
        };
        // Postings and rows come from one CRDT guard, so one snapshot.
        let from_postings = match &key {
            Some(key) => engine.push_schemaless_rows(&collection, &field, key, &mut sink)?,
            None => false,
        };
        if !from_postings {
            engine
                .execute_scan_into(&collection, &engine_type, &mut sink)
                .await?;
        }
        // WHERE was applied per row, so the post stages see no filters.
        apply_scan_post_processing(
            sink.into_result(),
            ScanPostArgs {
                filters: &[],
                sort_keys: &sort_keys,
                window_specs: &window_functions,
                projection: &projection,
                sequences: engine.sequences(),
                limit,
                offset,
                distinct,
            },
        )
    }))
}

/// Posting key for `field = value` on a schemaless collection, or `None`
/// when the caller scans instead:
/// - `value` is not a string, or parses as a number. The scan equality
///   coerces numbers, so `1` matches `'01'`, `'1.0'` and `true`, which hold
///   other keys. A null never matches a posting.
/// - The collection is bitemporal. Its scan reads the history table, which
///   the CRDT documents behind the postings can lag.
///
/// Runs before the CRDT lock is taken: the bitemporal check awaits.
async fn index_lookup_key<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    value: &SqlValue,
) -> Option<String> {
    let SqlValue::String(literal) = value else {
        return None;
    };
    if literal.parse::<f64>().is_ok() {
        return None;
    }
    let key = sql_value_to_index_key(value)?;
    let bitemporal =
        crate::engine::document::history::ops::is_bitemporal(engine.storage.as_ref(), collection)
            .await
            .unwrap_or(true);
    if bitemporal { None } else { Some(key) }
}

// ── RangeScan ────────────────────────────────────────────────────────────────

pub(super) fn lower_range_scan<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    field: &str,
    lower: Option<&SqlValue>,
    upper: Option<&SqlValue>,
    limit: usize,
) -> Result<LiteFut<'a>, LiteError> {
    // Lite holds a bare collection name; DatabaseId::DEFAULT keeps it unqualified.
    let col = nodedb_types::QualifiedCollection::new(nodedb_types::DatabaseId::DEFAULT, collection);
    let fld = field.to_string();

    let encode_bound = |v: &SqlValue| -> Result<Vec<u8>, LiteError> {
        let ndb_val = sql_value_to_value(v)?;
        zerompk::to_msgpack_vec(&ndb_val).map_err(|e| LiteError::Serialization {
            detail: format!("encode range bound: {e}"),
        })
    };

    let lo_bytes: Option<Vec<u8>> = lower.map(encode_bound).transpose()?;
    let hi_bytes: Option<Vec<u8>> = upper.map(encode_bound).transpose()?;

    let op = DocumentOp::RangeScan {
        collection: col,
        field: fld,
        lower: lo_bytes,
        upper: hi_bytes,
        limit,
        // Lite lowers this scan itself and applies no row-level security, so it
        // declares none rather than inventing filters it cannot enforce.
        rls_filters: Vec::new(),
    };

    let mut phys = LiteDataPlaneVisitor { engine };
    let fut = phys.document(&op)?;

    Ok(Box::pin(fut))
}

// ── Cte ──────────────────────────────────────────────────────────────────────

/// Non-recursive CTE lowering for Lite single-node.
///
/// Each CTE definition is executed in order (surfacing any errors it would
/// produce), then the outer query — which the planner has already resolved
/// against the CTE bodies — is executed. On Lite, the SQL planner inlines
/// non-recursive CTEs into the outer plan, so the outer query is always
/// self-contained; the definition executions here serve as an eager
/// validation pass.
pub(super) fn lower_cte<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    definitions: &[(String, SqlPlan)],
    outer: &SqlPlan,
) -> Result<LiteFut<'a>, LiteError> {
    let definitions = definitions.to_vec();
    let outer = outer.clone();

    Ok(Box::pin(async move {
        for (_name, def_plan) in &definitions {
            let _ = engine.execute_plan(def_plan).await?;
        }
        engine.execute_plan(&outer).await
    }))
}

// ── Subquery ─────────────────────────────────────────────────────────────────

/// Relational post-processing over a subquery / derived-table body, for Lite
/// single-node.
///
/// The body is materialized by executing `input`, then the outer constraints
/// the body's leaf could not absorb are applied in the same order the
/// distributed engine's `ProviderScan` tail uses — filter → window functions
/// → offset → sort → distinct → project → limit — so both engines answer an
/// `ORDER BY` / `OFFSET` / `DISTINCT` over a subquery identically. Lite has no shards, so
/// there is no gather step: the body already produces the full row stream.
pub(super) fn lower_subquery<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: SubqueryVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    let SubqueryVisitArgs {
        input,
        filters,
        projection,
        window_functions,
        sort_keys,
        offset,
        distinct,
        limit,
    } = args;
    let input = input.clone();
    let filters = filters.to_vec();
    let projection = subquery_projection_columns(projection)?;
    let window_functions = window_functions.to_vec();
    let sort_keys = sort_keys.to_vec();

    Ok(Box::pin(async move {
        let mut result = engine.execute_plan(&input).await?;

        filter_rows(&mut result, &filters)?;

        apply_window_functions(&mut result, &window_functions)?;

        if offset > 0 {
            result.rows = result.rows.into_iter().skip(offset).collect();
        }

        sort_rows(&mut result, &sort_keys)?;

        if distinct {
            distinct_rows(&mut result, &projection);
        }

        project_rows(&mut result, &projection);

        if let Some(n) = limit {
            result.rows.truncate(n);
        }

        Ok(result)
    }))
}

/// Lower an outer target list to bare column names. `Star` (and a qualified
/// star) means "inherit the body's columns" and yields an empty list.
///
/// A computed projection is rejected rather than silently dropped: the tail
/// reshapes materialized rows and has no expression evaluator, so the
/// expression must be projected inside the subquery instead.
fn subquery_projection_columns(projection: &[Projection]) -> Result<Vec<String>, LiteError> {
    let mut names = Vec::with_capacity(projection.len());
    for p in projection {
        match p {
            Projection::Column(qname) => {
                names.push(qname.rsplit('.').next().unwrap_or(qname).to_string());
            }
            Projection::Star | Projection::QualifiedStar(_) => return Ok(Vec::new()),
            Projection::Computed { .. } | Projection::CpComputed { .. } => {
                return Err(LiteError::BadRequest {
                    detail: "a computed projection over an ORDER BY / OFFSET / DISTINCT subquery \
                             is not supported; select the base columns in the subquery and \
                             compute them in an outer SELECT"
                        .into(),
                });
            }
        }
    }
    Ok(names)
}
