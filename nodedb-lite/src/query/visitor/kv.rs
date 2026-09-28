// SPDX-License-Identifier: Apache-2.0
//! SQL-visitor lowering for KV SqlPlan variants: KvInsert.

use nodedb_physical::PhysicalTaskVisitor;
use nodedb_physical::physical_plan::KvOp;
use nodedb_physical::physical_plan::document::UpdateValue;
use nodedb_sql::types::KvInsertIntent;
use nodedb_sql::types_expr::{SqlExpr, SqlValue};
use nodedb_types::Surrogate;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::expr_convert::convert_sql_expr;
use crate::query::filter_convert::sql_value_to_value;
use crate::query::kv_ops::sql_read::kv_key_bytes;
use crate::query::physical_visitor::LiteDataPlaneVisitor;
use crate::storage::engine::StorageEngine;

use super::adapter::LiteFut;

// ── Value encoding ────────────────────────────────────────────────────────────

/// Raw bytes for a lone `value` column, by Origin's rule: a scalar encodes
/// as `nodedb_types::scalar_to_raw_bytes` writes it, and an array is
/// PostgreSQL array text.
fn sql_value_raw_bytes(v: &SqlValue) -> Vec<u8> {
    match v {
        SqlValue::Bytes(b) => b.clone(),
        SqlValue::Array(values) => pg_array_text(values).into_bytes(),
        SqlValue::Int(_)
        | SqlValue::Float(_)
        | SqlValue::Decimal(_)
        | SqlValue::String(_)
        | SqlValue::Bool(_)
        | SqlValue::Null
        | SqlValue::Timestamp(_)
        | SqlValue::Timestamptz(_) => pg_text(v).into_bytes(),
    }
}

/// The text form of one SQL value, as Origin's `sql_value_to_string` writes
/// it.
fn pg_text(v: &SqlValue) -> String {
    match v {
        SqlValue::String(s) => s.clone(),
        SqlValue::Int(i) => i.to_string(),
        SqlValue::Float(f) => f.to_string(),
        SqlValue::Decimal(d) => d.to_string(),
        SqlValue::Bool(b) => b.to_string(),
        SqlValue::Timestamp(at) | SqlValue::Timestamptz(at) => at.to_iso8601(),
        SqlValue::Bytes(b) => {
            let hex: String = b.iter().map(|byte| format!("{byte:02x}")).collect();
            format!("\\x{hex}")
        }
        SqlValue::Array(values) => pg_array_text(values),
        SqlValue::Null => String::new(),
    }
}

/// PostgreSQL array text: `{a,"two words",NULL}`.
fn pg_array_text(values: &[SqlValue]) -> String {
    let elements: Vec<String> = values
        .iter()
        .map(|value| match value {
            SqlValue::Null => "NULL".to_string(),
            SqlValue::String(s) if pg_array_string_needs_quotes(s) => {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            }
            other => pg_text(other),
        })
        .collect();
    format!("{{{}}}", elements.join(","))
}

fn pg_array_string_needs_quotes(value: &str) -> bool {
    value.is_empty()
        || value.eq_ignore_ascii_case("null")
        || value
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, ',' | '{' | '}' | '"' | '\\'))
}

/// Encode a KV insert's value columns as the stored body.
///
/// A lone `value` column stores its raw bytes. Any other column set stores
/// the map body `kv_ops::body::encode_kv_body` writes, the same body an
/// Origin row apply stores.
fn encode_kv_value(value_cols: &[(String, SqlValue)]) -> Result<Vec<u8>, LiteError> {
    if value_cols.len() == 1 && value_cols[0].0 == "value" {
        return Ok(sql_value_raw_bytes(&value_cols[0].1));
    }
    let mut map = std::collections::HashMap::with_capacity(value_cols.len());
    for (col, sv) in value_cols {
        map.insert(col.clone(), sql_value_to_value(sv)?);
    }
    crate::query::kv_ops::body::encode_kv_body(map, nodedb_query::msgpack_scan::KvBodyShape::Map)
}

/// Convert one `ON CONFLICT DO UPDATE SET col = <expr>` assignment.
///
/// Mirrors Origin's `assignments_to_update_values`: a literal RHS
/// pre-encodes to msgpack (`UpdateValue::Literal`); anything else (a
/// column reference, arithmetic, `EXCLUDED.col`, a function call, ...)
/// converts to a query-side expression (`UpdateValue::Expr`) that
/// `kv_ops::writes::basic::kv_insert_on_conflict_update` evaluates via
/// `query::on_conflict::apply_patch`.
fn assignment_to_update_value(expr: &SqlExpr) -> Result<UpdateValue, LiteError> {
    match expr {
        SqlExpr::Literal(v) => {
            let value = sql_value_to_value(v)?;
            let bytes = zerompk::to_msgpack_vec(&value).map_err(|e| LiteError::Serialization {
                detail: format!("encode ON CONFLICT literal: {e}"),
            })?;
            Ok(UpdateValue::Literal(bytes))
        }
        other => Ok(UpdateValue::Expr(convert_sql_expr(other)?)),
    }
}

// ── KvInsert ─────────────────────────────────────────────────────────────────

/// Lower `SqlPlan::KvInsert` → `KvOp::{Insert, InsertIfAbsent, Put, InsertOnConflictUpdate}`.
pub(super) fn lower_kv_insert<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    entries: &[(SqlValue, Vec<(String, SqlValue)>)],
    ttl_secs: u64,
    intent: KvInsertIntent,
    on_conflict_updates: &[(String, SqlExpr)],
) -> Result<LiteFut<'a>, LiteError> {
    if entries.is_empty() {
        let verb = match intent {
            KvInsertIntent::Insert | KvInsertIntent::InsertIfAbsent => "INSERT",
            KvInsertIntent::Put if !on_conflict_updates.is_empty() => "INSERT",
            KvInsertIntent::Put => "UPSERT",
        };
        return Ok(Box::pin(async move {
            Ok(nodedb_types::result::QueryResult {
                columns: vec![],
                rows: vec![],
                rows_affected: 0,
                command: Some(verb.into()),
            })
        }));
    }

    let ttl_ms = ttl_secs * 1000;
    // Lite holds a bare collection name; DatabaseId::DEFAULT keeps it unqualified.
    let collection =
        nodedb_types::QualifiedCollection::new(nodedb_types::DatabaseId::DEFAULT, collection);

    // Convert `ON CONFLICT DO UPDATE SET` assignments once for the whole
    // statement, mirroring Origin's `assignments_to_update_values`: a
    // literal RHS pre-encodes to msgpack, anything else carries the
    // expression for `kv_insert_on_conflict_update` to evaluate against
    // the existing row (`col`) and the incoming row (`EXCLUDED.col`).
    let updates: Vec<(String, UpdateValue)> = on_conflict_updates
        .iter()
        .map(|(col, expr)| Ok((col.clone(), assignment_to_update_value(expr)?)))
        .collect::<Result<Vec<_>, LiteError>>()?;

    // Pre-encode all entries so errors surface before the future is spawned.
    let mut ops: Vec<KvOp> = Vec::with_capacity(entries.len());

    for (key_val, value_cols) in entries {
        let key = kv_key_bytes(key_val);
        let value = encode_kv_value(value_cols)?;
        let updates = updates.clone();

        let op = match intent {
            KvInsertIntent::Insert => KvOp::Insert {
                collection: collection.clone(),
                key,
                value,
                ttl_ms,
                surrogate: Surrogate::ZERO,
                returning: None,
                rls_filters: Vec::new(),
            },
            KvInsertIntent::InsertIfAbsent => KvOp::InsertIfAbsent {
                collection: collection.clone(),
                key,
                value,
                ttl_ms,
                surrogate: Surrogate::ZERO,
                returning: None,
                rls_filters: Vec::new(),
            },
            KvInsertIntent::Put if !updates.is_empty() => KvOp::InsertOnConflictUpdate {
                collection: collection.clone(),
                key,
                value,
                ttl_ms,
                updates,
                surrogate: Surrogate::ZERO,
                // Lite has no RLS policy engine: no write policy applies.
                rls_write_check: nodedb_types::RlsWriteCheck::NoPolicyApplies,
                returning: None,
                rls_filters: Vec::new(),
            },
            KvInsertIntent::Put => KvOp::Put {
                collection: collection.clone(),
                key,
                value,
                ttl_ms,
                surrogate: Surrogate::ZERO,
                returning: None,
                rls_filters: Vec::new(),
                provenance: None,
            },
        };
        ops.push(op);
    }

    // Execute all ops sequentially, accumulating rows_affected. Fold each
    // op's reported verb the same way Postgres folds a multi-row
    // `INSERT ... ON CONFLICT DO UPDATE`: `INSERT`/`UPDATE` mix collapses to
    // `INSERT`, any other verb is uniform across every op in one statement.
    Ok(Box::pin(async move {
        let mut total: u64 = 0;
        let mut verb: Option<&'static str> = None;
        for op in ops {
            let mut phys = LiteDataPlaneVisitor { engine };
            let result = phys.kv(&op)?.await?;
            total += result.rows_affected;
            if let Some(op_verb) = result.command.as_deref() {
                verb = Some(match (verb, op_verb) {
                    (None, "UPDATE") => "UPDATE",
                    (None, "UPSERT") => "UPSERT",
                    (None, _) => "INSERT",
                    (Some("INSERT"), "UPDATE") | (Some("UPDATE"), "INSERT") => "INSERT",
                    (Some(prev), _) => prev,
                });
            }
        }
        Ok(nodedb_types::result::QueryResult {
            columns: vec![],
            rows: vec![],
            rows_affected: total,
            command: Some(verb.unwrap_or("INSERT").into()),
        })
    }))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use nodedb_sql::types::KvInsertIntent;
    use nodedb_sql::types_expr::SqlValue;

    use crate::PagedbStorageMem;
    use crate::engine::array::engine::ArrayEngineState;
    use crate::engine::fts::FtsState;
    use crate::engine::spatial::SpatialIndexManager;
    use crate::engine::vector::VectorState;
    use crate::query::engine::{LiteQueryEngine, LiteQueryEngineParams};

    async fn make_engine() -> LiteQueryEngine<PagedbStorageMem> {
        let storage = Arc::new(
            PagedbStorageMem::open_in_memory()
                .await
                .expect("in-memory pagedb"),
        );
        let crdt = Arc::new(Mutex::new(
            crate::engine::crdt::CrdtEngine::new(1).expect("crdt"),
        ));
        let governor = crate::query::engine::test_governor();
        let strict = Arc::new(crate::engine::strict::StrictEngine::new(Arc::clone(
            &storage,
        )));
        let columnar = Arc::new(crate::engine::columnar::ColumnarEngine::new(
            Arc::clone(&storage),
            crate::query::engine::test_scoped_memory(&governor, nodedb_mem::EngineId::Columnar),
        ));
        let htap = Arc::new(crate::engine::htap::HtapBridge::new());
        let timeseries = Arc::new(Mutex::new(
            crate::engine::timeseries::engine::TimeseriesEngine::new(),
        ));
        let vector_state = Arc::new(VectorState::new(
            Arc::clone(&storage),
            100,
            crate::query::engine::test_scoped_memory(&governor, nodedb_mem::EngineId::Vector),
        ));
        let array_state = Arc::new(tokio::sync::Mutex::new(
            ArrayEngineState::open(&storage).await.expect("array"),
        ));
        let fts_state = Arc::new(FtsState::new(Arc::clone(&governor)));
        let spatial = Arc::new(Mutex::new(SpatialIndexManager::new(
            crate::query::engine::test_scoped_memory(&governor, nodedb_mem::EngineId::Spatial),
        )));
        LiteQueryEngine::new(LiteQueryEngineParams {
            crdt,
            strict,
            columnar,
            htap,
            storage,
            timeseries,
            vector_state,
            array_state,
            fts_state,
            sparse_state: Arc::new(crate::engine::sparse_vector::SparseVectorState::new()),
            spatial,
            csr: Arc::new(Mutex::new(std::collections::HashMap::new())),
            governor,
            kv_local: crate::query::engine::test_kv_local(),
        })
    }

    #[test]
    fn a_lone_value_column_stores_origin_raw_bytes() {
        let cases = [
            (SqlValue::String("v1".into()), b"v1".to_vec()),
            (SqlValue::Int(7), b"7".to_vec()),
            (SqlValue::Float(1.5), b"1.5".to_vec()),
            (SqlValue::Bool(false), b"false".to_vec()),
            (SqlValue::Bytes(vec![0xff]), vec![0xff]),
            (SqlValue::Null, Vec::new()),
            (
                SqlValue::Array(vec![
                    SqlValue::String("public".into()),
                    SqlValue::String("two words".into()),
                    SqlValue::Null,
                ]),
                b"{public,\"two words\",NULL}".to_vec(),
            ),
        ];
        for (sql, expected) in cases {
            assert_eq!(super::sql_value_raw_bytes(&sql), expected, "{sql:?}");
        }
    }

    #[tokio::test]
    async fn test_kv_insert_plain() {
        let engine = make_engine().await;
        let entries = vec![(
            SqlValue::String("key1".to_string()),
            vec![("value".to_string(), SqlValue::String("hello".to_string()))],
        )];
        let fut = super::lower_kv_insert(&engine, "mykv", &entries, 0, KvInsertIntent::Put, &[])
            .expect("lower");
        let r = fut.await.expect("execute");
        assert_eq!(r.rows_affected, 1);
    }

    #[tokio::test]
    async fn test_kv_insert_duplicate_raises() {
        let engine = make_engine().await;
        let entries = vec![(
            SqlValue::String("dup_key".to_string()),
            vec![("value".to_string(), SqlValue::Int(42))],
        )];
        super::lower_kv_insert(&engine, "mykv2", &entries, 0, KvInsertIntent::Put, &[])
            .unwrap()
            .await
            .unwrap();
        // Second INSERT (not PUT) on same key should error.
        let err =
            super::lower_kv_insert(&engine, "mykv2", &entries, 0, KvInsertIntent::Insert, &[])
                .unwrap()
                .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_kv_insert_if_absent_no_op() {
        let engine = make_engine().await;
        let entries = vec![(
            SqlValue::String("absent_key".to_string()),
            vec![("value".to_string(), SqlValue::Int(1))],
        )];
        super::lower_kv_insert(&engine, "mykv3", &entries, 0, KvInsertIntent::Put, &[])
            .unwrap()
            .await
            .unwrap();
        // InsertIfAbsent should succeed silently (0 rows affected).
        let r = super::lower_kv_insert(
            &engine,
            "mykv3",
            &entries,
            0,
            KvInsertIntent::InsertIfAbsent,
            &[],
        )
        .unwrap()
        .await
        .unwrap();
        assert_eq!(r.rows_affected, 0);
    }

    #[tokio::test]
    async fn test_kv_insert_multi_column_value() {
        let engine = make_engine().await;
        let entries = vec![(
            SqlValue::String("mkey".to_string()),
            vec![
                ("field_a".to_string(), SqlValue::Int(10)),
                ("field_b".to_string(), SqlValue::String("foo".to_string())),
            ],
        )];
        let fut = super::lower_kv_insert(&engine, "mykv4", &entries, 0, KvInsertIntent::Put, &[])
            .expect("lower");
        let r = fut.await.expect("execute");
        assert_eq!(r.rows_affected, 1);
    }

    /// A typed row a SQL insert stores takes an `INCR` on its integer
    /// column: the insert and the shared atomics use one body encoding.
    #[tokio::test]
    async fn a_sql_inserted_typed_row_takes_an_incr() {
        use nodedb_physical::physical_plan::KvCounterShape;
        use nodedb_types::value::Value;

        let engine = make_engine().await;
        let entries = vec![(
            SqlValue::String("k".to_string()),
            vec![
                ("n".to_string(), SqlValue::Int(5)),
                ("label".to_string(), SqlValue::String("x".to_string())),
            ],
        )];
        super::lower_kv_insert(&engine, "ctr", &entries, 0, KvInsertIntent::Put, &[])
            .expect("lower insert")
            .await
            .expect("insert");

        crate::query::kv_ops::writes::kv_incr(&engine, "ctr", b"k", 3, 0, &KvCounterShape::Raw)
            .await
            .expect("incr on the integer column");

        let stored = crate::query::kv_ops::reads::kv_get(&engine, "ctr", b"k", None)
            .await
            .expect("get");
        let Value::Bytes(bytes) = &stored.rows[0][1] else {
            panic!("value column is not bytes");
        };
        let row = crate::query::kv_ops::body::decode_kv_map(bytes)
            .expect("decode")
            .expect("map body");
        assert_eq!(row.get("n"), Some(&Value::Integer(8)));
        assert_eq!(row.get("label"), Some(&Value::String("x".into())));
    }

    /// `ON CONFLICT DO UPDATE SET n = n + 1` evaluates against the existing
    /// row, not the incoming one. End-to-end through
    /// `lower_kv_insert` → `KvOp::InsertOnConflictUpdate` → the KV engine.
    #[tokio::test]
    async fn test_kv_insert_on_conflict_expr_evaluates_against_existing_row() {
        use nodedb_sql::types_expr::{BinaryOp, SqlExpr};

        let engine = make_engine().await;
        let seed = vec![(
            SqlValue::String("k".to_string()),
            vec![("n".to_string(), SqlValue::Int(1))],
        )];
        super::lower_kv_insert(&engine, "mykv5", &seed, 0, KvInsertIntent::Put, &[])
            .expect("lower seed")
            .await
            .expect("seed");

        let entries = vec![(
            SqlValue::String("k".to_string()),
            vec![("n".to_string(), SqlValue::Int(99))],
        )];
        let on_conflict = vec![(
            "n".to_string(),
            SqlExpr::BinaryOp {
                left: Box::new(SqlExpr::Column {
                    table: None,
                    name: "n".to_string(),
                }),
                op: BinaryOp::Add,
                right: Box::new(SqlExpr::Literal(SqlValue::Int(1))),
            },
        )];
        let r = super::lower_kv_insert(
            &engine,
            "mykv5",
            &entries,
            0,
            KvInsertIntent::Put,
            &on_conflict,
        )
        .expect("lower")
        .await
        .expect("execute");
        assert_eq!(r.rows_affected, 1);

        let stored = crate::query::kv_ops::reads::kv_get(&engine, "mykv5", b"k", None)
            .await
            .expect("get");
        let nodedb_types::value::Value::Bytes(bytes) = &stored.rows[0][1] else {
            panic!("value column is not bytes");
        };
        let map = crate::query::kv_ops::body::decode_kv_map(bytes)
            .expect("decode row")
            .expect("map body");
        // `n + 1` against the existing row (1), not the incoming row (99).
        assert_eq!(map.get("n"), Some(&nodedb_types::value::Value::Integer(2)));
    }
}
