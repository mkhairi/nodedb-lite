//! DML execution methods for `LiteQueryEngine` (INSERT / UPDATE / DELETE /
//! TRUNCATE). Split out of `engine.rs` as a second inherent `impl` block to
//! keep that file under the size limit; behavior is unchanged.

use nodedb_sql::types::{EngineType, SqlValue, WriteRoute};
use nodedb_types::result::QueryResult;

use super::dml_targets::document_targets;
use super::engine::{LiteQueryEngine, sql_value_to_string};
use super::text_index::reindex_documents;
use crate::engine::crdt::{CrdtRowOp, CrdtRowWrite};
use crate::error::LiteError;
use crate::storage::engine::StorageEngine;
use nodedb_sql::types::filter::Filter;

impl<S: StorageEngine> LiteQueryEngine<S> {
    /// `route` is the planner's `EngineRules` decision and picks the store
    /// family: `ColumnarFamily` writes one columnar batch, `Document` writes
    /// per row. Within `Document`, `engine` picks the strict store or the
    /// CRDT store because Lite keeps those as separate engines.
    pub(super) async fn execute_insert(
        &self,
        collection: &str,
        engine: &EngineType,
        route: WriteRoute,
        rows: &[Vec<(String, SqlValue)>],
        if_absent: bool,
        primary_key: Option<&str>,
    ) -> Result<QueryResult, LiteError> {
        match route {
            WriteRoute::ColumnarFamily => {
                let (result, written) =
                    super::columnar_dml::insert_columnar(&self.columnar, collection, rows)?;
                super::text_index::index_columnar_rows(
                    self,
                    collection,
                    written.iter().map(Vec::as_slice),
                )?;
                // Durable outbound enqueue must run here (async) — the sync insert
                // path cannot await. Covers the SQL-INSERT route to Origin sync.
                #[cfg(not(target_arch = "wasm32"))]
                crate::sync::reconcile_outbound_enqueue(
                    self.columnar.enqueue_outbound(collection, &written).await,
                    "columnar insert (sql)",
                    collection,
                    "",
                )?;
                return Ok(result);
            }
            WriteRoute::Document => {}
        }
        if *engine == EngineType::DocumentStrict {
            return super::strict_dml::insert_strict(self, collection, rows, if_absent).await;
        }
        // CRDT / schemaless path.
        let mut crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        // Every row is checked before any is written, so a statement that
        // fails writes nothing.
        let mut planned: Vec<(String, Vec<(&str, loro::LoroValue)>)> =
            Vec::with_capacity(rows.len());
        for row in rows {
            let id = row
                .iter()
                .find(|(k, _)| match primary_key {
                    Some(pk) => k == pk,
                    None => k == "id",
                })
                .map(|(_, v)| sql_value_to_string(v))
                .unwrap_or_default();
            if crdt.exists(collection, &id) || planned.iter().any(|(p, _)| *p == id) {
                if if_absent {
                    continue;
                }
                return Err(LiteError::UniqueViolation {
                    collection: collection.to_string(),
                    detail: format!("primary key (id)=({id}) already exists"),
                });
            }
            let fields = row
                .iter()
                .map(|(k, v)| (k.as_str(), sql_value_to_loro(v)))
                .collect();
            planned.push((id, fields));
        }
        let checks: Vec<CrdtRowOp<'_>> = planned
            .iter()
            .map(|(id, fields)| {
                (
                    CrdtRowWrite::Upsert,
                    collection,
                    id.as_str(),
                    fields.as_slice(),
                )
            })
            .collect();
        crdt.check_unique_writes(&checks)?;
        let mut written: Vec<String> = Vec::with_capacity(planned.len());
        for (id, fields) in &planned {
            crdt.upsert(collection, id, fields)?;
            written.push(id.clone());
        }
        let affected = written.len() as u64;
        drop(crdt);
        reindex_documents(self, collection, written.iter().map(String::as_str))?;
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("INSERT".into()),
        })
    }

    pub(super) async fn execute_update(
        &self,
        collection: &str,
        engine: &EngineType,
        assignments: &[(String, nodedb_sql::types::SqlExpr)],
        filters: &[Filter],
        target_keys: &[SqlValue],
    ) -> Result<QueryResult, LiteError> {
        if is_columnar_family(engine) {
            return super::columnar_dml::update_columnar(
                self,
                collection,
                assignments,
                filters,
                target_keys,
            )
            .await;
        }
        if *engine == EngineType::DocumentStrict {
            return super::strict_dml::update_strict(
                self,
                collection,
                assignments,
                filters,
                target_keys,
            )
            .await;
        }
        let targets = document_targets(self, collection, filters, target_keys)?;
        // CRDT / schemaless path.
        let mut crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        // UPDATE changes existing rows only, and only the assigned fields.
        let fields: Vec<(&str, loro::LoroValue)> = assignments
            .iter()
            .filter_map(|(field, expr)| {
                if let nodedb_sql::types::SqlExpr::Literal(val) = expr {
                    Some((field.as_str(), sql_value_to_loro(val)))
                } else {
                    None
                }
            })
            .collect();
        let written: Vec<String> = targets
            .into_iter()
            .filter(|key| crdt.exists(collection, key))
            .collect();
        // Every row is checked before any is written, so a statement that
        // fails writes nothing.
        let checks: Vec<CrdtRowOp<'_>> = written
            .iter()
            .map(|id| {
                (
                    CrdtRowWrite::SetFields,
                    collection,
                    id.as_str(),
                    fields.as_slice(),
                )
            })
            .collect();
        crdt.check_unique_writes(&checks)?;
        for key_str in &written {
            crdt.set_fields(collection, key_str, &fields)?;
        }
        let affected = written.len() as u64;
        drop(crdt);
        reindex_documents(self, collection, written.iter().map(String::as_str))?;
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("UPDATE".into()),
        })
    }

    pub(super) async fn execute_delete(
        &self,
        collection: &str,
        engine: &EngineType,
        filters: &[Filter],
        target_keys: &[SqlValue],
    ) -> Result<QueryResult, LiteError> {
        if is_columnar_family(engine) {
            return super::columnar_dml::delete_columnar(self, collection, filters, target_keys)
                .await;
        }
        if *engine == EngineType::DocumentStrict {
            return super::strict_dml::delete_strict(self, collection, filters, target_keys).await;
        }
        let targets = document_targets(self, collection, filters, target_keys)?;
        // CRDT / schemaless path.
        let mut crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        let mut affected = 0;
        let mut removed: Vec<String> = Vec::with_capacity(targets.len());
        for key_str in targets {
            if !crdt.exists(collection, &key_str) {
                continue;
            }
            crdt.delete(collection, &key_str)?;
            affected += 1;
            removed.push(key_str);
        }
        drop(crdt);
        reindex_documents(self, collection, removed.iter().map(String::as_str))?;
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("DELETE".into()),
        })
    }
}
fn sql_value_to_loro(v: &SqlValue) -> loro::LoroValue {
    match v {
        SqlValue::Int(i) => loro::LoroValue::I64(*i),
        SqlValue::Float(f) => loro::LoroValue::Double(*f),
        SqlValue::String(s) => loro::LoroValue::String(s.clone().into()),
        SqlValue::Bool(b) => loro::LoroValue::Bool(*b),
        SqlValue::Null => loro::LoroValue::Null,
        SqlValue::Array(items) => loro::LoroValue::List(
            items
                .iter()
                .map(sql_value_to_loro)
                .collect::<Vec<_>>()
                .into(),
        ),
        _ => loro::LoroValue::Null,
    }
}

/// Whether `engine` is stored in the columnar engine: plain, timeseries, and
/// spatial collections alike.
fn is_columnar_family(engine: &EngineType) -> bool {
    matches!(
        engine,
        EngineType::Columnar | EngineType::Timeseries | EngineType::Spatial
    )
}
