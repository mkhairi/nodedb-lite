// SPDX-License-Identifier: Apache-2.0
use super::super::writes::{point_delete_admitted, point_insert_admitted, point_update_admitted};
use super::UpdateValue;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;
use nodedb_physical::physical_plan::document::merge_types::MergeActionOp;
use nodedb_types::value::Value;
use std::collections::HashMap;
/// Rewrite `updates` to resolve source-qualified expressions using the source row.
///
/// For `UpdateValue::Literal` arms: pass through unchanged.
/// For `UpdateValue::Expr` arms: we don't have a full expression evaluator,
/// so only literals are applied. This matches the existing `bulk_update` behaviour.
pub(super) fn qualify_updates_with_source(
    updates: &[(String, UpdateValue)],
    _source_val: &HashMap<String, Value>,
    _source_alias: &str,
) -> Result<Vec<(String, UpdateValue)>, LiteError> {
    // Lite's expression evaluator handles only Literal arms (same as bulk_update /
    // point_update). Non-literal arms are silently skipped — they carry origin-side
    // plan expressions that reference the execution context unavailable in Lite.
    Ok(updates.to_vec())
}

/// Resolve a parallel (columns, values) pair into a field map, evaluating each
/// value against the source row.
///
/// `UpdateValue::Literal` arms decode the pre-encoded msgpack directly.
/// `UpdateValue::Expr` arms (e.g. `s.new_embedding`, `s.qty * 2`) are evaluated
/// against the source row — Lite's `convert_sql_expr` strips table qualifiers,
/// so column references resolve against the source row's bare field names. The
/// result is stored under the *target* column name.
pub(in crate::query) fn build_insert_map(
    columns: &[String],
    values: &[UpdateValue],
    source_val: &HashMap<String, Value>,
) -> Result<HashMap<String, Value>, LiteError> {
    let source_ndb = Value::Object(source_val.clone());
    let mut map = HashMap::with_capacity(columns.len());
    for (col, val) in columns.iter().zip(values.iter()) {
        let resolved: Value = match val {
            UpdateValue::Literal(bytes) => {
                zerompk::from_msgpack(bytes).map_err(|e| LiteError::Serialization {
                    detail: format!("merge insert decode column '{col}': {e}"),
                })?
            }
            UpdateValue::Expr(expr) => expr.eval(&source_ndb)?,
        };
        map.insert(col.clone(), resolved);
    }
    Ok(map)
}

/// Apply a single MERGE arm action to a target document.
pub(super) async fn apply_merge_action<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    doc_id: &str,
    action: &MergeActionOp,
    source_val: &HashMap<String, Value>,
    source_alias: &str,
) -> Result<(), LiteError> {
    match action {
        MergeActionOp::Update { updates } => {
            let effective = qualify_updates_with_source(updates, source_val, source_alias)?;
            point_update_admitted(engine, permit, collection, doc_id, &effective).await?;
        }
        MergeActionOp::Delete => {
            point_delete_admitted(engine, permit, collection, doc_id).await?;
        }
        MergeActionOp::Insert { columns, values } => {
            let map = build_insert_map(columns, values, source_val)?;
            let bytes = zerompk::to_msgpack_vec(&Value::Object(map)).map_err(|e| {
                LiteError::Serialization {
                    detail: format!("merge action insert serialize: {e}"),
                }
            })?;
            point_insert_admitted(engine, permit, collection, doc_id, &bytes, true).await?;
        }
        MergeActionOp::DoNothing => {}
    }
    Ok(())
}
