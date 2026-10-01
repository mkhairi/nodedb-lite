// SPDX-License-Identifier: Apache-2.0

//! `MERGE` lowering.

use std::collections::HashMap;
use std::collections::HashSet;

use nodedb_physical::physical_plan::document::merge_types::{
    MergeActionOp, MergeClauseKind, MergeClauseOp,
};
use nodedb_sql::types::SqlPlan;
use nodedb_sql::types::plan::{MergeClauseKind as SqlMergeKind, MergePlanAction, MergePlanClause};
use nodedb_sql::types::query::EngineType;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::document_ops::sets::{
    build_insert_map, collect_ids_pub, fetch_document_value_pub,
};
use crate::query::document_ops::writes::{
    point_delete_admitted, point_insert_admitted, point_update_admitted,
};
use crate::query::engine::LiteQueryEngine;
use crate::query::value_utils::value_to_string;
use crate::storage::engine::StorageEngine;

use crate::query::visitor::adapter::LiteFut;

use super::rows::{
    convert_assignments, expr_to_update_value, extract_id, result_to_maps, row_to_msgpack,
};

// ── Merge ────────────────────────────────────────────────────────────────────

/// `MERGE INTO target USING source ON ... WHEN ...`
#[allow(clippy::too_many_arguments)]
pub(in crate::query::visitor) fn lower_merge<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    target: &str,
    _engine_type: EngineType,
    source: &SqlPlan,
    target_join_col: &str,
    source_join_col: &str,
    _source_alias: &str,
    clauses: &[MergePlanClause],
    _returning: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let target = target.to_string();
    let source = source.clone();
    let t_join = target_join_col.to_string();
    let s_join = source_join_col.to_string();
    let phys_clauses = convert_merge_clauses(clauses)?;

    Ok(Box::pin(async move {
        let source_result = engine.execute_plan_admitted(&source, Some(permit)).await?;
        let source_maps = result_to_maps(source_result);

        let mut source_index: HashMap<String, HashMap<String, Value>> = HashMap::new();
        for row in source_maps {
            if let Some(key_val) = row.get(&s_join) {
                source_index.insert(value_to_string(key_val), row);
            }
        }

        let target_ids = collect_ids_pub(engine, &target).await?;
        let mut matched_source_keys: HashSet<String> = HashSet::new();
        let mut affected: u64 = 0;

        for doc_id in &target_ids {
            let target_val = fetch_document_value_pub(engine, &target, doc_id).await?;
            let join_key = match target_val.get(&t_join).map(value_to_string) {
                Some(k) => k,
                None => continue,
            };

            if let Some(source_row) = source_index.get(&join_key) {
                matched_source_keys.insert(join_key);
                let arm = phys_clauses
                    .iter()
                    .find(|c| c.kind == MergeClauseKind::Matched);
                if let Some(arm) = arm {
                    apply_merge_action(engine, permit, &target, doc_id, &arm.action, source_row)
                        .await?;
                    affected += 1;
                }
            } else {
                let arm = phys_clauses
                    .iter()
                    .find(|c| c.kind == MergeClauseKind::NotMatchedBySource);
                if let Some(arm) = arm {
                    // No source row for this target — NOT MATCHED BY SOURCE arms
                    // are UPDATE/DELETE only, so an empty source suffices.
                    apply_merge_action(
                        engine,
                        permit,
                        &target,
                        doc_id,
                        &arm.action,
                        &HashMap::new(),
                    )
                    .await?;
                    affected += 1;
                }
            }
        }

        // Unmatched source rows → WHEN NOT MATCHED.
        let not_matched_arm = phys_clauses
            .iter()
            .find(|c| c.kind == MergeClauseKind::NotMatched);
        if let Some(arm) = not_matched_arm {
            for (source_key, source_row) in &source_index {
                if !matched_source_keys.contains(source_key) {
                    let doc_id = extract_id(source_row);
                    apply_merge_action(engine, permit, &target, &doc_id, &arm.action, source_row)
                        .await?;
                    affected += 1;
                }
            }
        }

        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("MERGE".into()),
        })
    }))
}

pub(super) async fn apply_merge_action<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    target: &str,
    doc_id: &str,
    action: &MergeActionOp,
    source_row: &HashMap<String, Value>,
) -> Result<(), LiteError> {
    match action {
        MergeActionOp::Update { updates } => {
            point_update_admitted(engine, permit, target, doc_id, updates).await?;
        }
        MergeActionOp::Delete => {
            point_delete_admitted(engine, permit, target, doc_id).await?;
        }
        MergeActionOp::Insert { columns, values } => {
            // Evaluate each value against the source row: literals decode
            // directly, expressions (`s.new_embedding`, `s.qty * 2`) evaluate
            // against the bare-keyed source fields. Result keyed by target column.
            let row_map = build_insert_map(columns, values, source_row)?;
            let id = extract_id(&row_map);
            let bytes = row_to_msgpack(row_map)?;
            point_insert_admitted(engine, permit, target, &id, &bytes, true).await?;
        }
        MergeActionOp::DoNothing => {}
    }
    Ok(())
}

pub(super) fn convert_merge_clauses(
    clauses: &[MergePlanClause],
) -> Result<Vec<MergeClauseOp>, LiteError> {
    clauses.iter().map(convert_one_clause).collect()
}

pub(super) fn convert_one_clause(clause: &MergePlanClause) -> Result<MergeClauseOp, LiteError> {
    let kind = match clause.kind {
        SqlMergeKind::Matched => MergeClauseKind::Matched,
        SqlMergeKind::NotMatched => MergeClauseKind::NotMatched,
        SqlMergeKind::NotMatchedBySource => MergeClauseKind::NotMatchedBySource,
    };
    let action = convert_merge_action(&clause.action)?;
    Ok(MergeClauseOp {
        kind,
        extra_predicate: Vec::new(),
        action,
    })
}

pub(super) fn convert_merge_action(action: &MergePlanAction) -> Result<MergeActionOp, LiteError> {
    match action {
        MergePlanAction::Update { assignments } => {
            let updates = convert_assignments(assignments)?;
            Ok(MergeActionOp::Update { updates })
        }
        MergePlanAction::Delete => Ok(MergeActionOp::Delete),
        MergePlanAction::Insert { columns, values } => {
            // Literal values are pre-encoded; source-referencing expressions
            // (`s.new_embedding`, `s.qty * 2`) are carried as `UpdateValue::Expr`
            // and evaluated against the source row at apply time.
            let encoded = values
                .iter()
                .map(expr_to_update_value)
                .collect::<Result<Vec<_>, LiteError>>()?;
            Ok(MergeActionOp::Insert {
                columns: columns.clone(),
                values: encoded,
            })
        }
        MergePlanAction::DoNothing => Ok(MergeActionOp::DoNothing),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::document_ops::writes::point_put;
    use crate::query::engine::test_engine;

    #[tokio::test]
    async fn merge_inserts_unmatched_schemaless_source_row()
    -> Result<(), Box<dyn std::error::Error>> {
        let engine = test_engine().await;
        engine.execute_sql("CREATE COLLECTION merge_target").await?;
        let fields = HashMap::from([
            ("id".into(), Value::String("source-1".into())),
            ("label".into(), Value::String("copied source".into())),
            ("quantity".into(), Value::Integer(7)),
        ]);
        let bytes = zerompk::to_msgpack_vec(&Value::Object(fields.clone()))?;
        point_put(&engine, "merge_source", "source-1", &bytes).await?;

        let result = engine
            .execute_sql(
                "MERGE INTO merge_target AS t \
             USING (SELECT id, label, quantity FROM merge_source) AS s \
             ON t.id = s.id \
             WHEN NOT MATCHED THEN INSERT (id, label, quantity) \
             VALUES (s.id, s.label, s.quantity)",
            )
            .await?;

        assert_eq!(result.rows_affected, 1);
        assert_eq!(result.command.as_deref(), Some("MERGE"));
        assert_eq!(
            collect_ids_pub(&engine, "merge_target").await?,
            vec!["source-1"]
        );
        let inserted = fetch_document_value_pub(&engine, "merge_target", "source-1").await?;
        assert_eq!(inserted.get("id"), fields.get("id"));
        assert_eq!(inserted.get("label"), fields.get("label"));
        assert_eq!(inserted.get("quantity"), fields.get("quantity"));
        Ok(())
    }
}
