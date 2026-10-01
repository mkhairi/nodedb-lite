// SPDX-License-Identifier: Apache-2.0
use nodedb_physical::PhysicalTaskVisitor;
use nodedb_physical::physical_plan::{VectorOp, VectorWriteTargets};
use nodedb_sql::types::filter::{Filter, FilterExpr};
use nodedb_sql::types_expr::SqlValue;
use nodedb_sql::{VectorPrimaryDeleteVisitArgs, VectorPrimaryUpdateVisitArgs};
use nodedb_types::RlsWriteCheck;

use super::identity::{key_column, rows_affected};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::physical_visitor::LiteDataPlaneVisitor;
use crate::query::scan_filter_convert::encode_scan_filters;
use crate::query::visitor::adapter::LiteFut;
use crate::query::visitor::dml::convert_assignments;
use crate::storage::engine::StorageEngine;
// ── VectorPrimaryDelete / VectorPrimaryUpdate ─────────────────────────────────

/// The rows a `DELETE` / `UPDATE` targets. Point keys become a predicate on
/// the key column, the only key binding Lite keeps; a statement with no
/// keys carries its WHERE clause.
fn write_targets(
    filters: &[Filter],
    target_keys: &[SqlValue],
    key_column: &str,
) -> Result<VectorWriteTargets, LiteError> {
    if target_keys.is_empty() {
        return Ok(VectorWriteTargets::Predicate(encode_scan_filters(filters)?));
    }
    let keys = Filter {
        expr: FilterExpr::InList {
            field: key_column.to_string(),
            values: target_keys.to_vec(),
        },
    };
    Ok(VectorWriteTargets::Predicate(encode_scan_filters(&[keys])?))
}

/// Lower `SqlPlan::VectorPrimaryDelete` to `VectorOp::DirectDelete`.
pub(in crate::query::visitor) fn lower_vector_primary_delete<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    args: VectorPrimaryDeleteVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    let VectorPrimaryDeleteVisitArgs {
        collection,
        field,
        filters,
        target_keys,
        primary_key,
    } = args;
    let targets = write_targets(filters, target_keys, key_column(primary_key))?;
    let op = VectorOp::DirectDelete {
        collection: nodedb_types::QualifiedCollection::new(
            nodedb_types::DatabaseId::DEFAULT,
            collection,
        ),
        field: field.to_string(),
        targets,
        returning: None,
        rls_filters: Vec::new(),
        rls_write_check: RlsWriteCheck::NoPolicyApplies,
    };
    Ok(Box::pin(async move {
        let mut phys = LiteDataPlaneVisitor {
            engine,
            permit: Some(permit),
        };
        let result = phys.vector(&op)?.await?;
        Ok(rows_affected(result.rows_affected, "DELETE"))
    }))
}

/// Lower `SqlPlan::VectorPrimaryTruncate` to `VectorOp::DirectTruncate`.
/// The physical op answers with the bare `TRUNCATE` tag and applies
/// `RESTART IDENTITY` itself.
pub(in crate::query::visitor) fn lower_vector_primary_truncate<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    field: &str,
    restart_identity: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let op = VectorOp::DirectTruncate {
        collection: nodedb_types::QualifiedCollection::new(
            nodedb_types::DatabaseId::DEFAULT,
            collection,
        ),
        field: field.to_string(),
        restart_identity,
    };
    Ok(Box::pin(async move {
        let mut phys = LiteDataPlaneVisitor {
            engine,
            permit: Some(permit),
        };
        phys.vector(&op)?.await
    }))
}

/// Lower `SqlPlan::VectorPrimaryUpdate` to `VectorOp::DirectUpdate`.
pub(in crate::query::visitor) fn lower_vector_primary_update<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    args: VectorPrimaryUpdateVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    let VectorPrimaryUpdateVisitArgs {
        collection,
        field,
        quantization,
        storage_dtype,
        payload_indexes,
        new_vector,
        assignments,
        filters,
        target_keys,
        returning,
        primary_key,
    } = args;
    if returning {
        return Err(LiteError::Unsupported {
            detail: "VectorPrimaryUpdate: RETURNING is unsupported on the Lite engine".into(),
        });
    }
    let targets = write_targets(filters, target_keys, key_column(primary_key))?;
    let op = VectorOp::DirectUpdate {
        collection: nodedb_types::QualifiedCollection::new(
            nodedb_types::DatabaseId::DEFAULT,
            collection,
        ),
        field: field.to_string(),
        targets,
        new_vector: new_vector.map(<[f32]>::to_vec),
        payload_patch: convert_assignments(assignments)?,
        quantization,
        storage_dtype,
        payload_indexes: payload_indexes.to_vec(),
        returning: None,
        rls_filters: Vec::new(),
        rls_write_check: RlsWriteCheck::NoPolicyApplies,
    };
    Ok(Box::pin(async move {
        let mut phys = LiteDataPlaneVisitor {
            engine,
            permit: Some(permit),
        };
        let result = phys.vector(&op)?.await?;
        Ok(rows_affected(result.rows_affected, "UPDATE"))
    }))
}

#[cfg(test)]
mod tests {
    use super::super::identity::support::*;
    use super::*;
    use crate::query::engine::test_engine;
    use nodedb_sql::types::filter::CompareOp;
    use nodedb_sql::types::plan::{VectorPrimaryInsertIntent, VectorPrimaryRow};
    use nodedb_sql::types_expr::SqlExpr;
    use nodedb_types::value::Value;
    #[tokio::test]
    async fn delete_by_key_counts_only_rows_that_existed() {
        let engine = test_engine().await;
        let rows = vec![row("a", vec![0.1, 0.2], &[])];
        insert(&engine, &rows, VectorPrimaryInsertIntent::Insert, &[])
            .await
            .expect("insert");
        let first = delete(&engine, &[], &[key("a")]).await.expect("delete");
        assert_eq!(first.rows_affected, 1);
        assert_eq!(live_nodes(&engine), 0);
        assert!(stored_field(&engine, "a", "id").is_none());
        let second = delete(&engine, &[], &[key("a")])
            .await
            .expect("delete again");
        assert_eq!(second.rows_affected, 0);
    }

    #[tokio::test]
    async fn delete_by_predicate_removes_matching_rows() {
        let engine = test_engine().await;
        let rows = vec![
            row(
                "a",
                vec![0.1, 0.2],
                &[("tier", SqlValue::String("gold".into()))],
            ),
            row(
                "b",
                vec![0.3, 0.4],
                &[("tier", SqlValue::String("free".into()))],
            ),
        ];
        insert(&engine, &rows, VectorPrimaryInsertIntent::Insert, &[])
            .await
            .expect("insert");
        let filters = vec![Filter {
            expr: FilterExpr::Comparison {
                field: "tier".into(),
                op: CompareOp::Eq,
                value: SqlValue::String("free".into()),
            },
        }];
        let qr = delete(&engine, &filters, &[]).await.expect("delete");
        assert_eq!(qr.rows_affected, 1);
        assert!(stored_field(&engine, "b", "id").is_none());
        assert!(stored_field(&engine, "a", "id").is_some());
        assert_eq!(live_nodes(&engine), 1);
    }

    #[tokio::test]
    async fn truncate_empties_the_collection_and_restarts_its_sequences() {
        use crate::sequence::LiteSequenceDef;
        let engine = test_engine().await;
        engine.sequences.register(LiteSequenceDef {
            name: format!("{COLLECTION}_id_seq"),
            start_value: 1,
            increment: 1,
            min_value: 1,
            max_value: i64::MAX,
            cycle: false,
        });
        engine
            .sequences
            .nextval(&format!("{COLLECTION}_id_seq"))
            .expect("nextval");
        let rows: Vec<VectorPrimaryRow> = (1..=3)
            .map(|i| row(&format!("r{i}"), vec![i as f32 * 0.1, i as f32 * 0.2], &[]))
            .collect();
        insert(&engine, &rows, VectorPrimaryInsertIntent::Insert, &[])
            .await
            .expect("insert");

        let guard = engine.fts_state.admit_mutation().await;
        let result =
            lower_vector_primary_truncate(&engine, guard.permit(), COLLECTION, FIELD, true)
                .expect("lower")
                .await;
        let qr = guard.finish(result).expect("truncate");
        assert_eq!(qr.rows_affected, 0);
        assert_eq!(qr.command.as_deref(), Some("TRUNCATE"));
        assert!(qr.columns.is_empty() && qr.rows.is_empty());
        assert_eq!(live_nodes(&engine), 0);
        assert!(stored_field(&engine, "r1", "id").is_none());
        assert_eq!(
            engine
                .sequences
                .nextval(&format!("{COLLECTION}_id_seq"))
                .expect("nextval"),
            1,
            "RESTART IDENTITY makes the start value the next value"
        );

        let again = vec![row("z", vec![0.5, 0.5], &[])];
        insert(&engine, &again, VectorPrimaryInsertIntent::Insert, &[])
            .await
            .expect("insert after truncate");
        assert_eq!(live_nodes(&engine), 1);
        assert!(stored_field(&engine, "z", "id").is_some());
    }

    #[tokio::test]
    async fn update_re_embed_keeps_one_node_and_patches_payload() {
        let engine = test_engine().await;
        let rows = vec![row("a", vec![0.1, 0.2], &[("n", SqlValue::Int(1))])];
        insert(&engine, &rows, VectorPrimaryInsertIntent::Insert, &[])
            .await
            .expect("insert");
        let patch = vec![("n".to_string(), SqlExpr::Literal(SqlValue::Int(5)))];
        let qr = update(&engine, Some(&[0.7, 0.7]), &patch, &[key("a")])
            .await
            .expect("update");
        assert_eq!(qr.rows_affected, 1);
        assert_eq!(live_nodes(&engine), 1);
        assert_eq!(stored_field(&engine, "a", "n"), Some(Value::Integer(5)));
        let missing = update(&engine, None, &patch, &[key("zzz")])
            .await
            .expect("update missing");
        assert_eq!(missing.rows_affected, 0);
    }
}
