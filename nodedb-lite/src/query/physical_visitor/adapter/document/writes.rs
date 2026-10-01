// SPDX-License-Identifier: Apache-2.0
//! DocumentOp dispatch for the Lite physical visitor.

use nodedb_physical::physical_plan::DocumentOp;
use nodedb_types::RlsWriteCheck;

use crate::error::LiteError;
use crate::query::document_ops;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use crate::query::physical_visitor::adapter::LitePhysicalFut;
use crate::query::physical_visitor::adapter::policy::deny_policy;

pub(super) fn dispatch<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: Option<&'a crate::engine::fts::coordinator::TextMutationPermit>,
    op: &DocumentOp,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    match op {
        DocumentOp::PointPut {
            collection,
            document_id,
            value,
            returning,
            rls_filters,
            ..
        } => {
            // PointPut has no rls_write_check slot: unconditional-overwrite
            // upsert semantics carry no write gate.
            deny_policy(
                "DocumentOp::PointPut",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                &RlsWriteCheck::NoPolicyApplies,
            )?;
            let col = collection.clone();
            let doc_id = document_id.clone();
            let val = value.clone();
            Ok(Box::pin(async move {
                document_ops::writes::point_put_coordinated(
                    engine,
                    permit,
                    col.as_str(),
                    &doc_id,
                    &val,
                )
                .await
            }))
        }

        DocumentOp::PointInsert {
            collection,
            document_id,
            value,
            if_absent,
            returning,
            rls_filters,
            ..
        } => {
            // PointInsert has no rls_write_check slot: an unconditional
            // first write carries no write gate.
            deny_policy(
                "DocumentOp::PointInsert",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                &RlsWriteCheck::NoPolicyApplies,
            )?;
            let col = collection.clone();
            let doc_id = document_id.clone();
            let val = value.clone();
            let if_absent = *if_absent;
            Ok(Box::pin(async move {
                document_ops::writes::point_insert_coordinated(
                    engine,
                    permit,
                    col.as_str(),
                    &doc_id,
                    &val,
                    if_absent,
                )
                .await
            }))
        }

        DocumentOp::PointUpdate {
            collection,
            document_id,
            updates,
            returning,
            rls_filters,
            rls_write_check,
            ..
        } => {
            deny_policy(
                "DocumentOp::PointUpdate",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let col = collection.clone();
            let doc_id = document_id.clone();
            let updates = updates.clone();
            Ok(Box::pin(async move {
                document_ops::writes::point_update_coordinated(
                    engine,
                    permit,
                    col.as_str(),
                    &doc_id,
                    &updates,
                )
                .await
            }))
        }

        DocumentOp::PointDelete {
            collection,
            document_id,
            returning,
            rls_filters,
            rls_write_check,
            ..
        } => {
            deny_policy(
                "DocumentOp::PointDelete",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let col = collection.clone();
            let doc_id = document_id.clone();
            Ok(Box::pin(async move {
                document_ops::writes::point_delete_coordinated(
                    engine,
                    permit,
                    col.as_str(),
                    &doc_id,
                )
                .await
            }))
        }

        DocumentOp::BatchInsert {
            collection,
            documents,
            returning,
            rls_filters,
            ..
        } => {
            // BatchInsert has no rls_write_check slot: an unconditional
            // batch of first writes carries no write gate.
            deny_policy(
                "DocumentOp::BatchInsert",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                &RlsWriteCheck::NoPolicyApplies,
            )?;
            let col = collection.clone();
            let docs = documents.clone();
            Ok(Box::pin(async move {
                document_ops::writes::batch_insert_coordinated(engine, permit, col.as_str(), &docs)
                    .await
            }))
        }

        DocumentOp::Upsert {
            collection,
            document_id,
            value,
            on_conflict_updates,
            rls_write_check,
            returning,
            rls_filters,
            ..
        } => {
            deny_policy(
                "DocumentOp::Upsert",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let col = collection.clone();
            let doc_id = document_id.clone();
            let val = value.clone();
            let conflict_updates = on_conflict_updates.clone();
            Ok(Box::pin(async move {
                document_ops::writes::upsert_coordinated(
                    engine,
                    permit,
                    col.as_str(),
                    &doc_id,
                    &val,
                    &conflict_updates,
                )
                .await
            }))
        }

        DocumentOp::Truncate {
            collection,
            restart_identity,
            ..
        } => {
            let col = collection.clone();
            let restart = *restart_identity;
            Ok(Box::pin(async move {
                let result =
                    document_ops::writes::truncate_coordinated(engine, permit, col.as_str())
                        .await?;
                crate::query::truncate::restart_identity(engine, col.as_str(), restart);
                Ok(result)
            }))
        }

        DocumentOp::BulkUpdate {
            collection,
            updates,
            returning,
            rls_filters,
            rls_write_check,
            ..
        } => {
            deny_policy(
                "DocumentOp::BulkUpdate",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let col = collection.clone();
            let updates = updates.clone();
            Ok(Box::pin(async move {
                document_ops::writes::bulk_update_coordinated(
                    engine,
                    permit,
                    col.as_str(),
                    &updates,
                )
                .await
            }))
        }

        DocumentOp::BulkDelete {
            collection,
            returning,
            rls_filters,
            rls_write_check,
            ..
        } => {
            deny_policy(
                "DocumentOp::BulkDelete",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let col = collection.clone();
            Ok(Box::pin(async move {
                document_ops::writes::bulk_delete(engine, col.as_str()).await
            }))
        }

        DocumentOp::InsertSelect {
            target_collection,
            source_collection,
            source_limit,
            ..
        } => {
            let target = target_collection.clone();
            let source = source_collection.clone();
            let limit = *source_limit;
            Ok(Box::pin(async move {
                document_ops::sets::insert_select_coordinated(
                    engine,
                    permit,
                    target.as_str(),
                    source.as_str(),
                    limit,
                )
                .await
            }))
        }

        DocumentOp::UpdateFromJoin {
            target_collection,
            source_collection,
            source_alias,
            target_join_col,
            source_join_col,
            updates,
            returning,
            rls_filters,
            rls_write_check,
            ..
        } => {
            deny_policy(
                "DocumentOp::UpdateFromJoin",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let target = target_collection.clone();
            let source = source_collection.clone();
            let alias = source_alias.clone();
            let target_join = target_join_col.clone();
            let source_join = source_join_col.clone();
            let updates = updates.clone();
            Ok(Box::pin(async move {
                document_ops::sets::update_from_join_coordinated(
                    engine,
                    permit,
                    document_ops::sets::DocumentJoin {
                        target_collection: target.as_str(),
                        source_collection: source.as_str(),
                        source_alias: &alias,
                        target_join_col: &target_join,
                        source_join_col: &source_join,
                    },
                    &updates,
                )
                .await
            }))
        }

        DocumentOp::Merge {
            target_collection,
            source_collection,
            source_alias,
            target_join_col,
            source_join_col,
            clauses,
            returning,
            rls_filters,
            rls_write_check,
            ..
        } => {
            deny_policy(
                "DocumentOp::Merge",
                returning.as_ref(),
                &[rls_filters.as_slice()],
                rls_write_check,
            )?;
            let target = target_collection.clone();
            let source = source_collection.clone();
            let alias = source_alias.clone();
            let target_join = target_join_col.clone();
            let source_join = source_join_col.clone();
            let clauses = clauses.clone();
            Ok(Box::pin(async move {
                document_ops::sets::merge_coordinated(
                    engine,
                    permit,
                    document_ops::sets::DocumentJoin {
                        target_collection: target.as_str(),
                        source_collection: source.as_str(),
                        source_alias: &alias,
                        target_join_col: &target_join,
                        source_join_col: &source_join,
                    },
                    &clauses,
                )
                .await
            }))
        }

        _ => Err(LiteError::BadRequest {
            detail: "document write dispatch received a non-write operation".into(),
        }),
    }
}
