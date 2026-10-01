// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::reads::ndb_value_to_loro;
use super::super::write_helpers::affected;
use super::super::write_helpers::{decode_literal_updates, strict_schema};
use super::UpdateValue;
use crate::engine::crdt::{CrdtRowOp, CrdtRowWrite};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::text_index::{reindex_documents, reindex_strict_rows};
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

/// BulkUpdate: scan matching documents and apply field updates to all.
///
/// Lite does not yet evaluate residual scan filters — every document in the
/// collection receives the update. Callers that need filtered bulk updates
/// should compose `Scan` + per-row `PointUpdate` at the application layer.
pub async fn bulk_update<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    bulk_update_coordinated(engine, None, collection, updates).await
}

pub(crate) async fn bulk_update_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return bulk_update_admitted(engine, permit, collection, updates).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = bulk_update_admitted(engine, guard.permit(), collection, updates).await;
    guard.finish(result)
}

pub(crate) async fn bulk_update_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    let field_updates = decode_literal_updates(updates)?;

    if is_strict(engine, collection) {
        let schema = strict_schema(engine, collection)?;
        let pk_idx = schema
            .columns
            .iter()
            .position(|c| c.primary_key)
            .ok_or_else(|| LiteError::BadRequest {
                detail: format!("strict collection '{collection}' has no primary key"),
            })?;
        let all_rows = engine.strict.list_rows(collection).await?;
        let mut affected_n: u64 = 0;
        let mut pks: Vec<Value> = Vec::with_capacity(all_rows.len());
        for row in &all_rows {
            let Some(pk) = row.get(pk_idx) else {
                continue;
            };
            if engine.strict.update(collection, pk, &field_updates).await? {
                affected_n += 1;
            }
            pks.push(pk.clone());
        }
        reindex_strict_rows(engine, collection, &pks).await?;
        Ok(affected(affected_n, "UPDATE"))
    } else {
        let loro_updates: Vec<(String, loro::LoroValue)> = field_updates
            .into_iter()
            .map(|(k, v)| (k, ndb_value_to_loro(v)))
            .collect();
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        let ids = crdt.list_ids(collection);
        let loro_slice: Vec<(&str, loro::LoroValue)> = loro_updates
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        let rows: Vec<CrdtRowOp<'_>> = ids
            .iter()
            .map(|id| {
                (
                    CrdtRowWrite::SetFields,
                    collection,
                    id.as_str(),
                    loro_slice.as_slice(),
                )
            })
            .collect();
        crdt.check_unique_writes(&rows)?;
        let mut affected_n: u64 = 0;
        for id in &ids {
            // A merge: fields the update does not assign keep their values.
            crdt.set_fields_deferred(collection, id, &loro_slice)?;
            affected_n += 1;
        }
        crdt.flush_deltas().map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
        drop(crdt);
        reindex_documents(engine, collection, ids.iter().map(String::as_str))?;
        Ok(affected(affected_n, "UPDATE"))
    }
}
