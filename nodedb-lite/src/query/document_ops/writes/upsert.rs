// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::reads::msgpack_bytes_to_crdt_fields;
use super::super::write_helpers::affected;
use super::super::write_helpers::{
    decode_literal_updates, decode_strict_fields, fields_to_values, strict_schema,
};
use super::UpdateValue;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::text_index::{reindex_documents, reindex_strict_rows};
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;
use std::collections::HashMap;

/// Upsert: insert or update. When `on_conflict_updates` is non-empty, applies
/// those assignments on conflict instead of merging the new value.
pub async fn upsert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    on_conflict_updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    upsert_coordinated(
        engine,
        None,
        collection,
        document_id,
        value_bytes,
        on_conflict_updates,
    )
    .await
}

pub(crate) async fn upsert_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    on_conflict_updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return upsert_admitted(
            engine,
            permit,
            collection,
            document_id,
            value_bytes,
            on_conflict_updates,
        )
        .await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = upsert_admitted(
        engine,
        guard.permit(),
        collection,
        document_id,
        value_bytes,
        on_conflict_updates,
    )
    .await;
    guard.finish(result)
}

pub(crate) async fn upsert_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    document_id: &str,
    value_bytes: &[u8],
    on_conflict_updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    if is_strict(engine, collection) {
        let pk = Value::String(document_id.to_string());
        let existed = engine.strict.get(collection, &pk).await?.is_some();
        if existed && !on_conflict_updates.is_empty() {
            let field_updates = decode_literal_updates(on_conflict_updates)?;
            engine
                .strict
                .update(collection, &pk, &field_updates)
                .await?;
        } else {
            let fields = decode_strict_fields(value_bytes)?;
            if existed {
                let updates: HashMap<String, Value> = fields.into_iter().collect();
                engine.strict.update(collection, &pk, &updates).await?;
            } else {
                let schema = strict_schema(engine, collection)?;
                let values = fields_to_values(&fields, &schema.columns);
                engine.strict.insert(collection, &values).await?;
            }
        }
        reindex_strict_rows(engine, collection, &[pk]).await?;
    } else {
        let crdt_fields = msgpack_bytes_to_crdt_fields(value_bytes)?;
        let loro_fields: Vec<(&str, loro::LoroValue)> = crdt_fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        engine
            .crdt
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .upsert(collection, document_id, &loro_fields)?;
        reindex_documents(engine, collection, [document_id])?;
    }
    Ok(affected(1, "UPSERT"))
}
