// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::reads::msgpack_bytes_to_crdt_fields;
use super::super::write_helpers::affected;
use super::super::write_helpers::{decode_strict_fields, fields_to_values, strict_schema};
use crate::engine::crdt::{CrdtRowOp, CrdtRowWrite};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::text_index::{index_strict_rows, reindex_documents};
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

/// BatchInsert: insert N documents in a single transaction.
pub async fn batch_insert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    documents: &[(String, Vec<u8>)],
) -> Result<QueryResult, LiteError> {
    batch_insert_coordinated(engine, None, collection, documents).await
}

pub(crate) async fn batch_insert_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    documents: &[(String, Vec<u8>)],
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return batch_insert_admitted(engine, permit, collection, documents).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = batch_insert_admitted(engine, guard.permit(), collection, documents).await;
    guard.finish(result)
}

pub(crate) async fn batch_insert_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    documents: &[(String, Vec<u8>)],
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    if is_strict(engine, collection) {
        let schema = strict_schema(engine, collection)?;
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(documents.len());
        for (_doc_id, value_bytes) in documents {
            let fields = decode_strict_fields(value_bytes)?;
            let values = fields_to_values(&fields, &schema.columns);
            rows.push(values);
        }
        let affected_n = rows.len() as u64;
        engine.strict.insert_batch(collection, &rows).await?;
        index_strict_rows(engine, collection, rows.iter().map(Vec::as_slice))?;
        Ok(affected(affected_n, "INSERT"))
    } else {
        let mut decoded: Vec<(&str, Vec<(String, loro::LoroValue)>)> =
            Vec::with_capacity(documents.len());
        for (doc_id, value_bytes) in documents {
            let crdt_fields = msgpack_bytes_to_crdt_fields(value_bytes)?;
            decoded.push((doc_id.as_str(), crdt_fields));
        }
        let affected_n = decoded.len() as u64;
        let slices: Vec<Vec<(&str, loro::LoroValue)>> = decoded
            .iter()
            .map(|(_, fields)| {
                fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect()
            })
            .collect();
        let rows: Vec<CrdtRowOp<'_>> = decoded
            .iter()
            .zip(&slices)
            .map(|((doc_id, _), f)| (CrdtRowWrite::Upsert, collection, *doc_id, f.as_slice()))
            .collect();
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        // The whole batch is checked before any document is written.
        crdt.check_unique_writes(&rows)?;
        for &(_, _, doc_id, fields) in &rows {
            crdt.upsert_deferred(collection, doc_id, fields)?;
        }
        crdt.flush_deltas().map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
        drop(crdt);
        reindex_documents(
            engine,
            collection,
            decoded.iter().map(|(doc_id, _)| *doc_id),
        )?;
        Ok(affected(affected_n, "INSERT"))
    }
}
