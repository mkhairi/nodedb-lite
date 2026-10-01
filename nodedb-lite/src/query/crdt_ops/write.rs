// SPDX-License-Identifier: Apache-2.0
//! CRDT write, policy-set, and delta-apply handlers.

use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

/// Apply a remote CRDT delta from another peer into `collection`'s document.
///
/// Imports the raw Loro delta bytes, re-indexes the text of the rows it
/// changed, then acknowledges the mutation on success or rejects it on
/// import failure.
pub async fn handle_apply<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    delta: &[u8],
    mutation_id: u64,
) -> Result<QueryResult, LiteError> {
    handle_apply_coordinated(engine, None, collection, delta, mutation_id).await
}

pub(crate) async fn handle_apply_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    delta: &[u8],
    mutation_id: u64,
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return handle_apply_admitted(engine, permit, collection, delta, mutation_id).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result =
        handle_apply_admitted(engine, guard.permit(), collection, delta, mutation_id).await;
    guard.finish(result)
}

pub(crate) async fn handle_apply_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    delta: &[u8],
    mutation_id: u64,
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    let result = {
        let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        crdt.import_remote(collection, delta)
    };

    match result {
        // The admission is already logged when it contributed nothing; the
        // delta is still acknowledged, since a fully-trimmed import is a
        // successful (idempotent) apply, not a failure to replicate.
        Ok(imported) => {
            // Re-index the rows the delta changed. It came from a peer, so
            // nothing is staged back.
            crate::engine::fts::maintain::reindex_crdt_documents(
                &engine.fts_state,
                &engine.crdt,
                None::<&crate::engine::fts::maintain::FtsOutbound<S>>,
                collection,
                imported.changed_rows.iter().map(String::as_str),
            )?;
            let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
            crdt.acknowledge(mutation_id);
            Ok(QueryResult {
                columns: vec![],
                rows: vec![],
                rows_affected: 1,
                command: None,
            })
        }
        Err(import_err) => {
            let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
            crdt.reject_delta(mutation_id);
            Err(import_err)
        }
    }
}

/// Import a per-collection Loro snapshot (durable RESTORE re-issue path).
///
/// The snapshot is merged into the target collection's own document. Loro's
/// import is monotonic, idempotent and commutative, so a re-issued snapshot
/// converges with whatever that document already holds.
pub async fn handle_import_snapshot<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    handle_import_snapshot_coordinated(engine, None, collection, bytes).await
}

pub(crate) async fn handle_import_snapshot_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
    bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return handle_import_snapshot_admitted(engine, permit, collection, bytes).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = handle_import_snapshot_admitted(engine, guard.permit(), collection, bytes).await;
    guard.finish(result)
}

pub(crate) async fn handle_import_snapshot_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    let imported = engine
        .crdt
        .lock()
        .map_err(|_| LiteError::LockPoisoned)?
        .import_local_tracked(collection, bytes)?;
    // Re-index the rows the snapshot changed. It is this device's own state,
    // so nothing is staged for Origin.
    crate::engine::fts::maintain::reindex_crdt_documents(
        &engine.fts_state,
        &engine.crdt,
        None::<&crate::engine::fts::maintain::FtsOutbound<S>>,
        collection,
        imported.changed_rows.iter().map(String::as_str),
    )?;
    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: None,
    })
}

/// Set the conflict resolution policy for a CRDT collection.
pub async fn handle_set_policy<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    policy_json: &str,
) -> Result<QueryResult, LiteError> {
    let policy: nodedb_crdt::CollectionPolicy =
        sonic_rs::from_str(policy_json).map_err(|e| LiteError::BadRequest {
            detail: format!("invalid CollectionPolicy JSON: {e}"),
        })?;

    let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
    crdt.set_policy(collection, policy);

    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: None,
    })
}
