// SPDX-License-Identifier: Apache-2.0
//! Point writes for the KV engine: put and the insert variants.

use nodedb_physical::physical_plan::document::UpdateValue;
use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::kv_ops::body::{encode_kv_body, kv_body_columns};
use crate::query::on_conflict::apply_patch;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::super::reads::{decode_value, encode_value, is_expired, kv_key};

/// Store one encoded value, with the index entries it implies.
pub(super) async fn put_encoded<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    rkey: Vec<u8>,
    encoded: Vec<u8>,
) -> Result<(), LiteError> {
    engine
        .kv_local
        .commit(
            &*engine.storage,
            vec![WriteOp::Put {
                ns: Namespace::Kv,
                key: rkey,
                value: encoded,
            }],
        )
        .await
}

/// `kv_put`, tagged `INSERT` instead of `UPSERT`. Backs every insert variant
/// that falls through to an unconditional put once absence is confirmed.
async fn insert_via_put<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let mut result = kv_put(engine, collection, key, value, ttl_ms).await?;
    result.command = Some("INSERT".into());
    Ok(result)
}

/// Put: unconditional upsert.
pub async fn kv_put<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let deadline = if ttl_ms > 0 {
        crate::runtime::now_millis().saturating_add(ttl_ms)
    } else {
        0
    };
    put_encoded(
        engine,
        kv_key(collection, key),
        encode_value(deadline, value),
    )
    .await?;
    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: Some("UPSERT".into()),
    })
}

/// Insert: write only if key absent; error on duplicate.
pub async fn kv_insert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let existing = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
    if let Some(raw) = existing
        && let Some((deadline, _)) = decode_value(&raw)
        && !is_expired(deadline)
    {
        return Err(LiteError::BadRequest {
            detail: format!("unique_violation: key already exists in collection '{collection}'"),
        });
    }
    insert_via_put(engine, collection, key, value, ttl_ms).await
}

/// InsertIfAbsent: write if absent, silently no-op on duplicate.
pub async fn kv_insert_if_absent<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let existing = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
    if let Some(raw) = existing
        && let Some((deadline, _)) = decode_value(&raw)
        && !is_expired(deadline)
    {
        return Ok(QueryResult {
            columns: vec![],
            rows: vec![],
            rows_affected: 0,
            command: Some("INSERT".into()),
        });
    }
    insert_via_put(engine, collection, key, value, ttl_ms).await
}

/// InsertOnConflictUpdate: write if absent; on conflict apply field updates.
///
/// `updates` carries `UpdateValue`: a `Literal` decodes and overwrites the
/// field directly; an `Expr` (`n + 1`, `EXCLUDED.n`, ...) evaluates via
/// `query::on_conflict::apply_patch` against the existing stored row, with
/// `EXCLUDED.col` bound to `value` (the row that would have been inserted).
/// The merge follows Origin's: a raw single-`value` row reads as
/// `{"value": <text>}` and stays raw, so `SET value = EXCLUDED.value`
/// overwrites it, and assigning any other column to it is a `BadRequest`.
pub async fn kv_insert_on_conflict_update<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    value: &[u8],
    ttl_ms: u64,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let existing = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    let raw = match existing {
        None => return insert_via_put(engine, collection, key, value, ttl_ms).await,
        Some(raw) => match decode_value(&raw) {
            None => return insert_via_put(engine, collection, key, value, ttl_ms).await,
            Some((deadline, _)) if is_expired(deadline) => {
                return insert_via_put(engine, collection, key, value, ttl_ms).await;
            }
            Some(_) => raw,
        },
    };

    let (old_deadline, old_user_bytes) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
        detail: "corrupt KV entry".into(),
    })?;

    // Origin's merge: both sides decode to rows (a raw body reads as
    // `{"value": <text>}`), and the merged row re-encodes in the stored
    // body's shape, so a raw row stays raw.
    let (mut map, shape) = kv_body_columns(old_user_bytes)?;
    let (excluded, _) = kv_body_columns(value)?;

    apply_patch(&mut map, updates, &excluded)?;

    let new_user_bytes = encode_kv_body(map, shape)?;

    let keep_deadline = if ttl_ms > 0 {
        crate::runtime::now_millis().saturating_add(ttl_ms)
    } else {
        old_deadline
    };

    put_encoded(engine, rkey, encode_value(keep_deadline, &new_user_bytes)).await?;

    Ok(QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected: 1,
        command: Some("UPDATE".into()),
    })
}
