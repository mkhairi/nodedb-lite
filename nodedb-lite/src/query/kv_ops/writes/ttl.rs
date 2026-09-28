// SPDX-License-Identifier: Apache-2.0
//! TTL writes for the KV engine: expire and persist.

use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::reads::{decode_value, encode_value, kv_key};
use super::basic::put_encoded;

/// Expire: set or update TTL on an existing key.
pub async fn kv_expire<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    ttl_ms: u64,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let stored = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    match stored {
        None => Ok(QueryResult {
            columns: vec![],
            rows: vec![],
            rows_affected: 0,
            command: None,
        }),
        Some(raw) => {
            let (_, user_bytes) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
                detail: "corrupt KV entry".into(),
            })?;
            let deadline = crate::runtime::now_millis().saturating_add(ttl_ms);
            put_encoded(engine, rkey, encode_value(deadline, user_bytes)).await?;
            Ok(QueryResult {
                columns: vec![],
                rows: vec![],
                rows_affected: 1,
                command: None,
            })
        }
    }
}

/// Persist: remove TTL from an existing key (make it permanent).
pub async fn kv_persist<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let stored = engine
        .storage
        .get(Namespace::Kv, &rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;

    match stored {
        None => Ok(QueryResult {
            columns: vec![],
            rows: vec![],
            rows_affected: 0,
            command: None,
        }),
        Some(raw) => {
            let (_, user_bytes) = decode_value(&raw).ok_or_else(|| LiteError::Storage {
                detail: "corrupt KV entry".into(),
            })?;
            put_encoded(engine, rkey, encode_value(0, user_bytes)).await?;
            Ok(QueryResult {
                columns: vec![],
                rows: vec![],
                rows_affected: 1,
                command: None,
            })
        }
    }
}
