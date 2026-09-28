// SPDX-License-Identifier: Apache-2.0
//! Secondary indexes on typed KV rows.
//!
//! The indexes live in the shared catalog (`crate::index`) as key-value
//! indexes. Every KV write commits through [`kv_commit`], which adds the
//! entries the written rows imply to the same storage batch. An entry names
//! a row by the hex of its key, which reads back whatever bytes the key is.
//! A raw (untyped) body has no columns to index; an expired row holds no
//! entries a read or a unique check counts.

use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::index::durable::RowImage;
use crate::index::key::{doc_id_of_key, key_of_doc_id};
use crate::index::{IndexCatalog, IndexEngine};
use crate::query::document_ops::indexes::{CreateIndexRequest, declare_index, drop_field_index};
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::body::decode_kv_map;
use super::reads::{decode_value, is_expired, kv_key, split_kv_key};

/// The row a stored KV value holds for its indexes: its typed columns, none
/// for a raw body, and no row at all once expired.
fn stored_row(stored: &[u8]) -> Result<Option<Value>, LiteError> {
    let Some((deadline, body)) = decode_value(stored) else {
        return Ok(None);
    };
    if is_expired(deadline) {
        return Ok(None);
    }
    Ok(Some(Value::Object(
        decode_kv_map(body)?.unwrap_or_default(),
    )))
}

/// Every live row of `collection` with the document id its entries carry.
pub(crate) async fn kv_index_rows<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<Vec<(String, Value)>, LiteError> {
    let mut prefix = collection.as_bytes().to_vec();
    prefix.push(0);
    let mut rows = Vec::new();
    for (composite, stored) in storage.scan_prefix(Namespace::Kv, &prefix).await? {
        let Some((coll, key)) = split_kv_key(&composite) else {
            continue;
        };
        if coll != collection {
            continue;
        }
        if let Some(row) = stored_row(&stored)? {
            rows.push((doc_id_of_key(key), row));
        }
    }
    Ok(rows)
}

/// The stored row an index entry of `collection` names.
pub(crate) async fn kv_row_by_doc_id<S: StorageEngine>(
    storage: &S,
    collection: &str,
    doc_id: &str,
) -> Result<Option<Value>, LiteError> {
    let Some(key) = key_of_doc_id(doc_id) else {
        return Ok(None);
    };
    match storage
        .get(Namespace::Kv, &kv_key(collection, &key))
        .await?
    {
        Some(stored) => stored_row(&stored),
        None => Ok(None),
    }
}

/// The rows KV writes `ops` leave behind, as index row images.
fn kv_images(ops: &[WriteOp]) -> Result<Vec<RowImage>, LiteError> {
    let mut images = Vec::new();
    for op in ops {
        let (key, row) = match op {
            WriteOp::Put { ns, key, value } if *ns == Namespace::Kv => (key, stored_row(value)?),
            WriteOp::Delete { ns, key } if *ns == Namespace::Kv => (key, None),
            _ => continue,
        };
        if let Some((collection, user_key)) = split_kv_key(key) {
            images.push(RowImage {
                collection: collection.to_string(),
                doc_id: doc_id_of_key(user_key),
                row,
            });
        }
    }
    Ok(images)
}

/// Refuse KV writes `ops` a unique index forbids, writing nothing.
pub(crate) async fn kv_check<S: StorageEngine>(
    storage: &S,
    indexes: &IndexCatalog,
    ops: &[WriteOp],
) -> Result<(), LiteError> {
    indexes
        .check_rows(
            IndexEngine::KeyValue,
            &kv_images(ops)?,
            |collection, id| async move { kv_row_by_doc_id(storage, &collection, &id).await },
        )
        .await
}

/// Commit KV writes `ops` — puts of encoded values and deletes in
/// `Namespace::Kv` — with the index entries they imply, in one storage
/// batch. A write a unique index refuses commits nothing.
pub(crate) async fn kv_commit<S: StorageEngine>(
    storage: &S,
    indexes: &IndexCatalog,
    ops: Vec<WriteOp>,
) -> Result<(), LiteError> {
    let images = kv_images(&ops)?;
    indexes
        .commit_rows(
            storage,
            IndexEngine::KeyValue,
            ops,
            images,
            |collection, id| async move { kv_row_by_doc_id(storage, &collection, &id).await },
        )
        .await
}

/// RegisterIndex: declare a key-value index on `field` of `collection` and
/// build it from the rows the collection holds. The index is built whether
/// or not `backfill` asks: an index missing existing rows would answer
/// lookups without them.
pub async fn kv_register_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    field: &str,
    _backfill: bool,
) -> Result<QueryResult, LiteError> {
    declare_index(
        engine,
        CreateIndexRequest {
            name: None,
            collection,
            field,
            unique: false,
            case_insensitive: false,
            predicate: None,
            if_not_exists: true,
        },
        IndexEngine::KeyValue,
    )
    .await
}

/// DropIndex: remove the key-value index on `field` of `collection`.
pub async fn kv_drop_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    field: &str,
) -> Result<QueryResult, LiteError> {
    drop_field_index(engine, collection, field).await
}
