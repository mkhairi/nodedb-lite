// SPDX-License-Identifier: Apache-2.0

//! Building an index's entries from its collection's rows: at `CREATE INDEX`,
//! at `REINDEX`, and at open.
//!
//! At open, stored entries are rebuilt in two cases:
//!
//! - The stored format is not the one this build writes (including a store
//!   last written before the format was recorded). Entries in another format
//!   are unreadable, so every index is rebuilt from its definition. Stores
//!   written before index definitions were persisted hold entries under
//!   `{collection}:{field}:{value}` and no definition: those indexes cannot
//!   be recovered and must be created again. `index::legacy` removes their
//!   old entries.
//! - The index is on a bitemporal collection. Its history is written to disk
//!   on every write, ahead of the CRDT state the entries are flushed with,
//!   so after an unclean exit the history can hold rows the stored entries
//!   lack. Rebuilding from the history closes that gap.
//!
//! Indexes on other document collections need no repair: their entries are
//! flushed in the same batch as the rows they index. Strict and key-value
//! indexes need none either: their entries are written in the same storage
//! batch as each row write.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use nodedb_types::value::Value;

use crate::engine::crdt::CrdtEngine;
use crate::engine::document::history::ops::{is_bitemporal, scan_live_documents};
use crate::engine::strict::StrictEngine;
use crate::error::LiteError;
use crate::query::kv_ops::indexes::kv_index_rows;
use crate::storage::engine::StorageEngine;

use super::catalog::{IndexDef, IndexEngine};
use super::document::row_value;
use super::store::IndexCatalog;

/// Every current row of a collection held in the CRDT store.
pub(crate) fn crdt_rows(crdt: &CrdtEngine, collection: &str) -> Vec<(String, Value)> {
    crdt.list_ids(collection)
        .into_iter()
        .filter_map(|id| {
            let row = crdt.read(collection, &id)?;
            Some((id, row_value(&row)))
        })
        .collect()
}

/// Every live row of a bitemporal collection, read from its history.
async fn history_rows<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<Vec<(String, Value)>, LiteError> {
    let mut rows = Vec::new();
    for (id, body) in scan_live_documents(storage, collection).await? {
        let row = if body.is_empty() {
            Value::Object(Default::default())
        } else {
            nodedb_types::json_msgpack::value_from_msgpack(&body).map_err(|e| {
                LiteError::Serialization {
                    detail: format!("live version of '{collection}'/'{id}' does not decode: {e}"),
                }
            })?
        };
        rows.push((id, row));
    }
    Ok(rows)
}

/// Install `def` with entries built from the current rows of its collection,
/// replacing whatever entries an index of the same name held. Returns the
/// number of entries written.
///
/// A strict or key-value index is written to storage before this returns:
/// its rows were. A document index follows the CRDT rows it is built from
/// and is persisted by the flush.
pub(crate) async fn build_index<S: StorageEngine>(
    catalog: &IndexCatalog,
    storage: &S,
    crdt: &Mutex<CrdtEngine>,
    strict: &StrictEngine<S>,
    def: Arc<IndexDef>,
) -> Result<u64, LiteError> {
    match def.engine {
        IndexEngine::Document => build_document_index(catalog, storage, crdt, def).await,
        IndexEngine::Strict | IndexEngine::KeyValue => {
            // No row write may land between the row read and the install.
            let _ddl = catalog.durable_ddl.write().await;
            let rows = match def.engine {
                IndexEngine::Strict => strict.rows_with_ids(&def.collection).await?,
                _ => kv_index_rows(storage, &def.collection).await?,
            };
            let name = def.name.clone();
            let fresh = catalog.def_named(&name).is_none();
            let (built, ops) = catalog.install(def, rows)?;
            if let Err(e) = storage.batch_write(&ops).await {
                // Nothing was stored: a new index leaves no trace in memory.
                if fresh {
                    catalog.forget(&name);
                }
                return Err(e);
            }
            Ok(built)
        }
    }
}

async fn build_document_index<S: StorageEngine>(
    catalog: &IndexCatalog,
    storage: &S,
    crdt: &Mutex<CrdtEngine>,
    def: Arc<IndexDef>,
) -> Result<u64, LiteError> {
    if is_bitemporal(storage, &def.collection).await? {
        // No bitemporal write may land between the history read and the
        // install; writes after the install maintain the new index.
        let _building = catalog.bitemporal_build.write().await;
        let rows = history_rows(storage, &def.collection).await?;
        let crdt = crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        // CRDT copies of rows the history holds no live version of were
        // deleted: they stay out of the index whatever later reaches them.
        let live: HashSet<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
        let deleted: Vec<String> = crdt
            .list_ids(&def.collection)
            .into_iter()
            .filter(|id| !live.contains(id.as_str()))
            .collect();
        let collection = def.collection.clone();
        let (built, _flushed_later) = catalog.install(def, rows)?;
        catalog.tombstone(&collection, deleted.iter().map(String::as_str));
        return Ok(built);
    }
    // One CRDT lock hold covers the row read and the install, so no write
    // falls between them.
    let crdt = crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
    let rows = crdt_rows(&crdt, &def.collection);
    let (built, _flushed_later) = catalog.install(def, rows)?;
    Ok(built)
}

/// Rebuild the indexes whose stored entries may not match their rows: all of
/// them when `rebuild_all`, otherwise those on bitemporal collections.
pub(crate) async fn rebuild_at_open<S: StorageEngine>(
    catalog: &IndexCatalog,
    storage: &S,
    crdt: &Mutex<CrdtEngine>,
    strict: &StrictEngine<S>,
    rebuild_all: bool,
) -> Result<(), LiteError> {
    for def in catalog.defs() {
        let stale = rebuild_all
            || (def.engine == IndexEngine::Document
                && is_bitemporal(storage, &def.collection).await?);
        if stale {
            build_index(catalog, storage, crdt, strict, def).await?;
        }
    }
    Ok(())
}
