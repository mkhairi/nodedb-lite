// SPDX-License-Identifier: Apache-2.0

//! Declaration metadata loading and durable revision writes.

use nodedb_types::Namespace;
use std::collections::BTreeMap;

use super::{DECLARATION_FORMAT, SearchDeclarationRecord};
use crate::error::LiteError;
use crate::storage::engine::StorageEngine;

const PREFIX: &[u8] = b"search_declaration:";
const MAX_BYTES: usize = 64 * 1024;

fn key(collection: &str) -> Vec<u8> {
    [PREFIX, collection.as_bytes()].concat()
}

pub(crate) async fn persist_declaration<S: StorageEngine>(
    storage: &S,
    collection: &str,
    record: &SearchDeclarationRecord,
) -> Result<(), LiteError> {
    record.check(collection)?;
    let bytes = zerompk::to_msgpack_vec(record).map_err(|error| LiteError::Serialization {
        detail: error.to_string(),
    })?;
    let key = key(collection);
    if key.len().saturating_add(bytes.len()) > MAX_BYTES {
        return Err(LiteError::Backpressure {
            detail: format!(
                "search declaration for '{collection}' exceeds {MAX_BYTES} bytes: reduce its fields"
            ),
        });
    }
    storage.put(Namespace::Meta, &key, &bytes).await
}

pub(crate) async fn persist_declaration_tombstone<S: StorageEngine>(
    storage: &S,
    collection: &str,
    revision: u64,
) -> Result<(), LiteError> {
    persist_declaration(
        storage,
        collection,
        &SearchDeclarationRecord {
            format_version: DECLARATION_FORMAT,
            revision,
            declaration: None,
        },
    )
    .await
}

pub(crate) async fn load_declarations<S: StorageEngine>(
    storage: &S,
) -> Result<BTreeMap<String, SearchDeclarationRecord>, LiteError> {
    // Catalog metadata remains allocated. Authoritative row scans use bounded continuation.
    let entries = storage.scan_prefix(Namespace::Meta, PREFIX).await?;
    let mut records = BTreeMap::new();
    for (key, bytes) in entries {
        if key.len().saturating_add(bytes.len()) > MAX_BYTES {
            return Err(LiteError::Backpressure {
                detail: format!(
                    "stored search declaration exceeds {MAX_BYTES} bytes: reduce its fields"
                ),
            });
        }
        let collection = std::str::from_utf8(key.strip_prefix(PREFIX).ok_or_else(|| {
            LiteError::Serialization {
                detail: "search declaration key excludes its prefix".into(),
            }
        })?)
        .map_err(|error| LiteError::Serialization {
            detail: format!("search declaration key is not UTF-8: {error}"),
        })?;
        let record: SearchDeclarationRecord =
            zerompk::from_msgpack(&bytes).map_err(|error| LiteError::Serialization {
                detail: format!("search declaration for '{collection}': {error}"),
            })?;
        record.check(collection)?;
        records.insert(collection.to_owned(), record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    #[tokio::test]
    async fn declaration_tombstone_survives_empty_collection_loading() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        persist_declaration_tombstone(&storage, "empty", 3)
            .await
            .unwrap();
        let records = load_declarations(&storage).await.unwrap();
        assert_eq!(records["empty"].revision, 3);
        assert_eq!(records["empty"].declaration, None);
    }
}
