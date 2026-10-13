// SPDX-License-Identifier: Apache-2.0

//! Declaration metadata loading and durable revision writes.

use nodedb_types::Namespace;
use std::collections::BTreeMap;

use super::{DECLARATION_FORMAT, SearchDeclarationRecord, SearchDeclarationRecordV1};
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

/// Decode a stored record of the current or the previous format.
///
/// A version-1 record with a declaration has one field fewer, so it fails the
/// current decode and is read as version 1. A version-1 tombstone has no
/// declaration and decodes as the current shape; only its version moves.
fn decode_record(bytes: &[u8]) -> Result<SearchDeclarationRecord, String> {
    match zerompk::from_msgpack::<SearchDeclarationRecord>(bytes) {
        Ok(mut record) => {
            if record.format_version == 1 && record.declaration.is_none() {
                record.format_version = DECLARATION_FORMAT;
            }
            Ok(record)
        }
        Err(current) => match zerompk::from_msgpack::<SearchDeclarationRecordV1>(bytes) {
            Ok(v1) if v1.format_version == 1 => Ok(v1.into()),
            _ => Err(current.to_string()),
        },
    }
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
        let record = decode_record(&bytes).map_err(|error| LiteError::Serialization {
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

    #[derive(zerompk::ToMessagePack)]
    struct V1Declaration {
        name: String,
        fields: Vec<String>,
        analyzer: String,
        fuzzy: bool,
    }

    #[derive(zerompk::ToMessagePack)]
    struct V1Record {
        format_version: u32,
        revision: u64,
        declaration: Option<V1Declaration>,
    }

    #[test]
    fn version_one_records_read_as_per_field() {
        let declared = zerompk::to_msgpack_vec(&V1Record {
            format_version: 1,
            revision: 3,
            declaration: Some(V1Declaration {
                name: "fts_docs".into(),
                fields: vec!["body".into()],
                analyzer: "standard".into(),
                fuzzy: true,
            }),
        })
        .unwrap();
        let record = decode_record(&declared).unwrap();
        record.check("docs").unwrap();
        assert_eq!(record.format_version, DECLARATION_FORMAT);
        assert_eq!(record.revision, 3);
        let declaration = record.declaration.unwrap();
        assert!(declaration.per_field);
        assert!(declaration.fuzzy);
        assert_eq!(declaration.fields, vec!["body".to_string()]);

        let tombstone = zerompk::to_msgpack_vec(&V1Record {
            format_version: 1,
            revision: 4,
            declaration: None,
        })
        .unwrap();
        let record = decode_record(&tombstone).unwrap();
        record.check("docs").unwrap();
        assert_eq!(
            (record.format_version, record.revision),
            (DECLARATION_FORMAT, 4)
        );
        assert!(record.declaration.is_none());
    }
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
