// SPDX-License-Identifier: Apache-2.0

//! Rebuild the `LatestVersion` index from existing `DocumentHistory` rows for
//! databases written before the index was introduced.

use nodedb_types::Namespace;

use crate::error::LiteError;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::super::key::{coll_prefix, format_sys_from, latest_version_key, parse_sys_from};
use super::super::value::{VersionTag, decode_value};

/// Populate the `LatestVersion` index for `collection` from existing history rows.
///
/// Call this once per collection at open time.  If the index already has
/// entries for the collection (i.e. this database was written with the current
/// code), the function scans history, computes the correct pointers, and
/// overwrites any that are missing or stale — safe to call repeatedly.
///
/// Pages retain one document across boundaries. Pointer writes use bounded batches.
pub async fn backfill_latest_version<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<(), LiteError> {
    backfill_pages(storage, collection, 128, 8 * 1024 * 1024).await
}

async fn backfill_pages<S: StorageEngine>(
    storage: &S,
    collection: &str,
    max_records: usize,
    max_bytes: usize,
) -> Result<(), LiteError> {
    let prefix = coll_prefix(collection);
    let mut cursor: Option<Vec<u8>> = None;
    let mut latest: Option<(String, i64, VersionTag)> = None;
    let mut operations = Vec::new();
    let mut operation_bytes = 0usize;
    loop {
        let page = storage
            .scan_prefix_from_budgeted(
                Namespace::DocumentHistory,
                &prefix,
                cursor.as_deref(),
                max_records,
                max_bytes,
            )
            .await?;
        if page.entries.is_empty() {
            break;
        }
        for (key, value) in page.entries {
            cursor = Some(key.clone());
            let suffix = &key[prefix.len()..];
            let Some(separator) = suffix.iter().position(|&byte| byte == 0) else {
                continue;
            };
            let Ok(id) = std::str::from_utf8(&suffix[..separator]) else {
                continue;
            };
            let Ok(decoded) = decode_value(&value) else {
                continue;
            };
            let Some(timestamp) = parse_sys_from(&key) else {
                continue;
            };
            if latest
                .as_ref()
                .is_some_and(|(previous, _, _)| previous != id)
                && let Some(previous) = latest.take()
            {
                queue_pointer(
                    storage,
                    collection,
                    previous,
                    &mut operations,
                    &mut operation_bytes,
                    max_records,
                    max_bytes,
                )
                .await?;
            }
            match &mut latest {
                Some((_, previous, tag)) if timestamp >= *previous => {
                    *previous = timestamp;
                    *tag = decoded.tag;
                }
                None => latest = Some((id.to_owned(), timestamp, decoded.tag)),
                _ => {}
            }
        }
    }
    if let Some(previous) = latest {
        queue_pointer(
            storage,
            collection,
            previous,
            &mut operations,
            &mut operation_bytes,
            max_records,
            max_bytes,
        )
        .await?;
    }
    if !operations.is_empty() {
        storage.batch_write(&operations).await?;
    }
    Ok(())
}

async fn queue_pointer<S: StorageEngine>(
    storage: &S,
    collection: &str,
    (id, timestamp, tag): (String, i64, VersionTag),
    operations: &mut Vec<WriteOp>,
    bytes: &mut usize,
    max_records: usize,
    max_bytes: usize,
) -> Result<(), LiteError> {
    let key = latest_version_key(collection, &id);
    let value = if tag == VersionTag::Live {
        format_sys_from(timestamp).into_bytes()
    } else {
        Vec::new()
    };
    let cost = key.len().saturating_add(value.len());
    if cost > max_bytes {
        return Err(LiteError::Backpressure {
            detail: format!(
                "history pointer '{collection}:{id}' exceeds {max_bytes} bytes. Shorten the document ID or increase the source page budget"
            ),
        });
    }
    if !operations.is_empty()
        && (operations.len() >= max_records || cost > max_bytes.saturating_sub(*bytes))
    {
        storage.batch_write(operations).await?;
        operations.clear();
        *bytes = 0;
    }
    *bytes += cost;
    operations.push(if tag == VersionTag::Live {
        WriteOp::Put {
            ns: Namespace::LatestVersion,
            key,
            value,
        }
    } else {
        WriteOp::Delete {
            ns: Namespace::LatestVersion,
            key,
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::storage::engine::WriteOp;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    use super::super::super::key::versioned_doc_key;
    use super::super::super::value::encode_value;
    use super::super::read::versioned_get_current;
    use super::super::write::versioned_put;
    use super::*;

    #[tokio::test]
    async fn history_page_boundaries_preserve_latest_versions_and_tombstones() {
        let storage = mem_storage().await;
        versioned_put(&storage, "c", "a", b"first", 10, None, None)
            .await
            .unwrap();
        versioned_put(&storage, "c", "a", b"second", 20, None, None)
            .await
            .unwrap();
        super::super::write::versioned_tombstone(&storage, "c", "a", 30, None)
            .await
            .unwrap();
        versioned_put(&storage, "c", "ab", b"neighbor", 10, None, None)
            .await
            .unwrap();
        storage
            .delete(Namespace::LatestVersion, &latest_version_key("c", "ab"))
            .await
            .unwrap();
        backfill_pages(&storage, "c", 1, 4096).await.unwrap();
        assert!(
            versioned_get_current(&storage, "c", "a")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            versioned_get_current(&storage, "c", "ab")
                .await
                .unwrap()
                .unwrap()
                .body,
            b"neighbor"
        );
    }

    async fn mem_storage() -> PagedbStorageMem {
        PagedbStorageMem::open_in_memory()
            .await
            .expect("open in-memory storage")
    }

    /// Simulate a database written before the LatestVersion index existed:
    /// write DocumentHistory rows directly (bypassing versioned_put to skip the
    /// pointer write), then call backfill and verify get_current works.
    #[tokio::test]
    async fn backfill_builds_index_from_history() {
        let s = mem_storage().await;

        // Write a history row without the LatestVersion pointer (pre-index state).
        let history_key = versioned_doc_key("c", "d1", 100).unwrap();
        let history_value = encode_value(VersionTag::Live, 100, i64::MAX, b"legacy_body");
        s.batch_write(&[WriteOp::Put {
            ns: Namespace::DocumentHistory,
            key: history_key,
            value: history_value,
        }])
        .await
        .unwrap();

        // No pointer yet.
        let ptr_key = latest_version_key("c", "d1");
        assert!(
            s.get(Namespace::LatestVersion, &ptr_key)
                .await
                .unwrap()
                .is_none(),
            "pointer must be absent before backfill"
        );

        // Run backfill.
        backfill_latest_version(&s, "c").await.unwrap();

        // Pointer now present.
        let ptr = s
            .get(Namespace::LatestVersion, &ptr_key)
            .await
            .unwrap()
            .expect("pointer must exist after backfill");
        assert_eq!(ptr, format_sys_from(100).into_bytes());

        // get_current works via the new pointer.
        let v = versioned_get_current(&s, "c", "d1").await.unwrap().unwrap();
        assert_eq!(v.body, b"legacy_body");
    }

    /// Backfill on a tombstoned doc removes any stale pointer.
    #[tokio::test]
    async fn backfill_removes_stale_pointer_for_tombstoned_doc() {
        let s = mem_storage().await;

        // Write a LIVE history row at t=100 and a TOMBSTONE at t=200 directly,
        // but manually insert a stale LatestVersion pointer pointing at t=100.
        let live_key = versioned_doc_key("c", "d1", 100).unwrap();
        let live_value = encode_value(VersionTag::Live, 100, i64::MAX, b"body");
        let tomb_key = versioned_doc_key("c", "d1", 200).unwrap();
        let tomb_value = encode_value(VersionTag::Tombstone, 200, i64::MAX, &[]);
        let ptr_key = latest_version_key("c", "d1");

        s.batch_write(&[
            WriteOp::Put {
                ns: Namespace::DocumentHistory,
                key: live_key,
                value: live_value,
            },
            WriteOp::Put {
                ns: Namespace::DocumentHistory,
                key: tomb_key,
                value: tomb_value,
            },
            // Stale pointer pointing at the old live row.
            WriteOp::Put {
                ns: Namespace::LatestVersion,
                key: ptr_key.clone(),
                value: format_sys_from(100).into_bytes(),
            },
        ])
        .await
        .unwrap();

        // Backfill corrects the pointer.
        backfill_latest_version(&s, "c").await.unwrap();

        // Pointer must be gone (tombstone is the latest row).
        assert!(
            s.get(Namespace::LatestVersion, &ptr_key)
                .await
                .unwrap()
                .is_none(),
            "stale pointer must be removed after backfill"
        );

        // get_current returns None.
        assert!(
            versioned_get_current(&s, "c", "d1")
                .await
                .unwrap()
                .is_none()
        );
    }

    /// Backfill is idempotent: calling it twice on an up-to-date index is a no-op.
    #[tokio::test]
    async fn backfill_idempotent() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"body", 100, None, None)
            .await
            .unwrap();

        // First call (pointer already correct from versioned_put).
        backfill_latest_version(&s, "c").await.unwrap();
        // Second call — must not corrupt anything.
        backfill_latest_version(&s, "c").await.unwrap();

        let v = versioned_get_current(&s, "c", "d1").await.unwrap().unwrap();
        assert_eq!(v.body, b"body");
    }

    /// Backfill on an empty collection is a no-op and does not error.
    #[tokio::test]
    async fn backfill_empty_collection_noop() {
        let s = mem_storage().await;
        backfill_latest_version(&s, "never_written").await.unwrap();
    }
}
