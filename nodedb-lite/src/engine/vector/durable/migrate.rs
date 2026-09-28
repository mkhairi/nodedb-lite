// SPDX-License-Identifier: Apache-2.0

//! One-shot rewrite of durable vector rows from the earlier key layout.
//!
//! The earlier layout keyed a row `v:<index_key>:<doc_id>`. A named index
//! key is itself `"{collection}:{field}"`, so `v:chat:emb:b` reads both as
//! document `emb:b` of index `chat` and as document `b` of index `chat:emb`.
//! The rewrite resolves each row against what the database already records:
//!
//! 1. An exact `(index_key, doc_id)` binding in the persisted id map.
//! 2. Else the longest known index key followed by `:` (known keys come from
//!    the flushed index list, the stored checkpoints, and the id map).
//! 3. Else the index key ends at the first `:`.
//!
//! Each index's rows move in one `batch_write` that puts the new rows and
//! deletes the old ones, so a crash leaves every row in exactly one layout
//! and a rerun picks up where it stopped. The layout marker in
//! `Namespace::Meta` is written last; once present, the rewrite never runs
//! again.

use std::collections::{BTreeMap, HashMap};

use nodedb_types::Namespace;

use crate::engine::vector::id_map::{PersistedIdEntry, persisted_index_key};
use crate::error::LiteError;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::layout::{LEGACY_ROW_PREFIX, key};

/// `Namespace::Meta` key recording the durable row layout version.
pub(crate) const LAYOUT_MARKER: &[u8] = b"meta:vector_row_layout";

/// The layout version this code writes.
pub(crate) const LAYOUT_VERSION: u8 = 2;

/// Rewrite every earlier-layout row into the current layout, then record the
/// layout marker. A no-op once the marker is present.
///
/// `listed_index_keys` is the flushed index list, one source of known index
/// keys. Returns the number of rows rewritten. Fails when storage fails; the
/// marker is then not written and the next open retries.
pub(crate) async fn migrate_layout<S: StorageEngine>(
    storage: &S,
    listed_index_keys: &[String],
) -> Result<usize, LiteError> {
    let current = storage
        .get(Namespace::Meta, LAYOUT_MARKER)
        .await?
        .and_then(|v| v.first().copied())
        .is_some_and(|version| version >= LAYOUT_VERSION);
    if current {
        return Ok(0);
    }

    let legacy = storage
        .scan_prefix(Namespace::Vector, LEGACY_ROW_PREFIX.as_bytes())
        .await?;
    let mut migrated = 0;
    if !legacy.is_empty() {
        let resolver = LegacyResolver::load(storage, listed_index_keys).await?;
        let mut groups: BTreeMap<String, Vec<WriteOp>> = BTreeMap::new();
        for (old_key, value) in legacy {
            let Some(rest) = old_key
                .strip_prefix(LEGACY_ROW_PREFIX.as_bytes())
                .and_then(|b| std::str::from_utf8(b).ok())
            else {
                tracing::warn!("durable vector row has a non-UTF-8 key; left in place");
                continue;
            };
            let Some((index_key, doc_id)) = resolver.split(rest) else {
                tracing::warn!(
                    key = rest,
                    "durable vector row key has no index key; left in place"
                );
                continue;
            };
            let ops = groups.entry(index_key.to_owned()).or_default();
            ops.push(WriteOp::Put {
                ns: Namespace::Vector,
                key: key(index_key, doc_id),
                value,
            });
            ops.push(WriteOp::Delete {
                ns: Namespace::Vector,
                key: old_key,
            });
            migrated += 1;
        }
        for (index_key, ops) in groups {
            storage.batch_write(&ops).await?;
            tracing::info!(
                index_key,
                rows = ops.len() / 2,
                "durable vector rows rewritten to the current key layout"
            );
        }
    }

    storage
        .put(Namespace::Meta, LAYOUT_MARKER, &[LAYOUT_VERSION])
        .await?;
    Ok(migrated)
}

/// Splits an earlier-layout key body `<index_key>:<doc_id>` using the index
/// keys and bindings the database records.
struct LegacyResolver {
    /// `"{index_key}:{doc_id}"` → index key, from the persisted id map.
    exact: HashMap<String, String>,
    /// Every known index key, longest first.
    index_keys: Vec<String>,
}

impl LegacyResolver {
    async fn load<S: StorageEngine>(
        storage: &S,
        listed_index_keys: &[String],
    ) -> Result<Self, LiteError> {
        let mut index_keys: Vec<String> = listed_index_keys.to_vec();
        for (checkpoint_key, _) in storage.scan_prefix(Namespace::Vector, b"hnsw:").await? {
            if let Some(name) = checkpoint_key
                .strip_prefix(b"hnsw:")
                .and_then(|b| std::str::from_utf8(b).ok())
            {
                index_keys.push(name.to_owned());
            }
        }

        let mut exact: HashMap<String, String> = HashMap::new();
        for entry in persisted_id_entries(storage).await? {
            let Some(index_key) = persisted_index_key(&entry) else {
                continue;
            };
            let body = format!("{index_key}:{}", entry.1);
            let longer = exact.get(&body).is_none_or(|k| k.len() < index_key.len());
            if longer {
                exact.insert(body, index_key.to_owned());
            }
            index_keys.push(index_key.to_owned());
        }

        index_keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
        index_keys.dedup();
        Ok(Self { exact, index_keys })
    }

    /// `(index_key, doc_id)` of one earlier-layout key body. `None` when the
    /// body has no `:` at all.
    fn split<'a>(&'a self, body: &'a str) -> Option<(&'a str, &'a str)> {
        if let Some(index_key) = self.exact.get(body)
            && let Some(doc_id) = body
                .get(index_key.len()..)
                .and_then(|r| r.strip_prefix(':'))
        {
            return Some((index_key.as_str(), doc_id));
        }
        for index_key in &self.index_keys {
            if let Some(doc_id) = body
                .strip_prefix(index_key.as_str())
                .and_then(|r| r.strip_prefix(':'))
                && !doc_id.is_empty()
            {
                return Some((index_key.as_str(), doc_id));
            }
        }
        body.split_once(':')
    }
}

/// The persisted id-map bindings, or none when the blob is absent or
/// unreadable.
async fn persisted_id_entries<S: StorageEngine>(
    storage: &S,
) -> Result<Vec<PersistedIdEntry>, LiteError> {
    let Some(envelope) = storage.get(Namespace::Vector, b"hnsw_id_map").await? else {
        return Ok(Vec::new());
    };
    Ok(crate::storage::checksum::unwrap(&envelope)
        .and_then(|bytes| zerompk::from_msgpack::<Vec<PersistedIdEntry>>(&bytes).ok())
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::vector::durable::load_collection;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    async fn storage_with_legacy_rows(rows: &[&str]) -> PagedbStorageMem {
        let storage = PagedbStorageMem::open_in_memory()
            .await
            .expect("in-memory pagedb");
        for row in rows {
            storage
                .put(
                    Namespace::Vector,
                    format!("v:{row}").as_bytes(),
                    &[0, 0, 128, 63],
                )
                .await
                .expect("put legacy row");
        }
        storage
    }

    #[tokio::test]
    async fn legacy_named_rows_move_to_their_named_index() {
        let storage = storage_with_legacy_rows(&["chat:a", "chat:emb:b"]).await;
        let listed = vec!["chat".to_owned(), "chat:emb".to_owned()];
        assert_eq!(migrate_layout(&storage, &listed).await.expect("migrate"), 2);

        let base = load_collection(&storage, "chat").await.expect("load base");
        assert_eq!(
            base.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>(),
            ["a"]
        );
        let named = load_collection(&storage, "chat:emb")
            .await
            .expect("load named");
        assert_eq!(
            named.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>(),
            ["b"]
        );
        assert!(
            storage
                .scan_prefix(Namespace::Vector, b"v:")
                .await
                .expect("scan")
                .is_empty(),
            "no earlier-layout row remains"
        );
    }

    #[tokio::test]
    async fn persisted_bindings_resolve_ids_that_contain_separators() {
        let storage = storage_with_legacy_rows(&["chat:emb:b"]).await;
        let entries: Vec<PersistedIdEntry> = vec![("chat:0".into(), "emb:b".into(), 0)];
        let blob = zerompk::to_msgpack_vec(&entries).expect("encode");
        storage
            .put(
                Namespace::Vector,
                b"hnsw_id_map",
                &crate::storage::checksum::wrap(&blob),
            )
            .await
            .expect("put id map");
        let listed = vec!["chat:emb".to_owned()];
        migrate_layout(&storage, &listed).await.expect("migrate");

        let base = load_collection(&storage, "chat").await.expect("load base");
        assert_eq!(
            base.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>(),
            ["emb:b"]
        );
    }

    #[tokio::test]
    async fn migration_runs_once() {
        let storage = storage_with_legacy_rows(&["docs:a"]).await;
        assert_eq!(migrate_layout(&storage, &[]).await.expect("first"), 1);
        storage
            .put(Namespace::Vector, b"v:docs:late", &[0, 0, 128, 63])
            .await
            .expect("put");
        assert_eq!(migrate_layout(&storage, &[]).await.expect("second"), 0);
    }
}
