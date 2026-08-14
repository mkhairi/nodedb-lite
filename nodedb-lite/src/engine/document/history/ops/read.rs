// SPDX-License-Identifier: Apache-2.0

//! Read primitives: current-version resolution, point-in-time lookups, and
//! live-document scans.

use nodedb_types::Namespace;

use crate::error::LiteError;
use crate::storage::engine::{KvPair, StorageEngine};

use super::super::key::{
    coll_prefix, doc_prefix, latest_version_key, parse_sys_from, versioned_doc_key,
};
use super::super::value::{DecodedVersion, VersionTag, decode_value};

/// Read the most recent `LIVE` version for `(collection, doc_id)`.
///
/// Uses the `LatestVersion` index for an O(1) pointer lookup followed by a
/// single `DocumentHistory` fetch.  Returns `None` when the pointer is absent
/// (document never written, tombstoned, or GDPR-erased).
///
/// Call [`backfill_latest_version`](super::backfill::backfill_latest_version)
/// on collection open to populate the index for databases written before this
/// index was introduced.
pub async fn versioned_get_current<S: StorageEngine>(
    storage: &S,
    collection: &str,
    doc_id: &str,
) -> Result<Option<DecodedVersion>, LiteError> {
    let pointer_key = latest_version_key(collection, doc_id);
    let Some(pointer_bytes) = storage.get(Namespace::LatestVersion, &pointer_key).await? else {
        return Ok(None);
    };

    let sys_from_str =
        std::str::from_utf8(&pointer_bytes).map_err(|_| LiteError::Serialization {
            detail: "LatestVersion pointer is not valid UTF-8".into(),
        })?;
    let sys_from_ms: i64 = sys_from_str
        .trim()
        .parse()
        .map_err(|_| LiteError::Serialization {
            detail: format!("LatestVersion pointer is not a valid i64 decimal: {sys_from_str:?}"),
        })?;

    let history_key = versioned_doc_key(collection, doc_id, sys_from_ms)?;
    let Some(history_bytes) = storage
        .get(Namespace::DocumentHistory, &history_key)
        .await?
    else {
        // Pointer refers to a missing history row — storage inconsistency.
        return Err(LiteError::Serialization {
            detail: format!(
                "LatestVersion pointer for {collection}/{doc_id} points to \
                 system_from_ms={sys_from_ms} but no DocumentHistory row exists"
            ),
        });
    };

    let decoded = decode_value(&history_bytes)?;
    if decoded.is_live() {
        Ok(Some(decoded))
    } else {
        // Pointer left stale (e.g. GdprErased row that wiped the live tag).
        Ok(None)
    }
}

/// Read the version that was current at `system_as_of_ms`.
///
/// Scans all history rows for the document in ascending key order and finds
/// the last version where `system_from_ms <= system_as_of_ms`. If that version
/// is not `Live`, returns `None`.
///
/// When `valid_time_ms` is `Some(vt)`, the returned version must additionally
/// satisfy `valid_from_ms <= vt < valid_until_ms`. Returns `None` if the
/// version visible at `system_as_of_ms` does not cover `valid_time_ms`.
pub async fn versioned_get_as_of<S: StorageEngine>(
    storage: &S,
    collection: &str,
    doc_id: &str,
    system_as_of_ms: i64,
    valid_time_ms: Option<i64>,
) -> Result<Option<DecodedVersion>, LiteError> {
    let prefix = doc_prefix(collection, doc_id);
    let entries = storage
        .scan_prefix(Namespace::DocumentHistory, &prefix)
        .await?;

    // Walk entries in reverse (most-recent first). The first entry where
    // system_from_ms <= system_as_of_ms is the version visible at that point
    // in system time.
    for (_key, value) in entries.iter().rev() {
        let decoded = decode_value(value)?;
        let sys_from = parse_sys_from(_key).ok_or_else(|| LiteError::Serialization {
            detail: "document history key missing NUL separator".into(),
        })?;

        if sys_from > system_as_of_ms {
            // This version was written after the requested point — skip.
            continue;
        }

        // This is the version visible at system_as_of_ms.
        if decoded.tag != VersionTag::Live {
            return Ok(None);
        }

        // Apply valid-time filter if requested.
        if let Some(vt) = valid_time_ms
            && (vt < decoded.valid_from_ms || vt >= decoded.valid_until_ms)
        {
            return Ok(None);
        }

        return Ok(Some(decoded));
    }

    Ok(None)
}

/// Scan all live documents in `collection` from the history table.
///
/// Scans every history row under the collection prefix, groups them by
/// `doc_id`, and retains only documents whose most-recent row (highest
/// `system_from_ms`) is tagged `Live`.  Tombstoned and GDPR-erased documents
/// are excluded.
///
/// Returns `(doc_id, body_bytes)` pairs where `body_bytes` is the raw
/// MessagePack body of the current live version (empty `Vec` if the live
/// entry has an empty body).
///
/// This is the authoritative source for bitemporal collection contents because
/// the CRDT Loro snapshot may lag storage (it is only saved on explicit flush).
pub async fn scan_live_documents<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<Vec<(String, Vec<u8>)>, LiteError> {
    let mut out = Vec::new();
    let mut collect = |id: &str, body: Vec<u8>| -> Result<bool, LiteError> {
        out.push((id.to_owned(), body));
        Ok(true)
    };
    for_each_live_document(storage, collection, &mut collect).await?;
    Ok(out)
}

/// Receiver of each live `(doc_id, body)` a live scan resolves. Returning
/// `false` ends the walk.
pub type LiveDocFn<'a> = dyn FnMut(&str, Vec<u8>) -> Result<bool, LiteError> + Send + 'a;

/// Stream the live documents of `collection` in `doc_id` order.
///
/// The streaming form of [`scan_live_documents`]: `on_doc` receives each live
/// `(doc_id, body)` as it is resolved and returns `false` to end the walk, so a
/// caller with a satisfied `LIMIT` stops the scan instead of paying for the rest
/// of the collection.
///
/// Reducing versions to live rows does not need the collection in memory, only
/// one document at a time. History keys are `{coll}:{doc_id}\x00{sys_from:020}`,
/// so every version of a document is contiguous and ascending in the key order
/// the scan already walks: the reduction holds the running latest version of the
/// document currently under the cursor and emits it — if it is `Live` — the
/// moment a key for the next document arrives. Which version wins is decided by
/// exactly the rule the materialising form used (highest `system_from_ms` per
/// `doc_id`); only the residency changes.
pub async fn for_each_live_document<S: StorageEngine>(
    storage: &S,
    collection: &str,
    on_doc: &mut LiveDocFn<'_>,
) -> Result<(), LiteError> {
    let prefix = coll_prefix(collection);
    // The running latest version of the document under the cursor.
    let mut pending: Option<(String, VersionTag, Vec<u8>)> = None;
    let mut stopped = false;

    {
        let mut on_entry = |(key, value): KvPair| -> Result<bool, LiteError> {
            // Extract doc_id from the key by splitting at the NUL separator.
            let Some(after_prefix) = key.get(prefix.len()..) else {
                return Ok(true);
            };
            let Some(nul) = after_prefix.iter().position(|&b| b == 0) else {
                return Ok(true);
            };
            let Ok(doc_id) = std::str::from_utf8(&after_prefix[..nul]) else {
                return Ok(true);
            };
            // An undecodable row is skipped, leaving the previous version of the
            // document standing — the same outcome as the map form.
            let Ok(decoded) = decode_value(&value) else {
                return Ok(true);
            };

            match &mut pending {
                // Ascending keys mean a later row of the same document is a
                // later version, so it replaces the one held.
                Some((id, tag, body)) if id == doc_id => {
                    *tag = decoded.tag;
                    *body = decoded.body;
                }
                _ => {
                    let flushed = pending.replace((doc_id.to_owned(), decoded.tag, decoded.body));
                    if let Some((id, tag, body)) = flushed
                        && tag == VersionTag::Live
                        && !on_doc(&id, body)?
                    {
                        stopped = true;
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        };

        storage
            .scan_prefix_streaming(Namespace::DocumentHistory, &prefix, &mut on_entry)
            .await?;
    }

    // The last document has no successor key to flush it.
    if !stopped
        && let Some((id, tag, body)) = pending
        && tag == VersionTag::Live
    {
        on_doc(&id, body)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::storage::engine::WriteOp;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    use super::super::super::key::{format_sys_from, latest_version_key, versioned_doc_key};
    use super::super::super::value::encode_value;
    use super::super::write::{versioned_put, versioned_tombstone};
    use super::*;

    async fn mem_storage() -> PagedbStorageMem {
        PagedbStorageMem::open_in_memory()
            .await
            .expect("open in-memory storage")
    }

    /// Insert a document and verify `versioned_get_current` returns it via the
    /// O(1) LatestVersion pointer, and the pointer is present in storage.
    #[tokio::test]
    async fn latest_version_insert_pointer_present() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"hello", 100, None, None)
            .await
            .unwrap();

        // Pointer must be present.
        let ptr_key = latest_version_key("c", "d1");
        let ptr = s
            .get(Namespace::LatestVersion, &ptr_key)
            .await
            .unwrap()
            .expect("LatestVersion pointer must exist after insert");
        assert_eq!(ptr, format_sys_from(100).into_bytes());

        // get_current returns the live row.
        let v = versioned_get_current(&s, "c", "d1").await.unwrap().unwrap();
        assert_eq!(v.body, b"hello");
        assert!(v.is_live());
    }

    /// Update a document (two successive puts): pointer tracks the new version.
    #[tokio::test]
    async fn latest_version_update_pointer_tracks_new() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"v1", 100, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "d1", b"v2", 200, None, None)
            .await
            .unwrap();

        // Pointer points to v2.
        let ptr_key = latest_version_key("c", "d1");
        let ptr = s
            .get(Namespace::LatestVersion, &ptr_key)
            .await
            .unwrap()
            .expect("pointer must exist");
        assert_eq!(ptr, format_sys_from(200).into_bytes());

        // get_current returns v2.
        let v = versioned_get_current(&s, "c", "d1").await.unwrap().unwrap();
        assert_eq!(v.body, b"v2");

        // Old version still accessible via as_of.
        let v1 = versioned_get_as_of(&s, "c", "d1", 150, None)
            .await
            .unwrap()
            .expect("v1 visible at t=150");
        assert_eq!(v1.body, b"v1");
    }

    /// Tombstone removes the pointer; get_current returns None.
    #[tokio::test]
    async fn latest_version_tombstone_removes_pointer() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"hello", 100, None, None)
            .await
            .unwrap();
        versioned_tombstone(&s, "c", "d1", 200, None).await.unwrap();

        // Pointer must be absent after tombstone.
        let ptr_key = latest_version_key("c", "d1");
        let ptr = s.get(Namespace::LatestVersion, &ptr_key).await.unwrap();
        assert!(ptr.is_none(), "pointer must be deleted after tombstone");

        // get_current returns None.
        assert!(
            versioned_get_current(&s, "c", "d1")
                .await
                .unwrap()
                .is_none()
        );
    }

    /// Multiple updates followed by a tombstone: pointer gone, history preserved.
    #[tokio::test]
    async fn latest_version_multi_update_then_tombstone() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"v1", 100, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "d1", b"v2", 200, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "d1", b"v3", 300, None, None)
            .await
            .unwrap();
        versioned_tombstone(&s, "c", "d1", 400, None).await.unwrap();

        // No current version.
        assert!(
            versioned_get_current(&s, "c", "d1")
                .await
                .unwrap()
                .is_none()
        );

        // All historical versions still accessible.
        for (t, body) in [(150, b"v1"), (250, b"v2"), (350, b"v3")] {
            let v = versioned_get_as_of(&s, "c", "d1", t, None)
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("version at t={t} must be present"));
            assert_eq!(v.body.as_slice(), body as &[u8]);
        }
    }

    /// Original put-then-get test (kept for regression coverage).
    #[tokio::test]
    async fn put_get_current_roundtrip() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"hello", 100, None, None)
            .await
            .unwrap();
        let v = versioned_get_current(&s, "c", "d1").await.unwrap().unwrap();
        assert_eq!(v.body, b"hello");
        assert!(v.is_live());
    }

    /// Original tombstone-hides-live test (kept for regression coverage).
    #[tokio::test]
    async fn tombstone_hides_live() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "d1", b"hello", 100, None, None)
            .await
            .unwrap();
        versioned_tombstone(&s, "c", "d1", 200, None).await.unwrap();
        assert!(
            versioned_get_current(&s, "c", "d1")
                .await
                .unwrap()
                .is_none()
        );
    }

    /// The live set is the latest version per document, and neighbouring
    /// document ids (`a` / `ab`, whose history keys interleave in the byte
    /// order the scan walks) must not bleed into each other's reduction.
    #[tokio::test]
    async fn live_reduction_is_latest_version_per_document() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "a", b"a-v1", 100, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "a", b"a-v2", 300, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "ab", b"ab-v1", 200, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "b", b"b-v1", 100, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "b", b"b-v2", 200, None, None)
            .await
            .unwrap();
        versioned_tombstone(&s, "c", "b", 400, None).await.unwrap();

        let mut docs = scan_live_documents(&s, "c").await.unwrap();
        docs.sort();
        assert_eq!(
            docs,
            vec![
                ("a".to_owned(), b"a-v2".to_vec()),
                ("ab".to_owned(), b"ab-v1".to_vec()),
            ]
        );
    }

    /// A caller that stops after the first document ends the storage walk
    /// there: the scan reads one page, not the collection.
    #[tokio::test]
    async fn early_stop_ends_the_storage_walk() {
        let s = MeteredStorage::new(mem_storage().await);
        write_history_rows(&s, "c", 4_000).await;

        let mut seen = Vec::new();
        let mut stop_after_first = |id: &str, _body: Vec<u8>| -> Result<bool, LiteError> {
            seen.push(id.to_owned());
            Ok(false)
        };
        for_each_live_document(&s, "c", &mut stop_after_first)
            .await
            .unwrap();

        assert_eq!(seen.len(), 1, "the walk stopped at the first document");
        let read = s.entries_read();
        assert!(
            read <= crate::storage::engine::SCAN_PAGE,
            "an early stop must not read past the first page: read {read} of 4000 rows"
        );
    }

    /// A full live scan still visits every row, but never holds more than one
    /// page of the collection at a time.
    #[tokio::test]
    async fn full_scan_streams_in_bounded_pages() {
        let s = MeteredStorage::new(mem_storage().await);
        let rows = 4_000;
        write_history_rows(&s, "c", rows).await;

        let docs = scan_live_documents(&s, "c").await.unwrap();

        assert_eq!(docs.len(), rows, "every document is live and returned");
        assert_eq!(s.entries_read(), rows, "the whole history was visited");
        let page = s.max_batch();
        assert!(
            page <= crate::storage::engine::SCAN_PAGE,
            "no single storage batch may hold the collection: largest was {page} of {rows}"
        );
    }

    /// Write `count` single-version live documents directly as history rows —
    /// one batch, no per-document write transaction.
    async fn write_history_rows(storage: &impl StorageEngine, collection: &str, count: usize) {
        let ops: Vec<WriteOp> = (0..count)
            .map(|i| WriteOp::Put {
                ns: Namespace::DocumentHistory,
                key: versioned_doc_key(collection, &format!("doc{i:06}"), 100)
                    .expect("history key"),
                value: encode_value(VersionTag::Live, 100, i64::MAX, b"body"),
            })
            .collect();
        storage.batch_write(&ops).await.expect("seed history rows");
    }

    /// Storage double that records how much of the store a scan actually read:
    /// total entries handed out, and the largest single batch — the quantity
    /// that used to be the whole collection.
    struct MeteredStorage<S: StorageEngine> {
        inner: S,
        entries_read: std::sync::atomic::AtomicUsize,
        max_batch: std::sync::atomic::AtomicUsize,
    }

    impl<S: StorageEngine> MeteredStorage<S> {
        fn new(inner: S) -> Self {
            Self {
                inner,
                entries_read: std::sync::atomic::AtomicUsize::new(0),
                max_batch: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn entries_read(&self) -> usize {
            self.entries_read.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn max_batch(&self) -> usize {
            self.max_batch.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn record(&self, batch: usize) {
            self.entries_read
                .fetch_add(batch, std::sync::atomic::Ordering::Relaxed);
            self.max_batch
                .fetch_max(batch, std::sync::atomic::Ordering::Relaxed);
        }
    }

    // Deliberately does not override `scan_prefix_streaming`, so these tests
    // measure the trait's default paging implementation.
    #[async_trait::async_trait]
    impl<S: StorageEngine> StorageEngine for MeteredStorage<S> {
        async fn get(&self, ns: Namespace, key: &[u8]) -> Result<Option<Vec<u8>>, LiteError> {
            self.inner.get(ns, key).await
        }

        async fn put(&self, ns: Namespace, key: &[u8], value: &[u8]) -> Result<(), LiteError> {
            self.inner.put(ns, key, value).await
        }

        async fn delete(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
            self.inner.delete(ns, key).await
        }

        async fn scan_prefix(
            &self,
            ns: Namespace,
            prefix: &[u8],
        ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
            let out = self.inner.scan_prefix(ns, prefix).await?;
            self.record(out.len());
            Ok(out)
        }

        async fn batch_write(&self, ops: &[WriteOp]) -> Result<(), LiteError> {
            self.inner.batch_write(ops).await
        }

        async fn count(&self, ns: Namespace) -> Result<u64, LiteError> {
            self.inner.count(ns).await
        }

        async fn scan_range(
            &self,
            ns: Namespace,
            start: &[u8],
            limit: usize,
        ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
            let out = self.inner.scan_range(ns, start, limit).await?;
            self.record(out.len());
            Ok(out)
        }

        async fn scan_range_bounded(
            &self,
            ns: Namespace,
            start: Option<&[u8]>,
            end: Option<&[u8]>,
            limit: Option<usize>,
        ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
            let out = self.inner.scan_range_bounded(ns, start, end, limit).await?;
            self.record(out.len());
            Ok(out)
        }
    }

    /// Live scan returns only the current live version per doc, skipping
    /// tombstoned docs.
    #[tokio::test]
    async fn scan_live_documents_skips_tombstoned() {
        let s = mem_storage().await;
        versioned_put(&s, "c", "alive", b"body", 100, None, None)
            .await
            .unwrap();
        versioned_put(&s, "c", "dead", b"body", 100, None, None)
            .await
            .unwrap();
        versioned_tombstone(&s, "c", "dead", 200, None)
            .await
            .unwrap();

        let mut docs = scan_live_documents(&s, "c").await.unwrap();
        docs.sort();
        assert_eq!(docs, vec![("alive".to_owned(), b"body".to_vec())]);
    }

    // Directly-written history rows exercise the value/key codecs without going
    // through versioned_put (used by the backfill tests, mirrored here for the
    // read path).
    #[tokio::test]
    async fn get_current_reads_pointerless_row_after_manual_pointer() {
        let s = mem_storage().await;
        let history_key = versioned_doc_key("c", "d1", 100).unwrap();
        let history_value = encode_value(VersionTag::Live, 100, i64::MAX, b"body");
        let ptr_key = latest_version_key("c", "d1");
        s.batch_write(&[
            WriteOp::Put {
                ns: Namespace::DocumentHistory,
                key: history_key,
                value: history_value,
            },
            WriteOp::Put {
                ns: Namespace::LatestVersion,
                key: ptr_key,
                value: format_sys_from(100).into_bytes(),
            },
        ])
        .await
        .unwrap();

        let v = versioned_get_current(&s, "c", "d1").await.unwrap().unwrap();
        assert_eq!(v.body, b"body");
    }
}
