// SPDX-License-Identifier: Apache-2.0

//! Document engine helpers for `NodeDbLite`.
//!
//! Read-path strategy for bitemporal collections (mirrors Origin's choice in
//! `nodedb/src/engine/document/store/engine/get.rs:10-28`):
//!
//! **Option A — switch the read path entirely.**  When a collection is
//! bitemporal, `document_get` reads from `versioned_get_current` (the history
//! table) rather than the CRDT store.  The CRDT store still receives the write
//! via `document_put` so that sync and current-state access both work, but for
//! bitemporal collections the history table is authoritative for reads and
//! `document_delete` appends a tombstone rather than performing a hard delete.

use nodedb_types::document::Document;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::engine::document::history::ops::{
    is_bitemporal, versioned_get_as_of, versioned_get_current, versioned_put, versioned_tombstone,
};
// Note: versioned_get_current is used only for the non-as_of path of document_get.
use crate::engine::document::history::value::DecodedVersion;
use crate::nodedb::LockExt;
use crate::nodedb::NodeDbLite;
use crate::nodedb::convert::{document_to_msgpack, loro_value_to_document, value_to_loro};
use crate::runtime::{monotonic_millis_i64, now_millis_i64};
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Read a single document by id.
    ///
    /// For bitemporal collections, delegates to `versioned_get_current` so the
    /// history table is the source of truth (mirrors Origin get.rs:10-28).
    /// For plain collections, reads directly from the CRDT store.
    pub(super) async fn document_get_impl(
        &self,
        collection: &str,
        id: &str,
    ) -> NodeDbResult<Option<Document>> {
        if is_bitemporal(&*self.storage, collection)
            .await
            .map_err(NodeDbError::storage)?
        {
            let version = versioned_get_current(&*self.storage, collection, id)
                .await
                .map_err(NodeDbError::storage)?;
            return Ok(version.map(|v| decoded_version_to_document(id, &v)));
        }

        let crdt = self.crdt.lock_or_recover();
        let Some(value) = crdt.read(collection, id) else {
            return Ok(None);
        };
        Ok(Some(loro_value_to_document(id, &value)))
    }

    /// Upsert a document.
    ///
    /// For bitemporal collections: writes to the CRDT store first (so sync and
    /// current-state CRDT reads continue to work), then appends a versioned
    /// `LIVE` record to the history table with `system_from_ms = now`.
    ///
    /// For plain collections: unchanged CRDT put + FTS indexing.
    pub(super) async fn document_put_impl(
        &self,
        collection: &str,
        doc: Document,
    ) -> NodeDbResult<()> {
        let guard = self.fts_state.admit_mutation().await;
        let result = async {
            if self.governor.worst_engine_pressure() == nodedb_mem::PressureLevel::Emergency {
                return Err(NodeDbError::storage(
                    crate::error::LiteError::Backpressure {
                        detail: "document put rejected: memory governor is at Emergency pressure"
                            .into(),
                    },
                ));
            }

            let doc_id = if doc.id.is_empty() {
                nodedb_types::id_gen::uuid_v7()
            } else {
                doc.id.clone()
            };

            let bitemporal = is_bitemporal(&*self.storage, collection)
                .await
                .map_err(NodeDbError::storage)?;
            let _index_build = self.hold_bitemporal_build(bitemporal).await;
            self.query_engine.indexes.revive(collection, &doc_id);

            // Always write to the CRDT store (current-state + sync).
            {
                let mut crdt = self.crdt.lock_or_recover();
                let fields: Vec<(&str, loro::LoroValue)> = doc
                    .fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), value_to_loro(v)))
                    .collect();
                let mutation_id = crdt
                    .upsert(collection, &doc_id, &fields)
                    .map_err(NodeDbError::from)?;
                // Keep local-only documents out of the outbound CRDT delta stream.
                if !self.should_sync_doc(collection, &doc.fields) {
                    crdt.drop_pending(mutation_id);
                }
            }

            // For bitemporal collections, also record the versioned history entry.
            if bitemporal {
                let now_ms = monotonic_millis_i64();
                let body = document_to_msgpack(&doc);
                versioned_put(
                    &*self.storage,
                    collection,
                    &doc_id,
                    &body,
                    now_ms,
                    // system-time (`now_ms`) is monotonic for a unique history key;
                    // valid_from must stay true wall-clock so "valid as-of now"
                    // queries see the row immediately (no monotonic future-skew).
                    Some(now_millis_i64()),
                    None,
                )
                .await
                .map_err(NodeDbError::storage)?;
            }

            self.index_document_text(collection, &doc_id, &doc.fields)?;
            self.index_document_sparse(collection, &doc_id, &doc.fields);

            Ok(())
        }
        .await;
        guard.finish(result)
    }

    /// Delete a document.
    ///
    /// For bitemporal collections: appends a Tombstone version to the history
    /// table (preserves history for AS-OF queries) but does NOT hard-delete from
    /// the CRDT store — the LIVE history entry takes precedence for reads via
    /// `document_get` which now routes through `versioned_get_current`.
    ///
    /// For plain collections: hard-delete from CRDT + FTS removal (unchanged).
    pub(super) async fn document_delete_impl(
        &self,
        collection: &str,
        id: &str,
    ) -> NodeDbResult<()> {
        let guard = self.fts_state.admit_mutation().await;
        let result = async {
            if is_bitemporal(&*self.storage, collection)
                .await
                .map_err(NodeDbError::storage)?
            {
                let _index_build = self.hold_bitemporal_build(true).await;
                let now_ms = monotonic_millis_i64();
                // Monotonic system-time key; wall-clock valid_from so the deletion is
                // visible to "valid as-of now" queries immediately (see versioned_put).
                versioned_tombstone(
                    &*self.storage,
                    collection,
                    id,
                    now_ms,
                    Some(now_millis_i64()),
                )
                .await
                .map_err(NodeDbError::storage)?;
                // The CRDT copy stays for sync, so the secondary indexes drop the
                // row here rather than following the CRDT store.
                self.query_engine.indexes.tombstone(collection, [id]);
                // FTS removal still applies — the document is logically gone now.
                self.remove_document_text(collection, id)?;
                self.remove_document_sparse(collection, id);
                return Ok(());
            }

            let mut crdt = self.crdt.lock_or_recover();
            crdt.delete(collection, id).map_err(NodeDbError::storage)?;
            drop(crdt);

            self.remove_document_text(collection, id)?;
            self.remove_document_sparse(collection, id);

            Ok(())
        }
        .await;
        guard.finish(result)
    }

    /// Read a document as-of a system time, optionally filtered by valid_time.
    ///
    /// Only valid on collections created `WITH (bitemporal=true)`. Returns an
    /// error when called on a plain document collection.
    ///
    /// When `as_of_ms` is `None`, delegates to `versioned_get_current` (same
    /// result as `document_get` for bitemporal collections). When `as_of_ms`
    /// is `Some(t)`, returns the version visible at system time `t`.
    pub(super) async fn document_get_as_of_impl(
        &self,
        collection: &str,
        id: &str,
        as_of_ms: Option<i64>,
        valid_time_ms: Option<i64>,
    ) -> NodeDbResult<Option<Document>> {
        if !is_bitemporal(&*self.storage, collection)
            .await
            .map_err(NodeDbError::storage)?
        {
            return Err(NodeDbError::storage(
                "document_get_as_of requires a collection created WITH (bitemporal=true)",
            ));
        }

        // When as_of_ms is None, use i64::MAX as the system time so we
        // always see the most-recent version — but still apply the
        // valid_time_ms filter via versioned_get_as_of.  Using
        // versioned_get_current would skip the valid_time filter.
        let sys_as_of = as_of_ms.unwrap_or(i64::MAX);
        let version = versioned_get_as_of(&*self.storage, collection, id, sys_as_of, valid_time_ms)
            .await
            .map_err(NodeDbError::storage)?;

        Ok(version.map(|v| decoded_version_to_document(id, &v)))
    }

    /// Put a document with explicit valid-time bounds into a bitemporal collection.
    ///
    /// Only valid on collections created `WITH (bitemporal=true)`. Returns an
    /// error when called on a plain document collection.
    pub(super) async fn document_put_with_valid_time_impl(
        &self,
        collection: &str,
        doc: Document,
        valid_from_ms: Option<i64>,
        valid_until_ms: Option<i64>,
    ) -> NodeDbResult<()> {
        let guard = self.fts_state.admit_mutation().await;
        let result = async {
        if !is_bitemporal(&*self.storage, collection)
            .await
            .map_err(NodeDbError::storage)?
        {
            return Err(NodeDbError::storage(
                "document_put_with_valid_time requires a collection created WITH (bitemporal=true)",
            ));
        }

        let doc_id = if doc.id.is_empty() {
            nodedb_types::id_gen::uuid_v7()
        } else {
            doc.id.clone()
        };
        let _index_build = self.hold_bitemporal_build(true).await;
        self.query_engine.indexes.revive(collection, &doc_id);

        // Write to CRDT store for current-state access + sync.
        {
            let mut crdt = self.crdt.lock_or_recover();
            let fields: Vec<(&str, loro::LoroValue)> = doc
                .fields
                .iter()
                .map(|(k, v)| (k.as_str(), value_to_loro(v)))
                .collect();
            crdt.upsert(collection, &doc_id, &fields)
                .map_err(NodeDbError::from)?;
        }

        let now_ms = monotonic_millis_i64();
        let body = document_to_msgpack(&doc);
        versioned_put(
            &*self.storage,
            collection,
            &doc_id,
            &body,
            now_ms,
            // Monotonic system-time key; an unspecified valid_from means "valid
            // from now", which must be true wall-clock (not the monotonic
            // system time) so it never lands ahead of a concurrent
            // valid_until = now (which would invert the window).
            valid_from_ms.or_else(|| Some(now_millis_i64())),
            valid_until_ms,
        )
        .await
        .map_err(NodeDbError::storage)?;

        self.index_document_text(collection, &doc_id, &doc.fields)?;
        self.index_document_sparse(collection, &doc_id, &doc.fields);

        Ok(())

        }.await;
        guard.finish(result)
    }
}

impl<S: StorageEngine> NodeDbLite<S> {
    /// For a bitemporal write, hold off any index build reading the
    /// collection's history until the write — CRDT row and history version —
    /// is complete. `None` for any other write.
    pub(super) async fn hold_bitemporal_build(
        &self,
        bitemporal: bool,
    ) -> Option<tokio::sync::RwLockReadGuard<'_, ()>> {
        if bitemporal {
            Some(self.query_engine.indexes.bitemporal_build.read().await)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Decode a `DecodedVersion` body (msgpack bytes) into a `Document`.
///
/// Uses `nodedb_types::json_msgpack::value_from_msgpack` for decoding,
/// falling back to an empty document on any parse error.
fn decoded_version_to_document(id: &str, version: &DecodedVersion) -> Document {
    use nodedb_types::value::Value;

    let mut doc = Document::new(id);
    if version.body.is_empty() {
        return doc;
    }

    if let Ok(Value::Object(fields)) = nodedb_types::json_msgpack::value_from_msgpack(&version.body)
    {
        for (k, v) in fields {
            doc.set(k, v);
        }
    }

    doc
}

#[cfg(test)]
mod tests {
    use crate::{LiteConfig, NodeDbLite, PagedbStorageMem};
    use nodedb_client::NodeDb;
    use nodedb_types::document::Document;
    use nodedb_types::value::Value;

    #[tokio::test]
    async fn document_write_waits_for_admission_and_nested_sql_writes_finish() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let db = NodeDbLite::open_with_config(
            storage,
            LiteConfig {
                auto_flush_ms: 0,
                sync_enabled: false,
                ..LiteConfig::default()
            },
        )
        .await
        .unwrap();
        let mut doc = Document::new("a");
        doc.set("body", Value::String("coordinated source".into()));
        let permit = db.fts_state.admit_exclusive().await;
        let mut write = Box::pin(db.document_put("docs", doc));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut write)
                .await
                .is_err()
        );
        assert!(db.document_get("docs", "a").await.unwrap().is_none());
        drop(permit);
        write.await.unwrap();
        assert!(db.document_get("docs", "a").await.unwrap().is_some());
        db.execute_sql("CREATE COLLECTION copied", &[])
            .await
            .unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            db.execute_sql("INSERT INTO copied SELECT * FROM docs", &[]),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.rows_affected, 1);
    }
}
