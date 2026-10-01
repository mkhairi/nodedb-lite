// SPDX-License-Identifier: Apache-2.0

//! Cold-start rebuild helpers for FTS (text) and Spatial indices.
//!
//! - `rebuild_text_indices` — two-pass: CRDT scan + DocumentHistory scan for
//!   bitemporal collections.
//! - `rebuild_spatial_indices` — single-pass CRDT scan for geometry fields.

use std::collections::HashSet;

use nodedb_types::Namespace;

use crate::engine::document::history::key::coll_prefix;
use crate::engine::document::history::ops::versioned_get_current;
use crate::engine::document::history::value::VersionTag;
use crate::error::LiteError;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

use crate::nodedb::core::types::NodeDbLite;

/// Meta key prefix for the document bitemporal flag (mirrors `history::ops`).
const META_DOCUMENT_BITEMPORAL_PREFIX: &str = "document_bitemporal:";
const CRDT_REBUILD_ID_COUNT: usize = 256;
const CRDT_REBUILD_ID_BYTES: usize = 1024 * 1024;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Rebuild all text indices from CRDT state and, for bitemporal collections,
    /// from the authoritative `DocumentHistory` table.
    ///
    /// Called once on cold start after CRDT snapshot restore, when no FTS
    /// checkpoint is present or the checkpoint predates per-field indexing.
    /// Each document is indexed whole and per string field, replacing any
    /// restored entry. Two-pass approach:
    ///
    /// **Pass 1 — CRDT scan** (non-bitemporal collections):
    /// Reads every document from the Loro CRDT engine and indexes all string
    /// fields. Bitemporal collections may not have had their CRDT snapshot
    /// flushed before the previous process exited, so Pass 1 alone is
    /// insufficient for them.
    ///
    /// **Pass 2 — DocumentHistory scan** (bitemporal collections only):
    /// Enumerates every collection flagged as bitemporal via `Namespace::Meta`,
    /// prefix-scans `Namespace::DocumentHistory` for unique doc_ids, fetches the
    /// current live version of each, and indexes its string fields. Documents
    /// already indexed in Pass 1 are skipped to avoid duplicate work.
    ///
    /// **Pass 3 — strict rows**: every row of every strict collection, per
    /// STRING column and whole.
    ///
    /// **Pass 4 — columnar rows**: every row of every columnar collection, per
    /// STRING column and whole, plus the geohash of a spatial profile.
    ///
    /// Fails if any document cannot be read or indexed: the rebuild is the
    /// only thing that populates the index on this path, so a swallowed
    /// failure would leave the document permanently unsearchable.
    pub(crate) async fn rebuild_text_indices(&self) -> Result<(), LiteError> {
        // ── Pass 1: CRDT scan (non-bitemporal collections) ───────────────────
        // Collect doc_ids indexed in this pass so Pass 2 can skip duplicates.
        let mut indexed: HashSet<(String, String)> = HashSet::new();

        {
            let crdt = self.crdt.lock_or_recover();
            let collections = crdt.collection_names();
            let mut fts = self.fts_state.manager.lock_or_recover();
            let mut sparse = self.sparse_state.manager.lock_or_recover();

            for collection in &collections {
                if collection.starts_with("__") {
                    continue;
                }
                let mut after_id = None;
                loop {
                    let ids = crdt.live_ids_page(
                        collection,
                        after_id.as_deref(),
                        CRDT_REBUILD_ID_COUNT,
                        CRDT_REBUILD_ID_BYTES,
                    )?;
                    if ids.is_empty() {
                        break;
                    }
                    after_id = ids.last().cloned();
                    for id in ids {
                        if let Some(loro_val) = crdt.read(collection, &id) {
                            let doc =
                                crate::nodedb::convert::loro_value_to_document(&id, &loro_val);
                            fts.index_document_fields(collection, &id, &doc.fields)?;
                            sparse.index_document_fields(collection, &id, &doc.fields);
                            indexed.insert((collection.clone(), id));
                        }
                    }
                }
            }
        }

        // ── Pass 2: DocumentHistory scan (bitemporal collections) ─────────────
        for collection in &self.list_bitemporal_collections().await? {
            for doc_id in &self.collect_doc_ids_from_history(collection).await? {
                // Skip if already indexed from CRDT in Pass 1.
                if indexed.contains(&(collection.clone(), doc_id.clone())) {
                    continue;
                }

                // Tombstoned or missing versions are not indexed.
                let Some(version) =
                    versioned_get_current(&*self.storage, collection, doc_id).await?
                else {
                    continue;
                };
                if version.tag != VersionTag::Live {
                    continue;
                }

                let fields = match nodedb_types::json_msgpack::value_from_msgpack(&version.body) {
                    Ok(nodedb_types::Value::Object(fields)) => fields,
                    Ok(other) => {
                        return Err(LiteError::Serialization {
                            detail: format!(
                                "live version of '{collection}'/'{doc_id}' is a {}, not a document",
                                other.type_name()
                            ),
                        });
                    }
                    Err(e) => {
                        return Err(LiteError::Serialization {
                            detail: format!(
                                "live version of '{collection}'/'{doc_id}' does not decode: {e}"
                            ),
                        });
                    }
                };
                self.fts_state
                    .manager
                    .lock_or_recover()
                    .index_document_fields(collection, doc_id, &fields)?;
                self.sparse_state
                    .manager
                    .lock_or_recover()
                    .index_document_fields(collection, doc_id, &fields);
            }
        }

        // ── Pass 3: strict rows ──────────────────────────────────────────────
        for collection in self.strict.collection_names() {
            let Some(schema) = self.strict.schema(&collection) else {
                continue;
            };
            for values in self.strict.list_rows(&collection).await? {
                crate::engine::index_integration::index_row_text(
                    &collection,
                    &crate::engine::index_integration::row_id(&schema.columns, &values),
                    &schema.columns,
                    &values,
                    &self.fts_state.manager,
                )?;
            }
        }

        // ── Pass 4: columnar rows (text columns and spatial geohash) ─────────
        for collection in self.columnar.collection_names() {
            let Some(schema) = self.columnar.schema(&collection) else {
                continue;
            };
            let profile = self.columnar.profile(&collection);
            for values in self.columnar.list_rows(&collection).await? {
                let row_id = crate::engine::index_integration::row_id(&schema.columns, &values);
                crate::engine::index_integration::index_row_text(
                    &collection,
                    &row_id,
                    &schema.columns,
                    &values,
                    &self.fts_state.manager,
                )?;
                crate::engine::index_integration::index_geohash(
                    &collection,
                    &row_id,
                    &schema,
                    profile.as_ref(),
                    &values,
                    &self.fts_state.manager,
                )?;
            }
        }

        Ok(())
    }

    /// Rebuild spatial indices from CRDT state (cold start fallback).
    ///
    /// Scans all collections for geometry-valued fields and indexes them.
    /// Called when checkpoint restore produces empty spatial indices.
    pub(crate) fn rebuild_spatial_indices(&self) {
        let crdt = self.crdt.lock_or_recover();
        let collections = crdt.collection_names();
        let mut spatial = self.spatial.lock_or_recover();

        for collection in &collections {
            if collection.starts_with("__") {
                continue;
            }
            let ids = crdt.list_ids(collection);
            for id in &ids {
                if let Some(loro_val) = crdt.read(collection, id) {
                    let doc = crate::nodedb::convert::loro_value_to_document(id, &loro_val);
                    for (field, value) in &doc.fields {
                        // Geometry fields are stored as GeoJSON strings.
                        if let nodedb_types::Value::String(s) = value
                            && let Ok(geom) =
                                sonic_rs::from_str::<nodedb_types::geometry::Geometry>(s)
                        {
                            spatial.index_document(collection, field, id, &geom);
                        }
                    }
                }
            }
        }
    }

    /// Return the names of all collections that have the bitemporal flag set.
    ///
    /// Reads `Namespace::Meta` keys prefixed with `document_bitemporal:` and
    /// returns only those whose stored byte equals `0x01` (enabled).
    async fn list_bitemporal_collections(&self) -> Result<Vec<String>, LiteError> {
        let prefix = META_DOCUMENT_BITEMPORAL_PREFIX.as_bytes();
        let entries = self.storage.scan_prefix(Namespace::Meta, prefix).await?;

        let mut names = Vec::new();
        for (key, value) in entries {
            // Value byte 0x01 = bitemporal enabled.
            if value.first().copied() != Some(1) {
                continue;
            }
            let key_str = String::from_utf8(key).map_err(|e| LiteError::Serialization {
                detail: format!("bitemporal flag key is not UTF-8: {e}"),
            })?;
            if let Some(name) = key_str.strip_prefix(META_DOCUMENT_BITEMPORAL_PREFIX) {
                names.push(name.to_owned());
            }
        }
        Ok(names)
    }

    /// Collect unique doc_ids from `Namespace::DocumentHistory` for `collection`.
    ///
    /// Key format: `{collection}:{doc_id}\x00{system_from_ms:020}`. Splits on
    /// the NUL separator to extract `{doc_id}` and deduplicates across versions.
    async fn collect_doc_ids_from_history(
        &self,
        collection: &str,
    ) -> Result<Vec<String>, LiteError> {
        let prefix = coll_prefix(collection);
        let entries = self
            .storage
            .scan_prefix(Namespace::DocumentHistory, &prefix)
            .await?;

        let prefix_str = format!("{collection}:");
        let mut seen: HashSet<String> = HashSet::new();
        let mut ids: Vec<String> = Vec::new();

        for (key, _value) in &entries {
            let key_str = std::str::from_utf8(key).map_err(|e| LiteError::Serialization {
                detail: format!("document history key of '{collection}' is not UTF-8: {e}"),
            })?;
            // key_str = "{collection}:{doc_id}\x00{timestamp}"
            // Split on NUL to get the "{collection}:{doc_id}" part.
            let coll_and_id = key_str.split('\x00').next().unwrap_or(key_str);
            let Some(doc_id) = coll_and_id.strip_prefix(&prefix_str) else {
                continue;
            };
            if seen.insert(doc_id.to_owned()) {
                ids.push(doc_id.to_owned());
            }
        }

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use loro::LoroValue;
    use nodedb_types::text_search::TextSearchParams;

    use super::{CRDT_REBUILD_ID_COUNT, NodeDbLite};
    use crate::nodedb::lock_ext::LockExt;
    use crate::{LiteConfig, PagedbStorageMem};

    #[tokio::test]
    async fn text_rebuild_indexes_every_crdt_id_page() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let config = LiteConfig {
            auto_flush_ms: 0,
            sync_enabled: false,
            ..LiteConfig::default()
        };
        let db = NodeDbLite::open_with_config(storage, config).await.unwrap();
        let count = CRDT_REBUILD_ID_COUNT + 17;
        let mut expected = HashSet::new();
        {
            let mut crdt = db.crdt.lock_or_recover();
            for position in 0..count {
                let id = format!("doc{position:04}");
                crdt.upsert(
                    "articles",
                    &id,
                    &[("body", LoroValue::String("pagecoverage".into()))],
                )
                .unwrap();
                expected.insert(id);
            }
        }
        {
            let fts = db.fts_state.manager.lock_or_recover();
            assert!(
                fts.search(
                    "articles",
                    "body",
                    "pagecoverage",
                    count,
                    &TextSearchParams::default()
                )
                .unwrap()
                .is_empty()
            );
        }

        db.rebuild_text_indices().await.unwrap();

        let fts = db.fts_state.manager.lock_or_recover();
        for field in ["body", ""] {
            let results = fts
                .search(
                    "articles",
                    field,
                    "pagecoverage",
                    count,
                    &TextSearchParams::default(),
                )
                .unwrap();
            let actual: HashSet<String> = results.into_iter().map(|result| result.doc_id).collect();
            assert_eq!(actual, expected, "field '{field}' omits rebuilt documents");
        }
    }
}
