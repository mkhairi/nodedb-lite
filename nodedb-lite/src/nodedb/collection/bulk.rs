//! Bulk update and delete by predicate (ScanFilter-based).
//!
//! Holds CRDT lock across scan+write to prevent concurrent modification
//! between the filter evaluation and the mutation application.

use std::collections::HashMap;

use nodedb_types::error::{NodeDbError, NodeDbResult};
use nodedb_types::value::Value;

use super::super::convert::value_to_loro;
use super::super::{LockExt, NodeDbLite};
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Bulk update documents matching a predicate.
    ///
    /// Scans all documents, evaluates `ScanFilter` predicates, and applies
    /// `updates` to matching documents — all under a single CRDT lock to
    /// prevent concurrent writes from being lost between scan and update.
    ///
    /// Returns the number of documents updated.
    pub fn bulk_update(
        &self,
        collection: &str,
        filters: &[nodedb_query::ScanFilter],
        updates: &HashMap<String, Value>,
    ) -> NodeDbResult<u64> {
        let guard = self
            .fts_state
            .try_admit_mutation("bulk_update")
            .map_err(NodeDbError::from)?;
        let result = (|| {
            // Single lock for scan + write: no gap for concurrent modifications.
            let mut crdt = self.crdt.lock_or_recover();
            let ids = crdt.list_ids(collection);

            let mut matching_ids = Vec::new();
            for id in &ids {
                if let Some(loro_val) = crdt.read(collection, id) {
                    let doc = crate::nodedb::convert::loro_value_to_document(id, &loro_val);
                    let json = serde_json::to_value(&doc.fields).unwrap_or_default();
                    let msgpack = nodedb_types::json_msgpack::json_to_msgpack_or_empty(&json);
                    if filters.is_empty()
                        || nodedb_query::ScanFilter::all_match_binary(filters, &msgpack)
                            .map_err(|_| NodeDbError::division_by_zero())?
                    {
                        matching_ids.push(id.clone());
                    }
                }
            }

            let update_fields: Vec<(&str, loro::LoroValue)> = updates
                .iter()
                .map(|(k, v)| (k.as_str(), value_to_loro(v)))
                .collect();

            let mut count = 0u64;
            for id in &matching_ids {
                // A merge: fields the update does not name keep their values.
                crdt.set_fields(collection, id, &update_fields)
                    .map_err(NodeDbError::from)?;
                count += 1;
            }
            drop(crdt);

            // Update text index outside the CRDT lock (text index has its own lock).
            for id in &matching_ids {
                let crdt = self.crdt.lock_or_recover();
                if let Some(loro_val) = crdt.read(collection, id) {
                    let doc = crate::nodedb::convert::loro_value_to_document(id, &loro_val);
                    drop(crdt);
                    self.index_document_text(collection, id, &doc.fields)?;
                    self.index_document_sparse(collection, id, &doc.fields);
                }
            }

            Ok(count)
        })();
        guard.finish(result)
    }

    /// Bulk delete documents matching a predicate.
    ///
    /// Same single-lock pattern as `bulk_update`.
    /// Returns the number of documents deleted.
    pub fn bulk_delete(
        &self,
        collection: &str,
        filters: &[nodedb_query::ScanFilter],
    ) -> NodeDbResult<u64> {
        let guard = self
            .fts_state
            .try_admit_mutation("bulk_delete")
            .map_err(NodeDbError::from)?;
        let result = (|| {
            let mut crdt = self.crdt.lock_or_recover();
            let ids = crdt.list_ids(collection);

            let mut matching_ids = Vec::new();
            for id in &ids {
                if let Some(loro_val) = crdt.read(collection, id) {
                    let doc = crate::nodedb::convert::loro_value_to_document(id, &loro_val);
                    let json = serde_json::to_value(&doc.fields).unwrap_or_default();
                    let msgpack = nodedb_types::json_msgpack::json_to_msgpack_or_empty(&json);
                    if filters.is_empty()
                        || nodedb_query::ScanFilter::all_match_binary(filters, &msgpack)
                            .map_err(|_| NodeDbError::division_by_zero())?
                    {
                        matching_ids.push(id.clone());
                    }
                }
            }

            let mut count = 0u64;
            for id in &matching_ids {
                crdt.delete(collection, id).map_err(NodeDbError::storage)?;
                count += 1;
            }
            drop(crdt);

            for id in &matching_ids {
                self.remove_document_text(collection, id)?;
                self.remove_document_sparse(collection, id);
            }

            Ok(count)
        })();
        guard.finish(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::nodedb::LockExt;
    use crate::{LiteConfig, NodeDbLite, PagedbStorageMem};
    use nodedb_types::value::Value;
    use std::collections::HashMap;

    #[tokio::test]
    async fn busy_bulk_update_preserves_source_and_checkpoint_trust() {
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
        db.crdt
            .lock_or_recover()
            .upsert("docs", "a", &[("value", loro::LoroValue::I64(1))])
            .unwrap();
        db.fts_state.mark_checkpoint_trusted();
        let permit = db.fts_state.admit_exclusive().await;
        let updates = HashMap::from([("value".into(), Value::Integer(2))]);
        let result = db.bulk_update("docs", &[], &updates);
        assert!(result.is_err());
        assert_eq!(
            db.crdt.lock_or_recover().read_field("docs", "a", "value"),
            Some(loro::LoroValue::I64(1))
        );
        assert!(db.fts_state.checkpoint_trusted());
        drop(permit);
        assert_eq!(db.bulk_update("docs", &[], &updates).unwrap(), 1);
    }
}
