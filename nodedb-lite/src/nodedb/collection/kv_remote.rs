// SPDX-License-Identifier: Apache-2.0
//! Apply a KV row that Origin pushes to this replica.
//!
//! Origin sends a KV write as the `{key, value…}` row that every KV read
//! returns. The row decodes to the body a local write of the same row
//! stores (`query::kv_ops::body`).
//!
//! The body goes through the KV write buffer, so it orders after every
//! buffered local write. The buffer then flushes, so the row is durable and
//! visible to SQL reads once the apply returns. The write skips the CRDT
//! delta log: the row came from Origin, so pushing it back would echo it.
//! Origin's row carries no TTL, so the entry stores no expiry.

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use super::super::{LockExt, NodeDbLite};
use super::ddl::CollectionMeta;
use super::kv::{encode_value, kv_key};
use crate::storage::engine::{StorageEngine, WriteOp};

impl<S: StorageEngine> NodeDbLite<S> {
    /// Whether `name` is a KV collection in this replica's catalog.
    ///
    /// A name with no persisted collection meta is not a KV collection.
    pub(crate) async fn is_kv_collection(&self, name: &str) -> NodeDbResult<bool> {
        let key = format!("collection:{name}");
        let Some(bytes) = self
            .storage
            .get(Namespace::Meta, key.as_bytes())
            .await
            .map_err(NodeDbError::storage)?
        else {
            return Ok(false);
        };
        let meta: CollectionMeta = sonic_rs::from_slice(&bytes).map_err(|e| {
            NodeDbError::serialization(
                "json",
                format!("collection meta of '{name}' does not decode: {e}"),
            )
        })?;
        Ok(meta.collection_type == "kv")
    }

    /// Store the KV row Origin pushed for `key`, or remove `key` when
    /// `delete` is set.
    ///
    /// A payload that is not a `{key, value…}` row map is a serialization
    /// error, and nothing is stored.
    pub(crate) async fn kv_apply_remote_row(
        &self,
        collection: &str,
        key: &str,
        row: &[u8],
        delete: bool,
    ) -> NodeDbResult<()> {
        let rkey = kv_key(collection, key.as_bytes());
        let entry = if delete {
            None
        } else {
            let body = crate::query::kv_ops::body::kv_body_from_row(key, row).map_err(|e| {
                NodeDbError::serialization(
                    "msgpack",
                    format!("KV row push for '{collection}': {e}"),
                )
            })?;
            Some(encode_value(0, &body))
        };

        // Scope the guard so it is not live at the await point.
        {
            let mut buf = self.kv_local.write_buf.lock_or_recover();
            buf.overlay.insert(rkey.clone(), entry.clone());
            buf.ops.push(match entry {
                Some(value) => WriteOp::Put {
                    ns: Namespace::Kv,
                    key: rkey.clone(),
                    value,
                },
                None => WriteOp::Delete {
                    ns: Namespace::Kv,
                    key: rkey.clone(),
                },
            });
        }
        self.kv_local.cache.lock_or_recover().pop(&rkey);

        self.kv_flush_inner().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nodedb_query::msgpack_scan::kv_row_msgpack;
    use nodedb_types::KvConfig;
    use nodedb_types::columnar::{ColumnDef, ColumnType, StrictSchema};

    use crate::storage::pagedb_storage::PagedbStorageMem;

    use super::*;

    async fn open_db_with_kv(collection: &str) -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory()
            .await
            .expect("open in-memory storage");
        let db = NodeDbLite::open(storage).await.expect("open NodeDbLite");
        let schema = StrictSchema::new(vec![
            ColumnDef::required("key", ColumnType::String).with_primary_key(),
            ColumnDef::nullable("value", ColumnType::String),
        ])
        .expect("kv schema");
        let config = KvConfig {
            schema,
            ttl: None,
            capacity_hint: 0,
            inline_threshold: nodedb_types::KV_DEFAULT_INLINE_THRESHOLD,
        };
        db.create_kv_collection(collection, &config)
            .await
            .expect("create kv collection");
        db
    }

    #[tokio::test]
    async fn a_pushed_kv_row_reads_back_as_the_same_value_a_local_write_stores() {
        let db = open_db_with_kv("cfg").await;
        db.kv_put("cfg", "local", b"v1").await.expect("local put");

        let row = kv_row_msgpack("remote", b"v1");
        db.apply_remote_row("cfg", "remote", &row, false)
            .await
            .expect("apply row");

        let local = db.kv_get("cfg", "local").await.expect("get local");
        let remote = db.kv_get("cfg", "remote").await.expect("get remote");
        assert_eq!(remote.as_deref(), Some(b"v1".as_slice()));
        assert_eq!(remote, local);
    }

    #[tokio::test]
    async fn a_pushed_kv_row_is_durable_without_a_flush() {
        let db = open_db_with_kv("cfg").await;
        let row = kv_row_msgpack("k", b"v1");
        db.apply_remote_row("cfg", "k", &row, false)
            .await
            .expect("apply row");

        let stored = db
            .storage
            .get(Namespace::Kv, &kv_key("cfg", b"k"))
            .await
            .expect("read storage");
        assert_eq!(stored, Some(encode_value(0, b"v1")));
    }

    #[tokio::test]
    async fn a_pushed_kv_delete_removes_the_key() {
        let db = open_db_with_kv("cfg").await;
        db.kv_put("cfg", "k", b"v1").await.expect("local put");

        db.apply_remote_row("cfg", "k", &[], true)
            .await
            .expect("apply delete");

        assert_eq!(db.kv_get("cfg", "k").await.expect("get"), None);
    }

    #[tokio::test]
    async fn a_pushed_kv_payload_that_is_not_a_row_map_is_refused() {
        let db = open_db_with_kv("cfg").await;

        let err = db
            .apply_remote_row("cfg", "k", b"v1", false)
            .await
            .expect_err("raw bytes are not a row");
        assert!(err.message().contains("KV row push for 'cfg'"), "{err}");
        assert_eq!(db.kv_get("cfg", "k").await.expect("get"), None);
    }

    #[tokio::test]
    async fn a_synced_typed_row_takes_a_local_increment() {
        let db = open_db_with_kv("cfg").await;
        let mut columns = std::collections::HashMap::new();
        columns.insert("n".to_string(), nodedb_types::value::Value::Integer(5));
        columns.insert(
            "label".to_string(),
            nodedb_types::value::Value::String("x".into()),
        );
        let origin_body = nodedb_query::msgpack_scan::row_to_kv_body(
            &nodedb_types::value::Value::Object(columns),
            nodedb_query::msgpack_scan::KvBodyShape::Map,
        )
        .expect("origin body");
        db.apply_remote_row("cfg", "t", &kv_row_msgpack("t", &origin_body), false)
            .await
            .expect("apply row");

        assert_eq!(db.kv_increment("cfg", "t", 2).await.expect("incr"), 7);
        let body = db.kv_get("cfg", "t").await.expect("get").expect("row");
        let row = crate::query::kv_ops::body::decode_kv_map(&body)
            .expect("decode")
            .expect("map body");
        assert_eq!(row.get("n"), Some(&nodedb_types::value::Value::Integer(7)));
        assert_eq!(
            row.get("label"),
            Some(&nodedb_types::value::Value::String("x".into()))
        );
    }

    #[tokio::test]
    async fn a_pushed_row_for_a_kv_collection_stays_out_of_the_crdt_log() {
        let db = open_db_with_kv("cfg").await;
        let row = kv_row_msgpack("k", b"v1");
        db.apply_remote_row("cfg", "k", &row, false)
            .await
            .expect("apply row");

        assert!(db.pending_crdt_deltas().expect("pending").is_empty());
    }
}
