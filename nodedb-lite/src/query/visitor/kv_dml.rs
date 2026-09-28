// SPDX-License-Identifier: Apache-2.0
//! SQL-visitor lowering for `UPDATE` and `DELETE` on a KV collection.
//!
//! Both lower to KV ops, the way Origin's plan conversion lowers them, and
//! run through the physical visitor. The KV write arms there record each
//! write for the sync push to Origin.
//! - `DELETE` is one `KvOp::Delete` over its keys.
//! - `UPDATE ... SET col = <literal>` is one `KvOp::FieldSet` per key, with
//!   `if_present` set, so an absent key is `UPDATE 0`.
//!
//! The keys come from the plan's key list when the WHERE names keys. Any
//! other WHERE, or none, is resolved when the statement runs: every live row
//! is read in its SQL read shape, the WHERE is applied to it, and the
//! matching rows' keys are written. This matches Origin's predicate update
//! and predicate delete.

use nodedb_physical::PhysicalTaskVisitor;
use nodedb_physical::physical_plan::KvOp;
use nodedb_sql::types::SqlValue;
use nodedb_sql::types::filter::Filter;
use nodedb_sql::types_expr::SqlExpr;
use nodedb_types::result::QueryResult;
use nodedb_types::{DatabaseId, QualifiedCollection, RlsWriteCheck, Surrogate};

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::filter_convert::sql_value_to_value;
use crate::query::physical_visitor::LiteDataPlaneVisitor;
use crate::storage::engine::StorageEngine;

use super::adapter::LiteFut;
use super::scan_post::filter_mask;
use crate::query::kv_ops::sql_read::{kv_key_bytes, kv_select_entries};

/// The keys a KV write targets.
enum KvTargets {
    /// The keys the WHERE names.
    Keys(Vec<Vec<u8>>),
    /// The rows the WHERE matches, resolved when the statement runs.
    Matching(Vec<Filter>),
}

impl KvTargets {
    fn of(filters: &[Filter], target_keys: &[SqlValue]) -> Self {
        if target_keys.is_empty() {
            Self::Matching(filters.to_vec())
        } else {
            Self::Keys(target_keys.iter().map(kv_key_bytes).collect())
        }
    }

    /// The target keys of `collection`.
    async fn resolve<S: StorageEngine>(
        self,
        engine: &LiteQueryEngine<S>,
        collection: &str,
    ) -> Result<Vec<Vec<u8>>, LiteError> {
        match self {
            Self::Keys(keys) => Ok(keys),
            Self::Matching(filters) => {
                let (rows, keys) = kv_select_entries(engine, collection).await?;
                let keep = filter_mask(&rows, &filters)?;
                Ok(keys
                    .into_iter()
                    .zip(keep)
                    .filter_map(|(key, keep)| keep.then_some(key))
                    .collect())
            }
        }
    }
}

/// Lower `DELETE` on the KV collection `collection` to `KvOp::Delete`.
pub(super) fn lower_kv_delete<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<LiteFut<'a>, LiteError> {
    let targets = KvTargets::of(filters, target_keys);
    let collection = collection.to_owned();
    Ok(Box::pin(async move {
        let keys = targets.resolve(engine, &collection).await?;
        if keys.is_empty() {
            return Ok(QueryResult {
                columns: Vec::new(),
                rows: Vec::new(),
                rows_affected: 0,
                command: Some("DELETE".into()),
            });
        }
        let op = KvOp::Delete {
            collection: qualified(&collection),
            keys,
            // Lite has no RLS policy engine: no write policy applies.
            rls_write_check: RlsWriteCheck::NoPolicyApplies,
            returning: None,
            rls_filters: Vec::new(),
            provenance: None,
        };
        let mut phys = LiteDataPlaneVisitor { engine };
        phys.kv(&op)?.await
    }))
}

/// Lower `UPDATE` on the KV collection `collection` to one
/// `KvOp::FieldSet` per key.
///
/// Every assignment must be a literal, as on Origin.
pub(super) fn lower_kv_update<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    assignments: &[(String, SqlExpr)],
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<LiteFut<'a>, LiteError> {
    let targets = KvTargets::of(filters, target_keys);
    let mut updates: Vec<(String, Vec<u8>)> = Vec::with_capacity(assignments.len());
    for (field, expr) in assignments {
        let SqlExpr::Literal(literal) = expr else {
            return Err(LiteError::BadRequest {
                detail: format!(
                    "UPDATE with non-literal RHS on KV collection '{collection}' \
                     (field '{field}') is not supported; use a literal value"
                ),
            });
        };
        let value = sql_value_to_value(literal)?;
        // Standard MessagePack, the encoding a field set decodes, as on Origin.
        let bytes =
            nodedb_types::value_to_msgpack(&value).map_err(|e| LiteError::Serialization {
                detail: format!("encode UPDATE literal for field '{field}': {e}"),
            })?;
        updates.push((field.clone(), bytes));
    }
    let collection = collection.to_owned();
    Ok(Box::pin(async move {
        let keys = targets.resolve(engine, &collection).await?;
        let mut affected = 0;
        for key in keys {
            let op = KvOp::FieldSet {
                collection: qualified(&collection),
                key,
                updates: updates.clone(),
                surrogate: Surrogate::ZERO,
                // SQL UPDATE: an absent key is `UPDATE 0`, never a create.
                if_present: true,
                // Lite has no RLS policy engine: no write policy applies.
                rls_write_check: RlsWriteCheck::NoPolicyApplies,
                returning: None,
                rls_filters: Vec::new(),
            };
            let mut phys = LiteDataPlaneVisitor { engine };
            affected += phys.kv(&op)?.await?.rows_affected;
        }
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("UPDATE".into()),
        })
    }))
}

/// Lite holds a bare collection name. `DatabaseId::DEFAULT` keeps it
/// unqualified.
fn qualified(collection: &str) -> QualifiedCollection {
    QualifiedCollection::new(DatabaseId::DEFAULT, collection)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nodedb_client::NodeDb;
    use nodedb_types::value::Value;

    use crate::sync::{PendingKvOp, PendingKvWrite};
    use crate::{NodeDbLite, PagedbStorageMem};

    async fn open_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.expect("storage");
        let db = NodeDbLite::open(storage).await.expect("open");
        for sql in [
            "CREATE COLLECTION items (key TEXT PRIMARY KEY, n INT, label TEXT) \
             WITH (engine='kv')",
            "INSERT INTO items (key, n, label) VALUES ('i1', 1, 'one')",
            "INSERT INTO items (key, n, label) VALUES ('i2', 2, 'two')",
            "INSERT INTO items (key, n, label) VALUES ('i3', 3, 'three')",
        ] {
            db.execute_sql(sql, &[])
                .await
                .unwrap_or_else(|e| panic!("{sql}: {e}"));
        }
        db
    }

    async fn queued(db: &NodeDbLite<PagedbStorageMem>) -> Vec<PendingKvWrite> {
        db.kv_outbound
            .as_ref()
            .expect("sync on opens the KV queue")
            .drain(usize::MAX)
            .await
            .expect("drain")
            .into_iter()
            .map(|(_, write)| write)
            .collect()
    }

    async fn rows(db: &NodeDbLite<PagedbStorageMem>, sql: &str) -> Vec<Vec<Value>> {
        db.execute_sql(sql, &[])
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .rows
    }

    #[tokio::test]
    async fn a_predicate_update_changes_and_syncs_only_the_matching_rows() {
        let db = open_db().await;
        let before = queued(&db).await.len();

        let r = db
            .execute_sql("UPDATE items SET label = 'big' WHERE n >= 2", &[])
            .await
            .expect("update");
        assert_eq!(r.rows_affected, 2);

        assert_eq!(
            rows(&db, "SELECT key, label FROM items ORDER BY key").await,
            vec![
                vec![Value::String("i1".into()), Value::String("one".into())],
                vec![Value::String("i2".into()), Value::String("big".into())],
                vec![Value::String("i3".into()), Value::String("big".into())],
            ]
        );
        let pushed: Vec<Vec<u8>> = queued(&db).await[before..]
            .iter()
            .map(|w| {
                assert!(matches!(w.op, PendingKvOp::Put { .. }));
                w.key.clone()
            })
            .collect();
        assert_eq!(pushed, vec![b"i2".to_vec(), b"i3".to_vec()]);
    }

    #[tokio::test]
    async fn a_predicate_delete_removes_and_syncs_only_the_matching_rows() {
        let db = open_db().await;
        let before = queued(&db).await.len();

        let r = db
            .execute_sql("DELETE FROM items WHERE label = 'two'", &[])
            .await
            .expect("delete");
        assert_eq!(r.rows_affected, 1);

        assert_eq!(
            rows(&db, "SELECT key FROM items ORDER BY key").await,
            vec![
                vec![Value::String("i1".into())],
                vec![Value::String("i3".into())],
            ]
        );
        let pushed = queued(&db).await;
        assert_eq!(pushed.len(), before + 1);
        assert_eq!(pushed[before].key, b"i2".to_vec());
        assert_eq!(pushed[before].op, PendingKvOp::Delete);
    }

    #[tokio::test]
    async fn a_predicate_matching_nothing_writes_nothing() {
        let db = open_db().await;
        let before = queued(&db).await.len();
        let r = db
            .execute_sql("DELETE FROM items WHERE n > 100", &[])
            .await
            .expect("delete");
        assert_eq!(r.rows_affected, 0);
        assert_eq!(queued(&db).await.len(), before);
    }
}
