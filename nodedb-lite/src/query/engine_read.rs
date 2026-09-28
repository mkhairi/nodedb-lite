// SPDX-License-Identifier: Apache-2.0
//! SQL reads the planner sends straight to an engine: the full scan behind
//! `SqlPlan::Scan` and the key lookup behind `SqlPlan::PointGet`.
//!
//! Both match every `EngineType`. An engine Lite cannot read this way
//! returns `Unsupported`, never an empty result.

use std::cmp::Ordering;

use nodedb_query::value_ops::compare_values;
use nodedb_sql::types::*;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use super::engine::{LiteQueryEngine, sql_value_to_string, sql_value_to_value};
use super::kv_ops::sql_read::{kv_key_bytes, kv_select_all, kv_select_key};
use crate::error::LiteError;
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> LiteQueryEngine<S> {
    pub(in crate::query) async fn execute_scan(
        &self,
        collection: &str,
        engine: &EngineType,
    ) -> Result<QueryResult, LiteError> {
        match engine {
            EngineType::DocumentSchemaless => {
                // For bitemporal collections the Loro snapshot may lag storage
                // (it is only saved on explicit flush).  Scan DocumentHistory
                // as the authoritative source for the current set of live IDs.
                let is_bt = crate::engine::document::history::ops::is_bitemporal(
                    &*self.storage,
                    collection,
                )
                .await
                .unwrap_or(false);

                if is_bt {
                    let live_docs = crate::engine::document::history::ops::scan_live_documents(
                        &*self.storage,
                        collection,
                    )
                    .await
                    .map_err(|e| LiteError::Query(e.to_string()))?;
                    let mut rows = Vec::with_capacity(live_docs.len());
                    for (id, body) in &live_docs {
                        // Decode the msgpack body to a JSON string for the
                        // document column so post-scan filters can match fields.
                        let doc_str = if body.is_empty() {
                            "{}".to_owned()
                        } else {
                            match nodedb_types::json_msgpack::value_from_msgpack(body) {
                                Ok(nodedb_types::value::Value::Object(fields)) => {
                                    let json_map: serde_json::Map<String, serde_json::Value> =
                                        fields
                                            .into_iter()
                                            .map(|(k, v)| (k, value_to_serde_json(v)))
                                            .collect();
                                    sonic_rs::to_string(&serde_json::Value::Object(json_map))
                                        .unwrap_or_else(|_| "{}".to_owned())
                                }
                                _ => "{}".to_owned(),
                            }
                        };
                        rows.push(vec![Value::String(id.clone()), Value::String(doc_str)]);
                    }
                    return Ok(QueryResult {
                        columns: vec!["id".into(), "document".into()],
                        rows,
                        rows_affected: 0,
                        command: None,
                    });
                }

                let crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
                let ids = crdt.list_ids(collection);
                let mut rows = Vec::with_capacity(ids.len());
                for id in &ids {
                    if let Some(val) = crdt.read(collection, id) {
                        let json = loro_value_to_json(&val);
                        let doc_str = sonic_rs::to_string(&json).unwrap_or_default();
                        rows.push(vec![Value::String(id.clone()), Value::String(doc_str)]);
                    }
                }
                Ok(QueryResult {
                    columns: vec!["id".into(), "document".into()],
                    rows,
                    rows_affected: 0,
                    command: None,
                })
            }
            EngineType::DocumentStrict => {
                let schema =
                    self.strict
                        .schema(collection)
                        .ok_or_else(|| LiteError::BadRequest {
                            detail: format!("strict collection '{collection}' does not exist"),
                        })?;
                let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
                let rows = self.strict.list_rows(collection).await?;
                Ok(QueryResult {
                    columns,
                    rows,
                    rows_affected: 0,
                    command: None,
                })
            }
            // Every columnar-family profile keeps its rows in the columnar
            // engine, which applies the timeseries or spatial profile itself.
            EngineType::Columnar | EngineType::Timeseries | EngineType::Spatial => {
                let schema =
                    self.columnar
                        .schema(collection)
                        .ok_or_else(|| LiteError::BadRequest {
                            detail: format!("columnar collection '{collection}' does not exist"),
                        })?;
                let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
                let rows = self.columnar.list_rows(collection).await?;
                Ok(QueryResult {
                    columns,
                    rows,
                    rows_affected: 0,
                    command: None,
                })
            }
            EngineType::KeyValue => kv_select_all(self, collection).await,
            EngineType::Array => Err(array_refusal("scan")),
        }
    }

    /// Fetch the row whose `key_column` equals `key`.
    pub(in crate::query) async fn execute_point_get(
        &self,
        collection: &str,
        engine: &EngineType,
        key_column: &str,
        key: &SqlValue,
    ) -> Result<QueryResult, LiteError> {
        let key_str = sql_value_to_string(key);
        match engine {
            EngineType::DocumentSchemaless => {
                let crdt = self.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
                match crdt.read(collection, &key_str) {
                    Some(val) => {
                        let json = loro_value_to_json(&val);
                        let doc_str = sonic_rs::to_string(&json).unwrap_or_default();
                        Ok(QueryResult {
                            columns: vec!["id".into(), "document".into()],
                            rows: vec![vec![Value::String(key_str), Value::String(doc_str)]],
                            rows_affected: 0,
                            command: None,
                        })
                    }
                    None => Ok(QueryResult::empty()),
                }
            }
            EngineType::DocumentStrict => {
                let schema =
                    self.strict
                        .schema(collection)
                        .ok_or_else(|| LiteError::BadRequest {
                            detail: format!("strict collection '{collection}' does not exist"),
                        })?;
                let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
                // The PK column type determines how to parse the key string.
                let pk_col = schema
                    .columns
                    .iter()
                    .find(|c| c.primary_key)
                    .ok_or_else(|| LiteError::BadRequest {
                        detail: format!(
                            "strict collection '{collection}' has no primary key column"
                        ),
                    })?;
                let pk_value = parse_pk_value(&key_str, &pk_col.column_type);
                match self.strict.get(collection, &pk_value).await? {
                    Some(values) => Ok(QueryResult {
                        columns,
                        rows: vec![values],
                        rows_affected: 0,
                        command: None,
                    }),
                    None => Ok(QueryResult {
                        columns,
                        rows: Vec::new(),
                        rows_affected: 0,
                        command: None,
                    }),
                }
            }
            EngineType::KeyValue => kv_select_key(self, collection, &kv_key_bytes(key)).await,
            // A columnar row is found by scanning for the key column's value.
            EngineType::Columnar | EngineType::Timeseries | EngineType::Spatial => {
                let mut result = self.execute_scan(collection, engine).await?;
                let column = result
                    .columns
                    .iter()
                    .position(|c| c == key_column)
                    .ok_or_else(|| LiteError::BadRequest {
                        detail: format!(
                            "collection '{collection}' has no key column '{key_column}'"
                        ),
                    })?;
                let wanted = sql_value_to_value(key);
                result.rows.retain(|row| {
                    row.get(column)
                        .is_some_and(|v| compare_values(v, &wanted) == Ordering::Equal)
                });
                Ok(result)
            }
            EngineType::Array => Err(array_refusal("point lookup")),
        }
    }
}

/// The refusal for a table-shaped `what` on an array. Arrays are read with
/// the `NDARRAY_*` table functions.
fn array_refusal(what: &str) -> LiteError {
    LiteError::Unsupported {
        detail: format!(
            "{what} is not supported on an array; read it with NDARRAY_SLICE or \
             NDARRAY_PROJECT"
        ),
    }
}

/// Convert a primary-key string from a SQL literal into the appropriate `Value`
/// variant based on the column's declared type.
pub(super) fn parse_pk_value(
    key_str: &str,
    col_type: &nodedb_types::columnar::ColumnType,
) -> Value {
    use nodedb_types::columnar::ColumnType;
    match col_type {
        ColumnType::Int64 => key_str
            .parse::<i64>()
            .map(Value::Integer)
            .unwrap_or_else(|_| Value::String(key_str.to_string())),
        ColumnType::Uuid => Value::Uuid(key_str.to_string()),
        _ => Value::String(key_str.to_string()),
    }
}

fn value_to_serde_json(v: nodedb_types::value::Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(b),
        Value::Integer(n) => serde_json::json!(n),
        Value::Float(f) => serde_json::json!(f),
        Value::String(s) => serde_json::Value::String(s),
        Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(value_to_serde_json).collect())
        }
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, val) in map {
                out.insert(k, value_to_serde_json(val));
            }
            serde_json::Value::Object(out)
        }
        _ => serde_json::Value::Null,
    }
}

fn loro_value_to_json(v: &loro::LoroValue) -> serde_json::Value {
    match v {
        loro::LoroValue::Null => serde_json::Value::Null,
        loro::LoroValue::Bool(b) => serde_json::Value::Bool(*b),
        loro::LoroValue::I64(n) => serde_json::json!(*n),
        loro::LoroValue::Double(f) => serde_json::json!(*f),
        loro::LoroValue::String(s) => serde_json::Value::String(s.to_string()),
        loro::LoroValue::Map(m) => {
            let mut obj = serde_json::Map::new();
            for (k, val) in m.iter() {
                obj.insert(k.to_string(), loro_value_to_json(val));
            }
            serde_json::Value::Object(obj)
        }
        loro::LoroValue::List(arr) => {
            serde_json::Value::Array(arr.iter().map(loro_value_to_json).collect())
        }
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nodedb_client::NodeDb;

    use super::*;
    use crate::{NodeDbLite, PagedbStorageMem};

    async fn open_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.expect("storage");
        let db = NodeDbLite::open(storage).await.expect("open");
        for sql in [
            "CREATE COLLECTION raw (key TEXT PRIMARY KEY, value TEXT) WITH (engine='kv')",
            "CREATE COLLECTION items (key TEXT PRIMARY KEY, n INT, label TEXT) \
             WITH (engine='kv')",
            "INSERT INTO raw (key, value) VALUES ('a', 'alpha')",
            "INSERT INTO raw (key, value) VALUES ('b', 'beta')",
            "INSERT INTO raw (key, value) VALUES ('c', 'gamma')",
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

    async fn query(db: &NodeDbLite<PagedbStorageMem>, sql: &str) -> QueryResult {
        db.execute_sql(sql, &[])
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    fn text(v: &str) -> Value {
        Value::String(v.to_owned())
    }

    #[tokio::test]
    async fn a_point_get_returns_the_raw_row() {
        let db = open_db().await;
        let r = query(&db, "SELECT * FROM raw WHERE key = 'b'").await;
        assert_eq!(r.columns, vec!["key", "value"]);
        assert_eq!(r.rows, vec![vec![text("b"), text("beta")]]);

        let r = query(&db, "SELECT value FROM raw WHERE key = 'b'").await;
        assert_eq!(r.columns, vec!["value"]);
        assert_eq!(r.rows, vec![vec![text("beta")]]);

        let r = query(&db, "SELECT * FROM raw WHERE key = 'missing'").await;
        assert!(r.rows.is_empty());
    }

    #[tokio::test]
    async fn an_in_list_returns_each_named_key() {
        let db = open_db().await;
        let r = query(
            &db,
            "SELECT key FROM raw WHERE key IN ('a', 'c') ORDER BY key",
        )
        .await;
        assert_eq!(r.rows, vec![vec![text("a")], vec![text("c")]]);
    }

    /// An `IN` list reaches the scan as an expression predicate. It must
    /// filter a document scan too, with an ORDER BY alongside.
    #[tokio::test]
    async fn an_in_list_with_order_by_filters_a_document_scan() {
        let db = open_db().await;
        for sql in [
            "CREATE COLLECTION docs",
            "INSERT INTO docs (id, name) VALUES ('d1', 'a')",
            "INSERT INTO docs (id, name) VALUES ('d2', 'b')",
            "INSERT INTO docs (id, name) VALUES ('d3', 'c')",
        ] {
            query(&db, sql).await;
        }
        let r = query(
            &db,
            "SELECT id FROM docs WHERE name IN ('a', 'c') ORDER BY id",
        )
        .await;
        assert_eq!(r.rows, vec![vec![text("d1")], vec![text("d3")]]);

        let r = query(
            &db,
            "SELECT id FROM docs WHERE name NOT IN ('a', 'c') ORDER BY id",
        )
        .await;
        assert_eq!(r.rows, vec![vec![text("d2")]]);
    }

    /// A `LIKE` inside a larger predicate and an `ARRAY[...]` with a column
    /// element both evaluate per row, as on Origin.
    #[tokio::test]
    async fn like_and_array_inside_expression_predicates() {
        let db = open_db().await;
        query(&db, "CREATE COLLECTION people").await;
        for (id, name, n) in [
            ("a1", "Alice", 1),
            ("a2", "amy", 3),
            ("b1", "bob", 9),
            ("c1", "Carl", 2),
        ] {
            query(
                &db,
                &format!("INSERT INTO people (id, name, n) VALUES ('{id}', '{name}', {n})"),
            )
            .await;
        }
        let ids = |r: QueryResult| -> Vec<Value> {
            r.rows.into_iter().map(|row| row[0].clone()).collect()
        };

        let r = query(
            &db,
            "SELECT id FROM people WHERE lower(name) LIKE 'a%' OR n > 5 ORDER BY id",
        )
        .await;
        assert_eq!(ids(r), vec![text("a1"), text("a2"), text("b1")]);

        let r = query(
            &db,
            "SELECT id FROM people WHERE lower(name) NOT LIKE 'a%' AND n < 5 ORDER BY id",
        )
        .await;
        assert_eq!(ids(r), vec![text("c1")]);

        let r = query(
            &db,
            "SELECT id FROM people WHERE name ILIKE 'A%' OR n > 100 ORDER BY id",
        )
        .await;
        assert_eq!(ids(r), vec![text("a1"), text("a2")]);

        let r = query(
            &db,
            "SELECT id FROM people WHERE array_contains(ARRAY[n, 100], 9) ORDER BY id",
        )
        .await;
        assert_eq!(ids(r), vec![text("b1")]);
    }

    #[tokio::test]
    async fn a_scan_filters_projects_orders_and_limits_typed_rows() {
        let db = open_db().await;
        let r = query(
            &db,
            "SELECT key, n FROM items WHERE n > 1 ORDER BY n DESC LIMIT 1",
        )
        .await;
        assert_eq!(r.columns, vec!["key", "n"]);
        assert_eq!(r.rows, vec![vec![text("i3"), Value::Integer(3)]]);

        let r = query(&db, "SELECT label FROM items ORDER BY key LIMIT 2 OFFSET 1").await;
        assert_eq!(r.rows, vec![vec![text("two")], vec![text("three")]]);
    }

    #[tokio::test]
    async fn a_count_aggregates_the_live_rows() {
        let db = open_db().await;
        let r = query(&db, "SELECT COUNT(*) FROM items WHERE n >= 2").await;
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0][0], Value::Integer(2));
    }

    #[tokio::test]
    async fn an_expired_row_is_not_returned() {
        let db = open_db().await;
        db.kv_put_with_ttl("raw", "short", b"soon gone", 1)
            .await
            .expect("put with ttl");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        let r = query(&db, "SELECT key FROM raw ORDER BY key").await;
        assert_eq!(
            r.rows,
            vec![vec![text("a")], vec![text("b")], vec![text("c")]]
        );
        let r = query(&db, "SELECT * FROM raw WHERE key = 'short'").await;
        assert!(r.rows.is_empty());
    }

    #[tokio::test]
    async fn a_public_api_write_is_visible_to_sql() {
        let db = open_db().await;
        db.kv_put("raw", "d", b"delta").await.expect("put");
        let r = query(&db, "SELECT value FROM raw WHERE key = 'd'").await;
        assert_eq!(r.rows, vec![vec![text("delta")]]);
    }
}
