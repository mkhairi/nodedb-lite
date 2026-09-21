// SPDX-License-Identifier: Apache-2.0
//! Write operations for the columnar engine physical visitor.

use nodedb_physical::physical_plan::columnar::ColumnarInsertIntent;
use nodedb_query::scan_filter::ScanFilter;
use nodedb_types::columnar::ColumnarSchema;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::reads::row_to_object;

/// Parameters for a columnar insert operation.
pub struct InsertParams<'a> {
    pub payload: &'a [u8],
    pub format: &'a str,
    pub intent: ColumnarInsertIntent,
    pub on_conflict_updates: &'a [(
        String,
        nodedb_physical::physical_plan::document::types::UpdateValue,
    )],
    pub surrogates: &'a [nodedb_types::Surrogate],
    pub schema_bytes: &'a [u8],
}

/// Insert rows into a columnar collection.
///
/// Decodes the payload per `format` ("json", "msgpack", "ilp"), respects
/// `intent` (Insert / InsertIfAbsent / Put), and assigns surrogates from the
/// provided list falling back to 0 when the list is shorter than the row count.
///
/// Returns the `QueryResult` together with the column-ordered rows that were
/// actually written to the memtable. The caller is responsible for calling
/// `engine.columnar.enqueue_outbound` from an async context to durably queue
/// those rows for replication to Origin.
pub async fn insert<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    params: InsertParams<'_>,
) -> Result<(QueryResult, Vec<Vec<Value>>), LiteError> {
    let InsertParams {
        payload,
        format,
        intent,
        on_conflict_updates,
        surrogates,
        schema_bytes,
    } = params;
    let schema = engine
        .columnar
        .schema(collection)
        .ok_or(LiteError::BadRequest {
            detail: format!("columnar collection '{collection}' does not exist"),
        })?;

    // If caller supplied a schema override, decode it and use it for column ordering.
    let effective_schema: ColumnarSchema = if !schema_bytes.is_empty() {
        zerompk::from_msgpack(schema_bytes).unwrap_or(schema)
    } else {
        schema
    };

    let col_names: Vec<String> = effective_schema
        .columns
        .iter()
        .map(|c| c.name.clone())
        .collect();

    // The incoming row is ordered by `effective_schema`, so its primary key
    // sits wherever that schema puts it. Column 0 is only the common case,
    // and `pk_exists`, `find_row` and `columnar.delete` all resolve the key
    // against the stored schema, so a key elsewhere compared the wrong value.
    let pk_idx = effective_schema
        .columns
        .iter()
        .position(|c| c.primary_key)
        .unwrap_or(0);

    let rows = decode_payload(payload, format, &col_names)?;

    let mut affected: u64 = 0;
    let mut inserted_rows: Vec<Vec<Value>> = Vec::new();

    for (row_idx, row_values) in rows.into_iter().enumerate() {
        let _surrogate = surrogates
            .get(row_idx)
            .copied()
            .unwrap_or(nodedb_types::Surrogate(0));

        match intent {
            ColumnarInsertIntent::Insert => {
                engine.columnar.insert(collection, &row_values)?;
                inserted_rows.push(row_values);
                affected += 1;
            }
            ColumnarInsertIntent::InsertUnique => {
                // PRIMARY KEY on a natural-key column: a duplicate refuses the
                // row rather than tombstoning the prior one the way `Insert`
                // does.
                if let Some(pk) = row_values.get(pk_idx)
                    && pk_exists(engine, collection, pk).await?
                {
                    return Err(LiteError::ConstraintViolation {
                        collection: collection.to_string(),
                        constraint: "unique".into(),
                        detail: format!(
                            "duplicate key value violates the primary key on '{collection}'"
                        ),
                    });
                }
                engine.columnar.insert(collection, &row_values)?;
                inserted_rows.push(row_values);
                affected += 1;
            }
            ColumnarInsertIntent::InsertIfAbsent => {
                // Skip if the key is already present.
                if let Some(pk) = row_values.get(pk_idx)
                    && pk_exists(engine, collection, pk).await?
                {
                    continue;
                }
                engine.columnar.insert(collection, &row_values)?;
                inserted_rows.push(row_values);
                affected += 1;
            }
            ColumnarInsertIntent::Put => {
                if on_conflict_updates.is_empty() {
                    // Plain upsert: delete-then-insert (whole-row overwrite).
                    if let Some(pk) = row_values.get(pk_idx) {
                        let _ = engine.columnar.delete(collection, pk);
                    }
                    engine.columnar.insert(collection, &row_values)?;
                    inserted_rows.push(row_values);
                    affected += 1;
                } else {
                    // Merge: read existing row, apply conflict updates, write merged.
                    let merged = if let Some(pk) = row_values.get(pk_idx) {
                        if pk_exists(engine, collection, pk).await? {
                            let existing = find_row(engine, collection, pk).await?;
                            let incoming_obj = row_to_object(&col_names, &row_values);
                            apply_conflict_updates(
                                existing,
                                &incoming_obj,
                                on_conflict_updates,
                                &col_names,
                            )?
                        } else {
                            row_values.clone()
                        }
                    } else {
                        row_values.clone()
                    };
                    if let Some(pk) = merged.get(pk_idx) {
                        let _ = engine.columnar.delete(collection, pk);
                    }
                    engine.columnar.insert(collection, &merged)?;
                    inserted_rows.push(merged);
                    affected += 1;
                }
            }
        }
    }

    Ok((
        QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
        },
        inserted_rows,
    ))
}

/// Update rows matching filter predicates.
pub async fn update<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    filters_bytes: &[u8],
    updates: &[(String, Vec<u8>)],
) -> Result<QueryResult, LiteError> {
    let schema = engine
        .columnar
        .schema(collection)
        .ok_or(LiteError::BadRequest {
            detail: format!("columnar collection '{collection}' does not exist"),
        })?;

    let col_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let pk_idx = schema
        .columns
        .iter()
        .position(|c| c.primary_key)
        .unwrap_or(0);

    let filters: Vec<ScanFilter> = if filters_bytes.is_empty() {
        Vec::new()
    } else {
        zerompk::from_msgpack(filters_bytes).map_err(|e| LiteError::Serialization {
            detail: format!("decode update filters: {e}"),
        })?
    };

    // Parse update value bytes (each value is msgpack-encoded).
    let parsed_updates: Vec<(String, Value)> = updates
        .iter()
        .map(|(field, bytes)| {
            let v: Value = zerompk::from_msgpack(bytes).unwrap_or(Value::Null);
            (field.clone(), v)
        })
        .collect();

    // Read current rows, apply filters, build modified rows.
    let all_rows = engine.columnar.list_rows(collection).await?;

    let mut affected: u64 = 0;

    for row in all_rows {
        let doc = row_to_object(&col_names, &row);
        let matches = ScanFilter::all_match_value(&filters, &doc)?;
        if !matches {
            continue;
        }

        let pk = row.get(pk_idx).cloned().unwrap_or(Value::Null);

        // Build new_values: copy current row then apply updates.
        let mut new_values = row.clone();
        for (field, new_val) in &parsed_updates {
            if let Some(col_idx) = col_names.iter().position(|n| n == field)
                && col_idx < new_values.len()
            {
                new_values[col_idx] = new_val.clone();
            }
        }

        engine.columnar.update(collection, &pk, &new_values)?;
        affected += 1;
    }

    Ok(QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: affected,
    })
}

/// Delete rows matching filter predicates.
pub async fn delete<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    filters_bytes: &[u8],
) -> Result<QueryResult, LiteError> {
    let schema = engine
        .columnar
        .schema(collection)
        .ok_or(LiteError::BadRequest {
            detail: format!("columnar collection '{collection}' does not exist"),
        })?;

    let col_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let pk_idx = schema
        .columns
        .iter()
        .position(|c| c.primary_key)
        .unwrap_or(0);

    let filters: Vec<ScanFilter> = if filters_bytes.is_empty() {
        Vec::new()
    } else {
        zerompk::from_msgpack(filters_bytes).map_err(|e| LiteError::Serialization {
            detail: format!("decode delete filters: {e}"),
        })?
    };

    let all_rows = engine.columnar.list_rows(collection).await?;

    let mut pks_to_delete: Vec<Value> = Vec::new();
    for row in all_rows {
        let doc = row_to_object(&col_names, &row);
        let matches = filters.is_empty() || ScanFilter::all_match_value(&filters, &doc)?;
        if matches {
            pks_to_delete.push(row.get(pk_idx).cloned().unwrap_or(Value::Null));
        }
    }

    let mut affected: u64 = 0;
    for pk in pks_to_delete {
        if engine.columnar.delete(collection, &pk)? {
            affected += 1;
        }
    }

    Ok(QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: affected,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Decode the insert payload per format into a list of column-ordered rows.
fn decode_payload(
    payload: &[u8],
    format: &str,
    col_names: &[String],
) -> Result<Vec<Vec<Value>>, LiteError> {
    match format {
        "json" => {
            let arr: serde_json::Value =
                sonic_rs::from_slice(payload).map_err(|e| LiteError::Serialization {
                    detail: format!("json payload decode: {e}"),
                })?;
            match arr {
                serde_json::Value::Array(objects) => objects
                    .into_iter()
                    .map(|obj| json_object_to_row(obj, col_names))
                    .collect(),
                serde_json::Value::Object(_) => Ok(vec![json_object_to_row(arr, col_names)?]),
                _ => Err(LiteError::Serialization {
                    detail: "json payload must be an object or array of objects".into(),
                }),
            }
        }
        "msgpack" => {
            // Try array of rows first, then single row.
            let top: Value =
                zerompk::from_msgpack(payload).map_err(|e| LiteError::Serialization {
                    detail: format!("msgpack payload decode: {e}"),
                })?;
            match top {
                Value::Array(items) => items
                    .into_iter()
                    .map(|v| value_object_to_row(v, col_names))
                    .collect(),
                obj @ Value::Object(_) => Ok(vec![value_object_to_row(obj, col_names)?]),
                _ => Err(LiteError::Serialization {
                    detail: "msgpack payload must be an object or array of objects".into(),
                }),
            }
        }
        "ilp" => parse_ilp(payload, col_names),
        other => Err(LiteError::BadRequest {
            detail: format!("unknown columnar insert format '{other}'; expected json/msgpack/ilp"),
        }),
    }
}

/// Minimal InfluxDB Line Protocol parser.
///
/// Grammar: `measurement[,tag=val]* field=val[,field=val]* [timestamp]`
/// Produces one row per non-empty, non-comment line. Column names that are
/// not in `col_names` are silently skipped; absent columns default to Null.
fn parse_ilp(payload: &[u8], col_names: &[String]) -> Result<Vec<Vec<Value>>, LiteError> {
    let text = std::str::from_utf8(payload).map_err(|e| LiteError::Serialization {
        detail: format!("ILP payload is not valid UTF-8: {e}"),
    })?;

    let mut rows: Vec<Vec<Value>> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Split on the first unescaped space to separate the key+tags from fields.
        let (key_part, rest) = split_ilp_space(line).ok_or_else(|| LiteError::Serialization {
            detail: format!("ILP line missing field set: {line}"),
        })?;

        // Split timestamp off the end of rest (optional trailing integer after space).
        let (fields_part, _timestamp) = split_ilp_space(rest)
            .map(|(f, t)| (f, Some(t)))
            .unwrap_or((rest, None));

        // Parse measurement and tags from key_part.
        let mut pairs: std::collections::HashMap<String, Value> = std::collections::HashMap::new();

        // key_part: measurement,tag=val,tag=val
        let mut key_iter = key_part.splitn(2, ',');
        let _measurement = key_iter.next().unwrap_or("");
        if let Some(tags) = key_iter.next() {
            for kv in tags.split(',') {
                if let Some((k, v)) = kv.split_once('=') {
                    pairs.insert(k.to_string(), Value::String(v.to_string()));
                }
            }
        }

        // Parse fields.
        for kv in fields_part.split(',') {
            if let Some((k, v)) = kv.split_once('=') {
                let val = parse_ilp_field_value(v);
                pairs.insert(k.to_string(), val);
            }
        }

        // Build column-ordered row.
        let row: Vec<Value> = col_names
            .iter()
            .map(|name| pairs.get(name).cloned().unwrap_or(Value::Null))
            .collect();
        rows.push(row);
    }

    Ok(rows)
}

/// Split an ILP line on the first unescaped space.
fn split_ilp_space(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == b' ' {
            return Some((&s[..i], s[i + 1..].trim_start()));
        }
        i += 1;
    }
    None
}

/// Parse an ILP field value string to a `Value`.
fn parse_ilp_field_value(v: &str) -> Value {
    // Integer suffix: 12345i
    if let Some(stripped) = v.strip_suffix('i')
        && let Ok(n) = stripped.parse::<i64>()
    {
        return Value::Integer(n);
    }
    // Boolean
    match v {
        "true" | "True" | "TRUE" | "t" | "T" => return Value::Bool(true),
        "false" | "False" | "FALSE" | "f" | "F" => return Value::Bool(false),
        _ => {}
    }
    // Quoted string
    if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
        return Value::String(v[1..v.len() - 1].replace("\\\"", "\""));
    }
    // Float
    if let Ok(f) = v.parse::<f64>() {
        return Value::Float(f);
    }
    // Fall back to string
    Value::String(v.to_string())
}

/// Convert a serde_json object to a column-ordered row.
fn json_object_to_row(
    obj: serde_json::Value,
    col_names: &[String],
) -> Result<Vec<Value>, LiteError> {
    match obj {
        serde_json::Value::Object(map) => {
            let row: Vec<Value> = col_names
                .iter()
                .map(|name| map.get(name).map(json_value_to_ndb).unwrap_or(Value::Null))
                .collect();
            Ok(row)
        }
        _ => Err(LiteError::Serialization {
            detail: "each element in JSON array must be an object".into(),
        }),
    }
}

fn json_value_to_ndb(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::Array(arr.iter().map(json_value_to_ndb).collect()),
        serde_json::Value::Object(map) => {
            let m: std::collections::HashMap<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_value_to_ndb(v)))
                .collect();
            Value::Object(m)
        }
    }
}

/// Convert a Value::Object to a column-ordered row.
fn value_object_to_row(obj: Value, col_names: &[String]) -> Result<Vec<Value>, LiteError> {
    match obj {
        Value::Object(map) => {
            let row: Vec<Value> = col_names
                .iter()
                .map(|name| map.get(name).cloned().unwrap_or(Value::Null))
                .collect();
            Ok(row)
        }
        _ => Err(LiteError::Serialization {
            detail: "each msgpack element must be an object map".into(),
        }),
    }
}

/// Check whether a PK value currently exists in the columnar collection.
///
/// Scans `list_rows`. Lite's columnar engine is in-memory so this is cheap.
async fn pk_exists<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    pk: &Value,
) -> Result<bool, LiteError> {
    let schema = match engine.columnar.schema(collection) {
        Some(s) => s,
        None => return Ok(false),
    };
    let pk_idx = schema
        .columns
        .iter()
        .position(|c| c.primary_key)
        .unwrap_or(0);

    let rows = engine.columnar.list_rows(collection).await?;

    Ok(rows
        .iter()
        .any(|row| row.get(pk_idx).map(|v| v == pk).unwrap_or(false)))
}

/// Find a specific row by PK.
async fn find_row<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    pk: &Value,
) -> Result<Vec<Value>, LiteError> {
    let schema = engine
        .columnar
        .schema(collection)
        .ok_or(LiteError::BadRequest {
            detail: format!("columnar collection '{collection}' does not exist"),
        })?;
    let pk_idx = schema
        .columns
        .iter()
        .position(|c| c.primary_key)
        .unwrap_or(0);

    let rows = engine.columnar.list_rows(collection).await?;

    rows.into_iter()
        .find(|row| row.get(pk_idx).map(|v| v == pk).unwrap_or(false))
        .ok_or(LiteError::BadRequest {
            detail: format!("row with pk {pk:?} not found in '{collection}'"),
        })
}

type UpdateValue = nodedb_physical::physical_plan::document::types::UpdateValue;

/// Apply `ON CONFLICT DO UPDATE` assignments to an existing row.
fn apply_conflict_updates(
    mut existing: Vec<Value>,
    incoming: &Value,
    updates: &[(String, UpdateValue)],
    col_names: &[String],
) -> Result<Vec<Value>, LiteError> {
    for (field, update_val) in updates {
        let new_val = match update_val {
            UpdateValue::Literal(bytes) => {
                zerompk::from_msgpack::<Value>(bytes).unwrap_or(Value::Null)
            }
            UpdateValue::Expr(expr) => {
                // Evaluate expr against the existing row document. The expr
                // may reference EXCLUDED columns via the incoming object; we
                // use the incoming value for the target field as a safe fallback
                // when the expr cannot be fully resolved.
                let doc = incoming.clone();
                let evaled = expr.eval(&doc)?;
                if matches!(evaled, Value::Null) {
                    incoming.get(field).cloned().unwrap_or(Value::Null)
                } else {
                    evaled
                }
            }
        };
        if let Some(col_idx) = col_names.iter().position(|n| n == field)
            && col_idx < existing.len()
        {
            existing[col_idx] = new_val;
        }
    }
    Ok(existing)
}

#[cfg(test)]
mod tests {
    use nodedb_client::NodeDb;

    use super::*;

    #[test]
    fn parse_ilp_basic() {
        let col_names = vec!["host".to_string(), "cpu".to_string(), "ts".to_string()];
        let ilp = b"cpu,host=server01 cpu=0.64 1465839830100400200";
        let rows = parse_ilp(ilp, &col_names).unwrap();
        assert_eq!(rows.len(), 1);
        // host tag
        assert_eq!(rows[0][0], Value::String("server01".into()));
        // cpu field float
        assert_eq!(rows[0][1], Value::Float(0.64));
    }

    #[test]
    fn parse_ilp_integer_field() {
        let col_names = vec!["count".to_string()];
        let ilp = b"events count=42i";
        let rows = parse_ilp(ilp, &col_names).unwrap();
        assert_eq!(rows[0][0], Value::Integer(42));
    }

    #[test]
    fn parse_ilp_bool_field() {
        let col_names = vec!["active".to_string()];
        let ilp = b"status active=true";
        let rows = parse_ilp(ilp, &col_names).unwrap();
        assert_eq!(rows[0][0], Value::Bool(true));
    }

    #[test]
    fn parse_ilp_comment_and_empty_lines() {
        let col_names = vec!["v".to_string()];
        let ilp = b"# comment\n\nevents v=1i";
        let rows = parse_ilp(ilp, &col_names).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], Value::Integer(1));
    }

    #[test]
    fn json_object_to_row_basic() {
        let obj = serde_json::json!({"a": 1, "b": "hello"});
        let cols = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let row = json_object_to_row(obj, &cols).unwrap();
        assert_eq!(row[0], Value::Integer(1));
        assert_eq!(row[1], Value::String("hello".into()));
        assert_eq!(row[2], Value::Null);
    }

    async fn make_db() -> std::sync::Arc<crate::NodeDbLite<crate::PagedbStorageMem>> {
        let storage = crate::PagedbStorageMem::open_in_memory().await.unwrap();
        crate::NodeDbLite::open(storage).await.unwrap()
    }

    fn unique_params(payload: &[u8]) -> InsertParams<'_> {
        InsertParams {
            payload,
            format: "json",
            intent: ColumnarInsertIntent::InsertUnique,
            on_conflict_updates: &[],
            surrogates: &[],
            schema_bytes: &[],
        }
    }

    async fn insert_unique(
        db: &std::sync::Arc<crate::NodeDbLite<crate::PagedbStorageMem>>,
        collection: &str,
        payload: &[u8],
    ) -> Result<(), LiteError> {
        insert(&db.query_engine, collection, unique_params(payload))
            .await
            .map(|_| ())
    }

    /// A declared PRIMARY KEY means uniqueness: `InsertUnique` refuses a
    /// duplicate natural key rather than tombstoning the prior row the way
    /// `Insert` does.
    #[tokio::test(flavor = "multi_thread")]
    async fn insert_unique_refuses_a_duplicate_primary_key() {
        let db = make_db().await;
        db.execute_sql(
            "CREATE COLLECTION iu_dup (id TEXT NOT NULL PRIMARY KEY, n INTEGER) \
             WITH storage = 'columnar'",
            &[],
        )
        .await
        .unwrap();

        insert_unique(&db, "iu_dup", br#"[{"id":"a","n":1}]"#)
            .await
            .expect("first insert");

        let err = insert_unique(&db, "iu_dup", br#"[{"id":"a","n":2}]"#)
            .await
            .expect_err("a duplicate primary key must be refused");
        assert!(
            matches!(
                err,
                LiteError::ConstraintViolation {
                    ref constraint,
                    ref detail,
                    ..
                } if constraint == "unique" && detail.contains("duplicate key")
            ),
            "unexpected error: {err}"
        );

        let rows = db.query_engine.columnar.list_rows("iu_dup").await.unwrap();
        assert_eq!(rows.len(), 1, "the refused row must not be written");
        assert_eq!(rows[0][1], Value::Integer(1), "the prior row survives");
    }

    /// The primary key is read from the schema, not assumed to be column 0.
    /// With the key declared second, resolving it positionally compared the
    /// wrong column and let a duplicate through.
    #[tokio::test(flavor = "multi_thread")]
    async fn insert_unique_finds_a_primary_key_that_is_not_column_zero() {
        let db = make_db().await;
        db.execute_sql(
            "CREATE COLLECTION iu_late_pk (n INTEGER, id TEXT NOT NULL PRIMARY KEY) \
             WITH storage = 'columnar'",
            &[],
        )
        .await
        .unwrap();

        insert_unique(&db, "iu_late_pk", br#"[{"n":1,"id":"a"}]"#)
            .await
            .expect("first insert");

        // Same key, different leading column: positional resolution would
        // compare `n` and see 1 vs 2, admitting the duplicate.
        let err = insert_unique(&db, "iu_late_pk", br#"[{"n":2,"id":"a"}]"#)
            .await
            .expect_err("a duplicate primary key must be refused wherever it sits");
        assert!(
            matches!(
                err,
                LiteError::ConstraintViolation {
                    ref constraint,
                    ref detail,
                    ..
                } if constraint == "unique" && detail.contains("duplicate key")
            ),
            "unexpected error: {err}"
        );

        let rows = db
            .query_engine
            .columnar
            .list_rows("iu_late_pk")
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "the refused row must not be written");
    }

    /// `InsertIfAbsent` shares the same `pk_idx`, and skips rather than
    /// refusing. With the key declared second, positional resolution made it
    /// re-insert a row it should have left alone.
    #[tokio::test(flavor = "multi_thread")]
    async fn insert_if_absent_finds_a_primary_key_that_is_not_column_zero() {
        let db = make_db().await;
        db.execute_sql(
            "CREATE COLLECTION iia_late_pk (n INTEGER, id TEXT NOT NULL PRIMARY KEY) \
             WITH storage = 'columnar'",
            &[],
        )
        .await
        .unwrap();

        let params = |payload: &'static [u8]| InsertParams {
            payload,
            format: "json",
            intent: ColumnarInsertIntent::InsertIfAbsent,
            on_conflict_updates: &[],
            surrogates: &[],
            schema_bytes: &[],
        };
        let engine = &db.query_engine;
        insert(engine, "iia_late_pk", params(br#"[{"n":1,"id":"a"}]"#))
            .await
            .expect("first insert");
        insert(engine, "iia_late_pk", params(br#"[{"n":2,"id":"a"}]"#))
            .await
            .expect("a duplicate is skipped, not an error");

        let rows = engine.columnar.list_rows("iia_late_pk").await.unwrap();
        assert_eq!(
            rows.len(),
            1,
            "the duplicate must be skipped wherever the key sits"
        );
        assert_eq!(rows[0][0], Value::Integer(1), "the prior row survives");
    }

    /// The refusal is keyed on the PK, not on the insert itself: a fresh key
    /// still goes in.
    #[tokio::test(flavor = "multi_thread")]
    async fn insert_unique_admits_a_fresh_primary_key() {
        let db = make_db().await;
        db.execute_sql(
            "CREATE COLLECTION iu_fresh (id TEXT NOT NULL PRIMARY KEY, n INTEGER) \
             WITH storage = 'columnar'",
            &[],
        )
        .await
        .unwrap();

        insert_unique(&db, "iu_fresh", br#"[{"id":"a","n":1}]"#)
            .await
            .expect("first insert");
        insert_unique(&db, "iu_fresh", br#"[{"id":"b","n":2}]"#)
            .await
            .expect("second insert");

        let rows = db
            .query_engine
            .columnar
            .list_rows("iu_fresh")
            .await
            .unwrap();
        assert_eq!(rows.len(), 2, "distinct keys both insert");
    }
}
