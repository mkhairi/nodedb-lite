// SPDX-License-Identifier: Apache-2.0
//! A SELECT on a schemaless collection with a Ready field index returns the
//! rows a full scan returns, in the same shape.

use nodedb_client::NodeDb;
use nodedb_sql::types::{SqlPlan, SqlValue};
use nodedb_types::Namespace;
use nodedb_types::document::Document;
use nodedb_types::value::Value;

use crate::storage::engine::StorageEngine;
use crate::{NodeDbLite, PagedbStorageMem};

/// Collection with an index on `scope`.
const INDEXED: &str = "ko";
/// Twin collection with the same documents and no index.
const PLAIN: &str = "ko_plain";

fn doc(id: &str, fields: &[(&str, Value)]) -> Document {
    let mut doc = Document::new(id);
    for (key, value) in fields {
        doc.set(*key, value.clone());
    }
    doc
}

fn obj(id: &str, scope: Option<Value>, kind: &str, hash: &str, latest: bool) -> Document {
    let mut fields = vec![
        ("type", Value::String(kind.into())),
        ("content_hash", Value::String(hash.into())),
        ("is_latest", Value::Bool(latest)),
    ];
    if let Some(scope) = scope {
        fields.push(("scope", scope));
    }
    doc(id, &fields)
}

fn text(s: &str) -> Option<Value> {
    Some(Value::String(s.into()))
}

/// Documents written before `CREATE INDEX`: several scopes, `is_latest` both
/// ways, a missing and a null scope, and scope values whose posting keys
/// collide across types.
fn seed_docs() -> Vec<Document> {
    vec![
        obj("k01", text("a"), "note", "h1", true),
        obj("k02", text("a"), "note", "h1", false),
        obj("k03", text("a"), "fact", "h2", true),
        obj("k04", text("b"), "note", "h1", true),
        obj("k05", None, "note", "h1", true),
        obj("k06", Some(Value::Null), "note", "h1", true),
        obj("k07", text(""), "note", "h1", true),
        obj("k08", Some(Value::Integer(1)), "note", "h1", true),
        obj("k09", text("1"), "note", "h1", true),
        obj("k10", Some(Value::Float(1.5)), "note", "h1", true),
        obj("k11", text("01"), "note", "h1", true),
        obj("k12", Some(Value::Bool(true)), "note", "h1", true),
        obj("k13", text("true"), "note", "h1", true),
    ]
}

async fn put_all<S: StorageEngine>(db: &NodeDbLite<S>, docs: Vec<Document>) {
    for collection in [INDEXED, PLAIN] {
        for d in &docs {
            db.document_put(collection, d.clone()).await.unwrap();
        }
    }
}

async fn seeded_db() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    let db = NodeDbLite::open(storage).await.unwrap();
    put_all(&db, seed_docs()).await;
    db.execute_sql(
        &format!("CREATE INDEX idx_ko_scope ON {INDEXED} (scope)"),
        &[],
    )
    .await
    .unwrap();
    // Written after CREATE INDEX: the postings must pick these up.
    put_all(
        &db,
        vec![
            obj("k14", text("a"), "note", "h1", true),
            obj("k02", text("a"), "note", "h3", false),
            // Scope change b -> a.
            obj("k04", text("a"), "note", "h1", true),
        ],
    )
    .await;
    // Delete a document whose scope is "a".
    for collection in [INDEXED, PLAIN] {
        db.document_delete(collection, "k03").await.unwrap();
    }
    db
}

/// JSON text with object keys sorted, so two collections that serialize the
/// same document with a different key order compare equal.
fn canonical_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let inner: Vec<String> = entries
                .into_iter()
                .map(|(k, v)| format!("{k:?}:{}", canonical_json(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", inner.join(","))
        }
        other => other.to_string(),
    }
}

fn canonical_cell(column: &str, value: &Value) -> String {
    if column == "document"
        && let Value::String(s) = value
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(s)
    {
        return canonical_json(&json);
    }
    format!("{value:?}")
}

/// Columns and canonical rows of `sql_template` run against `collection`
/// (`{c}` in the template). Rows are sorted unless the query orders them.
async fn run<S: StorageEngine>(
    db: &NodeDbLite<S>,
    sql_template: &str,
    collection: &str,
    params: &[Value],
    ordered: bool,
) -> (Vec<String>, Vec<Vec<String>>) {
    let sql = sql_template.replace("{c}", collection);
    let result = db
        .execute_sql(&sql, params)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut rows: Vec<Vec<String>> = result
        .rows
        .iter()
        .map(|row| {
            result
                .columns
                .iter()
                .zip(row)
                .map(|(c, v)| canonical_cell(c, v))
                .collect()
        })
        .collect();
    if !ordered {
        rows.sort();
    }
    (result.columns, rows)
}

/// The variant name of the single plan `sql_template` produces on
/// `collection`.
async fn plan_variant<S: StorageEngine>(
    db: &NodeDbLite<S>,
    sql_template: &str,
    collection: &str,
    params: &[Value],
) -> &'static str {
    let sql = sql_template.replace("{c}", collection);
    let plans = db
        .query_engine
        .plan_sql_with_params(&sql, params)
        .await
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    assert_eq!(plans.len(), 1, "{sql}: one plan");
    plans[0].variant_name()
}

struct Case {
    sql: &'static str,
    params: Vec<Value>,
    ordered: bool,
    /// `Some(true)`: the indexed collection must plan an index lookup.
    /// `Some(false)`: it must not. `None`: either plan is acceptable.
    uses_index: Option<bool>,
}

fn case(sql: &'static str, params: Vec<Value>, uses_index: Option<bool>) -> Case {
    Case {
        sql,
        params,
        ordered: false,
        uses_index,
    }
}

#[tokio::test]
async fn indexed_select_matches_scan_rows() {
    let db = seeded_db().await;
    let s = |v: &str| Value::String(v.into());
    let cases = vec![
        case(
            "SELECT id, document FROM {c} WHERE scope = $1 AND is_latest = $2",
            vec![s("a"), Value::Bool(true)],
            Some(true),
        ),
        case(
            "SELECT id, document FROM {c} WHERE scope = $1 AND is_latest = $2",
            vec![s("a"), Value::Bool(false)],
            Some(true),
        ),
        case(
            "SELECT * FROM {c} WHERE scope = 'a'",
            Vec::new(),
            Some(true),
        ),
        Case {
            sql: "SELECT id FROM {c} WHERE scope = 'a' ORDER BY id LIMIT 2",
            params: Vec::new(),
            ordered: true,
            uses_index: None,
        },
        case(
            "SELECT id, document FROM {c} \
             WHERE scope = $1 AND type = $2 AND content_hash = $3 AND is_latest = $4",
            vec![s("a"), s("note"), s("h1"), Value::Bool(true)],
            Some(true),
        ),
        case("SELECT id FROM {c} WHERE scope = NULL", Vec::new(), None),
        case(
            "SELECT id FROM {c} WHERE scope = $1",
            vec![Value::Null],
            None,
        ),
        case("SELECT id FROM {c} WHERE scope = 1.5", Vec::new(), None),
        case(
            "SELECT id FROM {c} WHERE scope = $1",
            vec![Value::Float(1.5)],
            None,
        ),
        case("SELECT id FROM {c} WHERE scope = '1'", Vec::new(), None),
        case("SELECT id FROM {c} WHERE scope = 1", Vec::new(), None),
        // Bool `true` and the string `'true'` share a posting key. The
        // equality re-check keeps the bool document out.
        case(
            "SELECT id FROM {c} WHERE scope = 'true'",
            Vec::new(),
            Some(true),
        ),
        case(
            "SELECT id FROM {c} WHERE scope = ''",
            Vec::new(),
            Some(true),
        ),
        case(
            "SELECT id FROM {c} WHERE scope = 'zzz'",
            Vec::new(),
            Some(true),
        ),
    ];

    for c in &cases {
        let indexed = run(&db, c.sql, INDEXED, &c.params, c.ordered).await;
        let plain = run(&db, c.sql, PLAIN, &c.params, c.ordered).await;
        assert_eq!(indexed, plain, "{} {:?}", c.sql, c.params);

        assert_ne!(
            plan_variant(&db, c.sql, PLAIN, &c.params).await,
            "DocumentIndexLookup",
            "{}: the twin has no index",
            c.sql
        );
        if let Some(expected) = c.uses_index {
            let variant = plan_variant(&db, c.sql, INDEXED, &c.params).await;
            assert_eq!(
                variant == "DocumentIndexLookup",
                expected,
                "{} {:?} planned {variant}",
                c.sql,
                c.params
            );
        }
    }

    // Not vacuous: the consumer query finds the documents written before and
    // after CREATE INDEX.
    let (columns, rows) = run(
        &db,
        "SELECT id, document FROM {c} WHERE scope = $1 AND is_latest = $2",
        INDEXED,
        &[s("a"), Value::Bool(true)],
        false,
    )
    .await;
    assert_eq!(columns, ["id", "document"]);
    let ids: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    assert_eq!(
        ids,
        [
            format!("{:?}", s("k01")),
            format!("{:?}", s("k04")),
            format!("{:?}", s("k14")),
        ]
    );
    let (_, rows) = run(
        &db,
        "SELECT id FROM {c} WHERE scope = 'true'",
        INDEXED,
        &[],
        false,
    )
    .await;
    assert_eq!(rows, [[format!("{:?}", s("k13"))]]);
}

/// `LIMIT` without `ORDER BY` returns as many rows as the scan, from the
/// index lookup plan.
#[tokio::test]
async fn indexed_select_limit_without_order_matches_scan_count() {
    let db = seeded_db().await;
    let sql = "SELECT id FROM {c} WHERE scope = 'a' LIMIT 2";
    let (_, indexed) = run(&db, sql, INDEXED, &[], false).await;
    let (_, plain) = run(&db, sql, PLAIN, &[], false).await;
    assert_eq!(indexed.len(), 2);
    assert_eq!(indexed.len(), plain.len());
    assert_eq!(
        plan_variant(&db, sql, INDEXED, &[]).await,
        "DocumentIndexLookup"
    );
}

/// The consumer query with bound parameters plans an index lookup on the
/// indexed field, with the bound value as the lookup key.
#[tokio::test]
async fn explain_select_eq_uses_index_lookup() {
    let db = seeded_db().await;
    let params = [Value::String("a".into()), Value::Bool(true)];
    let sql = format!("SELECT id, document FROM {INDEXED} WHERE scope = $1 AND is_latest = $2");
    let plans = db
        .query_engine
        .plan_sql_with_params(&sql, &params)
        .await
        .unwrap();
    match plans.as_slice() {
        [
            SqlPlan::DocumentIndexLookup {
                collection,
                field,
                value,
                filters,
                ..
            },
        ] => {
            assert_eq!(collection, INDEXED);
            assert_eq!(field, "$.scope");
            assert_eq!(value, &SqlValue::String("a".into()));
            assert_eq!(filters.len(), 1, "is_latest stays a residual filter");
        }
        other => panic!(
            "expected one DocumentIndexLookup, got {:?}",
            other.iter().map(SqlPlan::variant_name).collect::<Vec<_>>()
        ),
    }

    let dedup = format!(
        "SELECT id, document FROM {INDEXED} \
         WHERE scope = $1 AND type = $2 AND content_hash = $3 AND is_latest = $4"
    );
    let params = [
        Value::String("a".into()),
        Value::String("note".into()),
        Value::String("h1".into()),
        Value::Bool(true),
    ];
    let plans = db
        .query_engine
        .plan_sql_with_params(&dedup, &params)
        .await
        .unwrap();
    assert!(
        matches!(
            plans.as_slice(),
            [SqlPlan::DocumentIndexLookup { field, .. }] if field == "$.scope"
        ),
        "dedup query plans an index lookup on scope"
    );
}

/// The postings index top-level keys only, so a dotted path is refused and
/// writes no spec.
#[tokio::test]
async fn create_index_refuses_nested_path_on_schemaless() {
    let db = seeded_db().await;
    let err = db
        .execute_sql(
            &format!("CREATE INDEX idx_nested ON {INDEXED} (\"a.b\")"),
            &[],
        )
        .await
        .expect_err("nested path refused");
    assert!(err.to_string().contains("top-level fields only"), "{err}");
    let specs = crate::query::document_ops::index_spec::load_index_specs(db.storage.as_ref())
        .await
        .unwrap();
    assert_eq!(
        specs.get(INDEXED).map(Vec::len),
        Some(1),
        "only the scope spec"
    );
}

/// Schemaless lookups read the in-memory postings, so `CREATE INDEX` writes no
/// Meta index entries.
#[tokio::test]
async fn schemaless_create_index_writes_no_meta_entries() {
    let db = seeded_db().await;
    let entries = db
        .storage
        .scan_prefix(Namespace::Meta, format!("{INDEXED}:scope:").as_bytes())
        .await
        .unwrap();
    assert!(entries.is_empty(), "no Meta index entries");
}
