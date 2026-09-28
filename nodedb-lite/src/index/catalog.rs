// SPDX-License-Identifier: Apache-2.0

//! Secondary-index definitions: what `CREATE INDEX` declared, persisted in
//! `Namespace::Meta` and loaded at open.

use nodedb_query::SqlExpr;
use nodedb_types::value::Value;

use crate::error::LiteError;

use super::key;

/// The engine whose rows an index covers.
///
/// Encoded as its discriminant: do not reorder the variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, zerompk::ToMessagePack, zerompk::FromMessagePack)]
#[msgpack(c_enum)]
pub enum IndexEngine {
    /// Schemaless documents held in the CRDT store.
    Document = 0,
    /// Strict collections. Declared and persisted; entries are not built yet.
    Strict = 1,
    /// Key-value collections. Declared and persisted; entries are not built yet.
    KeyValue = 2,
}

/// A partial-index predicate: the `WHERE` body as written and its parsed form.
///
/// Only rows for which the expression evaluates to exactly `true` are indexed.
/// NULL, `false`, a non-boolean result and an evaluation error all exclude the
/// row, as a Postgres partial index does.
#[derive(Debug, Clone, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub struct IndexPredicate {
    /// The SQL text. The planner parses it to decide whether a query's WHERE
    /// clause entails the predicate.
    pub sql: String,
    /// The parsed expression the write path evaluates per row.
    pub expr: SqlExpr,
}

impl IndexPredicate {
    /// Parse the `WHERE` body of a `CREATE INDEX`.
    pub fn parse(sql: &str) -> Result<Self, LiteError> {
        let (expr, _deps) = nodedb_query::expr_parse::parse_generated_expr(sql).map_err(|e| {
            LiteError::BadRequest {
                detail: format!(
                    "CREATE INDEX: partial-index predicate '{sql}' does not parse: {e}"
                ),
            }
        })?;
        Ok(Self {
            sql: sql.to_string(),
            expr,
        })
    }

    /// Whether `row` belongs in the index.
    pub fn matches(&self, row: &Value) -> bool {
        matches!(self.expr.eval(row), Ok(Value::Bool(true)))
    }
}

/// One secondary index.
#[derive(Debug, Clone, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub struct IndexDef {
    /// Index name, unique across the database.
    pub name: String,
    /// Collection the index covers.
    pub collection: String,
    /// Canonical JSON path of the indexed field, e.g. `$.email` or `$.a.b`.
    pub path: String,
    /// Two rows may not share an indexed value.
    pub unique: bool,
    /// String values are indexed and probed lowercased.
    pub case_insensitive: bool,
    /// An array value is indexed once per element.
    pub is_array: bool,
    /// Partial-index predicate, or `None` for a full index.
    pub predicate: Option<IndexPredicate>,
    /// The engine whose rows the index covers.
    pub engine: IndexEngine,
}

impl IndexDef {
    /// The field as the planner and `CREATE INDEX` spell it: the canonical
    /// path, with `[]` appended for an array index.
    pub fn field_spec(&self) -> String {
        field_spec(&self.path, self.is_array)
    }

    /// Prefix of every entry of this index.
    pub(crate) fn entry_prefix(&self) -> Vec<u8> {
        key::index_prefix(&self.collection, &self.name)
    }

    /// The planner's view of this index. Only document indexes drive query
    /// rewrites: strict and key-value indexes hold no entries yet.
    pub(crate) fn planner_spec(&self) -> nodedb_sql::types::IndexSpec {
        nodedb_sql::types::IndexSpec {
            name: self.name.clone(),
            field: self.field_spec(),
            unique: self.unique,
            case_insensitive: self.case_insensitive,
            state: nodedb_sql::types::IndexState::Ready,
            predicate: self.predicate.as_ref().map(|p| p.sql.clone()),
        }
    }

    /// Serialized form stored under [`key::def_key`].
    pub(crate) fn encode(&self) -> Result<Vec<u8>, LiteError> {
        zerompk::to_msgpack_vec(self).map_err(|e| LiteError::Serialization {
            detail: format!("encode index definition '{}': {e}", self.name),
        })
    }

    /// Decode a stored definition.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, LiteError> {
        zerompk::from_msgpack(bytes).map_err(|e| LiteError::Corrupted {
            detail: format!("stored index definition does not decode: {e}"),
        })
    }
}

/// Split a `CREATE INDEX` field into its canonical path and array flag.
///
/// `email` and `$.email` both name `$.email`. A trailing `[]` (`tags[]`)
/// declares an array index over the elements of `$.tags`.
pub fn canonical_field(field: &str) -> (String, bool) {
    let (field, is_array) = match field.strip_suffix("[]") {
        Some(inner) => (inner, true),
        None => (field, false),
    };
    let path = if field.starts_with('$') {
        field.to_string()
    } else {
        format!("$.{field}")
    };
    (path, is_array)
}

/// A field as the planner spells it: the canonical path, with `[]` appended
/// for an array index.
pub fn field_spec(path: &str, is_array: bool) -> String {
    if is_array {
        format!("{path}[]")
    } else {
        path.to_string()
    }
}

/// The name `CREATE INDEX` gives an index it was not given a name for.
pub fn default_index_name(collection: &str, field: &str) -> String {
    format!("idx_{collection}_{field}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_canonicalize_to_json_paths() {
        assert_eq!(canonical_field("email"), ("$.email".into(), false));
        assert_eq!(canonical_field("$.a.b"), ("$.a.b".into(), false));
        assert_eq!(canonical_field("tags[]"), ("$.tags".into(), true));
    }

    #[test]
    fn a_definition_round_trips_with_its_predicate() {
        let def = IndexDef {
            name: "idx".into(),
            collection: "users".into(),
            path: "$.email".into(),
            unique: true,
            case_insensitive: true,
            is_array: false,
            predicate: Some(IndexPredicate::parse("active = true").expect("parse")),
            engine: IndexEngine::Document,
        };
        let decoded = IndexDef::decode(&def.encode().expect("encode")).expect("decode");
        assert_eq!(decoded.name, "idx");
        assert!(decoded.unique && decoded.case_insensitive);
        assert_eq!(decoded.engine, IndexEngine::Document);
        let predicate = decoded.predicate.expect("predicate");
        assert_eq!(predicate.sql, "active = true");
        let mut row = std::collections::HashMap::new();
        row.insert("active".to_string(), Value::Bool(true));
        assert!(predicate.matches(&Value::Object(row)));
    }
}
