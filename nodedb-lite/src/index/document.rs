// SPDX-License-Identifier: Apache-2.0

//! Document glue: what an index holds for one document row.

use std::collections::BTreeSet;

use loro::LoroValue;
use nodedb_types::value::Value;

use crate::engine::crdt::CrdtRowWrite;
use crate::nodedb::convert::loro_value_to_value;

use super::catalog::IndexDef;
use super::key;

/// The value at `path` (`$.a.b`) inside `row`, or `None` when any segment is
/// absent or crosses a non-object.
pub(crate) fn field_at<'a>(row: &'a Value, path: &str) -> Option<&'a Value> {
    let path = path.trim_start_matches('$').trim_start_matches('.');
    if path.is_empty() {
        return Some(row);
    }
    let mut current = row;
    for segment in path.split('.') {
        match current {
            Value::Object(map) => current = map.get(segment)?,
            _ => return None,
        }
    }
    Some(current)
}

/// Lowercase a string value for a case-insensitive index.
pub(crate) fn fold_case(def: &IndexDef, value: Value) -> Value {
    match value {
        Value::String(s) if def.case_insensitive => Value::String(s.to_lowercase()),
        other => other,
    }
}

fn indexable(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Array(_) | Value::Object(_))
}

/// The values `def` holds for `row`: none when the row fails the partial
/// predicate or lacks the field, one per scalar element of an array under an
/// array index, otherwise the field's scalar value. NULL is never indexed.
pub(crate) fn index_values(def: &IndexDef, row: &Value) -> Vec<Value> {
    if let Some(predicate) = &def.predicate
        && !predicate.matches(row)
    {
        return Vec::new();
    }
    let Some(field) = field_at(row, &def.path) else {
        return Vec::new();
    };
    match field {
        Value::Array(items) if def.is_array => items
            .iter()
            .filter(|v| indexable(v))
            .map(|v| fold_case(def, v.clone()))
            .collect(),
        v if indexable(v) => vec![fold_case(def, v.clone())],
        _ => Vec::new(),
    }
}

/// Every entry key `def` holds for document `doc_id` with contents `row`.
pub(crate) fn entry_keys(def: &IndexDef, doc_id: &str, row: &Value) -> BTreeSet<Vec<u8>> {
    let prefix = def.entry_prefix();
    index_values(def, row)
        .iter()
        .flat_map(key::encode_coercible)
        .map(|encoded| key::entry_key(&prefix, &encoded, doc_id))
        .collect()
}

/// Whether `row` satisfies `<field> = probe` as a scan evaluates it: the
/// coerced SQL equality on the field's own value, case-folded for a
/// case-insensitive index.
pub(crate) fn matches_probe(def: &IndexDef, row: &Value, probe: &Value) -> bool {
    let Some(field) = field_at(row, &def.path) else {
        return false;
    };
    let field = fold_case(def, field.clone());
    let probe = fold_case(def, probe.clone());
    field.eq_coerced(&probe)
}

/// A CRDT row as a document value.
pub(crate) fn row_value(row: &LoroValue) -> Value {
    loro_value_to_value(row)
}

/// The row a CRDT write leaves behind: `fields` alone for a full-row upsert,
/// `fields` merged over `base` for a field merge.
pub(crate) fn predicted_row(
    base: Option<Value>,
    mode: CrdtRowWrite,
    fields: &[(&str, LoroValue)],
) -> Value {
    let mut map = match (mode, base) {
        (CrdtRowWrite::SetFields, Some(Value::Object(map))) => map,
        _ => std::collections::HashMap::with_capacity(fields.len()),
    };
    for (name, value) in fields {
        map.insert((*name).to_string(), loro_value_to_value(value));
    }
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::index::catalog::{IndexEngine, IndexPredicate, canonical_field};

    fn def(field: &str, case_insensitive: bool, predicate: Option<&str>) -> IndexDef {
        let (path, is_array) = canonical_field(field);
        IndexDef {
            name: "i".into(),
            collection: "c".into(),
            path,
            unique: false,
            case_insensitive,
            is_array,
            predicate: predicate.map(|p| IndexPredicate::parse(p).expect("predicate")),
            engine: IndexEngine::Document,
        }
    }

    fn row(pairs: &[(&str, Value)]) -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect::<HashMap<_, _>>(),
        )
    }

    #[test]
    fn nested_paths_resolve_through_objects() {
        let inner = row(&[("b", Value::Integer(1))]);
        let r = row(&[("a", inner)]);
        assert_eq!(field_at(&r, "$.a.b"), Some(&Value::Integer(1)));
        assert_eq!(field_at(&r, "$.a.c"), None);
    }

    #[test]
    fn an_array_index_holds_each_scalar_element() {
        let r = row(&[(
            "tags",
            Value::Array(vec![
                Value::String("x".into()),
                Value::Null,
                Value::String("y".into()),
            ]),
        )]);
        let values = index_values(&def("tags[]", false, None), &r);
        assert_eq!(
            values,
            vec![Value::String("x".into()), Value::String("y".into())]
        );
        assert!(index_values(&def("tags", false, None), &r).is_empty());
    }

    #[test]
    fn a_partial_index_skips_rows_failing_its_predicate() {
        let d = def("email", true, Some("active = true"));
        let active = row(&[
            ("email", Value::String("A@X".into())),
            ("active", Value::Bool(true)),
        ]);
        let inactive = row(&[
            ("email", Value::String("A@X".into())),
            ("active", Value::Bool(false)),
        ]);
        assert_eq!(index_values(&d, &active), vec![Value::String("a@x".into())]);
        assert!(index_values(&d, &inactive).is_empty());
    }
}
