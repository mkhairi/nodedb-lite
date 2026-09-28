// SPDX-License-Identifier: Apache-2.0

//! Index reads: the candidate documents for an equality or a range.
//!
//! Both return a superset of the rows the comparison matches under coerced
//! SQL semantics (see [`key::encode_coercible`]); the reader confirms each
//! candidate against its row.

use std::collections::HashSet;
use std::ops::Bound;

use nodedb_types::value::Value;

use super::catalog::IndexDef;
use super::document::fold_case;
use super::key;
use super::maintain::ids_under;
use super::store::IndexCatalog;

/// One value class a range covers: `(type tag, lower key, upper key)`.
type ClassRange = (u8, Option<Vec<u8>>, Option<Vec<u8>>);

/// Keep the first occurrence of each id, in order.
fn dedup(ids: impl Iterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    ids.filter(|id| seen.insert(id.clone())).collect()
}

impl IndexCatalog {
    /// Documents whose indexed value may equal `probe`.
    pub(crate) fn lookup_eq(&self, def: &IndexDef, probe: &Value) -> Vec<String> {
        let probe = fold_case(def, probe.clone());
        let prefix = def.entry_prefix();
        let state = self.lock();
        let mut ids = Vec::new();
        for encoded in key::encode_coercible(&probe) {
            let mut scan = prefix.clone();
            scan.extend_from_slice(&encoded);
            ids.extend(ids_under(&state, &prefix, scan).map(str::to_string));
        }
        dedup(ids.into_iter())
    }

    /// Documents whose indexed value may lie within the bounds, each
    /// `(value, inclusive)`, in index order: ascending within each value
    /// class. `None` when the two bounds share no value class, so no ordered
    /// range covers both and the caller scans instead.
    pub(crate) fn lookup_range(
        &self,
        def: &IndexDef,
        lower: Option<&Value>,
        upper: Option<&Value>,
    ) -> Option<Vec<String>> {
        let classes = |bound: Option<&Value>| -> Option<Vec<Vec<u8>>> {
            bound.map(|v| key::encode_coercible(&fold_case(def, v.clone())))
        };
        let lower_keys = classes(lower);
        let upper_keys = classes(upper);
        let tag_of = |k: &Vec<u8>| k.first().copied();

        // One (tag, lower key, upper key) per value class both bounds share.
        let mut ranges: Vec<ClassRange> = Vec::new();
        match (&lower_keys, &upper_keys) {
            (Some(lo), Some(hi)) => {
                for l in lo {
                    if let Some(h) = hi.iter().find(|h| tag_of(h) == tag_of(l))
                        && let Some(tag) = tag_of(l)
                    {
                        ranges.push((tag, Some(l.clone()), Some(h.clone())));
                    }
                }
            }
            (Some(lo), None) => {
                for l in lo {
                    if let Some(tag) = tag_of(l) {
                        ranges.push((tag, Some(l.clone()), None));
                    }
                }
            }
            (None, Some(hi)) => {
                for h in hi {
                    if let Some(tag) = tag_of(h) {
                        ranges.push((tag, None, Some(h.clone())));
                    }
                }
            }
            (None, None) => return None,
        }
        if ranges.is_empty() {
            return None;
        }

        let prefix = def.entry_prefix();
        let state = self.lock();
        let mut ids = Vec::new();
        for (tag, lo, hi) in ranges {
            let with_prefix = |bytes: &[u8]| {
                let mut k = prefix.clone();
                k.extend_from_slice(bytes);
                k
            };
            let start = with_prefix(&lo.unwrap_or_else(|| vec![tag]));
            // Every entry of value `hi` is `prefix | hi | doc id`, and a doc id
            // is UTF-8, which never contains 0xFF.
            let end = match hi {
                Some(h) => {
                    let mut k = with_prefix(&h);
                    k.push(0xFF);
                    Bound::Included(k)
                }
                None => Bound::Excluded(with_prefix(&[tag + 1])),
            };
            // Bounds that cross (`x > 10 AND x < 5`) select nothing; handing
            // them to `range` would panic.
            let crossed = match &end {
                Bound::Included(e) => start > *e,
                Bound::Excluded(e) => start >= *e,
                Bound::Unbounded => false,
            };
            if crossed {
                continue;
            }
            let start = Bound::Included(start);
            ids.extend(
                state
                    .entries
                    .range::<Vec<u8>, _>((start, end))
                    .filter_map(|k| key::entry_doc_id(k, &prefix))
                    .map(str::to_string),
            );
        }
        Some(dedup(ids.into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::index::catalog::{IndexEngine, canonical_field};

    fn catalog_with(values: &[(&str, Value)]) -> (IndexCatalog, Arc<IndexDef>) {
        let (path, is_array) = canonical_field("n");
        let def = Arc::new(IndexDef {
            name: "idx_n".into(),
            collection: "c".into(),
            path,
            unique: false,
            case_insensitive: false,
            is_array,
            predicate: None,
            engine: IndexEngine::Document,
        });
        let rows = values.iter().map(|(id, v)| {
            (
                (*id).to_string(),
                Value::Object(HashMap::from([("n".to_string(), v.clone())])),
            )
        });
        let catalog = IndexCatalog::new();
        catalog.install(Arc::clone(&def), rows).expect("install");
        (catalog, def)
    }

    #[test]
    fn a_numeric_range_follows_numeric_order() {
        let (catalog, def) = catalog_with(&[
            ("a", Value::Integer(100)),
            ("b", Value::Integer(2)),
            ("c", Value::Integer(10)),
            ("d", Value::Float(5.5)),
        ]);
        let ids = catalog
            .lookup_range(&def, Some(&Value::Integer(3)), Some(&Value::Integer(50)))
            .expect("shared class");
        assert_eq!(ids, vec!["d", "c"]);
        let open = catalog
            .lookup_range(&def, Some(&Value::Integer(10)), None)
            .expect("class");
        assert_eq!(open, vec!["c", "a"]);
        let crossed = catalog
            .lookup_range(&def, Some(&Value::Integer(50)), Some(&Value::Integer(3)))
            .expect("shared class");
        assert!(crossed.is_empty());
    }

    #[test]
    fn an_equality_probe_finds_coercible_values() {
        let (catalog, def) = catalog_with(&[
            ("a", Value::String("7".into())),
            ("b", Value::Integer(7)),
            ("c", Value::Integer(8)),
        ]);
        let mut ids = catalog.lookup_eq(&def, &Value::Integer(7));
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
