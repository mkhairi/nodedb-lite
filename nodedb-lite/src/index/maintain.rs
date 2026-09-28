// SPDX-License-Identifier: Apache-2.0

//! Index maintenance: the entry changes a row write makes, and the unique
//! check that runs before the write.

use std::collections::BTreeSet;
use std::sync::Arc;

use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::storage::engine::WriteOp;

use super::catalog::{IndexDef, IndexEngine};
use super::document::{entry_keys, index_values};
use super::key;
use super::store::{IndexCatalog, IndexState};

/// One change to the stored entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IndexWriteOp {
    Put(Vec<u8>),
    Delete(Vec<u8>),
}

/// The entry changes that move document `doc_id` from the entries it holds
/// now (`old_keys`, every index of `defs`) to what `new` holds. `new = None`
/// is a removed row.
pub(crate) fn plan_index_ops(
    defs: &[Arc<IndexDef>],
    doc_id: &str,
    old_keys: &BTreeSet<Vec<u8>>,
    new: Option<&Value>,
) -> Vec<IndexWriteOp> {
    let new_keys: BTreeSet<Vec<u8>> = match new {
        Some(row) => defs
            .iter()
            .flat_map(|def| entry_keys(def, doc_id, row))
            .collect(),
        None => BTreeSet::new(),
    };
    old_keys
        .difference(&new_keys)
        .map(|k| IndexWriteOp::Delete(k.clone()))
        .chain(
            new_keys
                .difference(old_keys)
                .map(|k| IndexWriteOp::Put(k.clone())),
        )
        .collect()
}

/// Document ids of the entries under `prefix`, in key order.
pub(super) fn ids_under<'a>(
    state: &'a IndexState,
    index_prefix: &'a [u8],
    prefix: Vec<u8>,
) -> impl Iterator<Item = &'a str> + 'a {
    let scan = prefix.clone();
    state
        .entries
        .range(scan..)
        .take_while(move |k| k.starts_with(&prefix))
        .filter_map(move |k| key::entry_doc_id(k, index_prefix))
}

/// A stored document whose entry shares a key with a value a write gives a
/// unique index. It is a violation when its row holds an equal value.
pub(crate) struct UniqueCandidate {
    pub def: Arc<IndexDef>,
    pub value: Value,
    /// The stored document's id.
    pub other: String,
}

impl UniqueCandidate {
    /// Whether `stored`, the candidate's current row, holds the value.
    pub(crate) fn held_by(&self, stored: &Value) -> bool {
        index_values(&self.def, stored)
            .iter()
            .any(|held| held.eq_coerced(&self.value))
    }

    pub(crate) fn violation(&self) -> LiteError {
        unique_violation(&self.def, &self.value, &self.other)
    }
}

/// The stored documents that may already hold a value `rows` give a unique
/// index of `collection`, for the caller to confirm against their rows.
/// Conflicts among `rows` themselves are refused here.
pub(super) fn unique_candidates(
    state: &IndexState,
    collection: &str,
    engine: IndexEngine,
    rows: &[(&str, Option<Value>)],
) -> Result<Vec<UniqueCandidate>, LiteError> {
    let defs: Vec<Arc<IndexDef>> = state
        .defs_on(collection, engine)
        .into_iter()
        .filter(|d| d.unique)
        .collect();
    let written: BTreeSet<&str> = rows.iter().map(|(id, _)| *id).collect();
    let mut candidates = Vec::new();
    for def in &defs {
        let prefix = def.entry_prefix();
        // Values the write's rows hold so far, for conflicts among them.
        let mut claimed: Vec<(&str, Value)> = Vec::new();
        for (id, row) in rows {
            // A later write of the same row replaces what it claimed.
            claimed.retain(|(other, _)| other != id);
            let Some(row) = row else {
                continue;
            };
            for value in index_values(def, row) {
                // An array may repeat an element within one document.
                if let Some((other, _)) = claimed
                    .iter()
                    .find(|(owner, held)| owner != id && held.eq_coerced(&value))
                {
                    return Err(unique_violation(def, &value, other));
                }
                for encoded in key::encode_coercible(&value) {
                    let mut probe = prefix.clone();
                    probe.extend_from_slice(&encoded);
                    for other in ids_under(state, &prefix, probe) {
                        let gone = state
                            .tombstoned
                            .contains(&(collection.to_string(), other.to_string()));
                        if written.contains(other) || gone {
                            continue;
                        }
                        candidates.push(UniqueCandidate {
                            def: Arc::clone(def),
                            value: value.clone(),
                            other: other.to_string(),
                        });
                    }
                }
                claimed.push((id, value));
            }
        }
    }
    Ok(candidates)
}

fn unique_violation(def: &IndexDef, value: &Value, other: &str) -> LiteError {
    LiteError::UniqueViolation {
        collection: def.collection.clone(),
        detail: format!(
            "index '{}': key ({})=({}) already exists in document '{other}'",
            def.name,
            def.field_spec(),
            display_value(value)
        ),
    }
}

fn display_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

impl IndexCatalog {
    /// Bring the entries of `ids` in `collection` in line with their current
    /// rows. `read_row` returns a document's current contents, or `None` when
    /// it no longer exists. Call under the lock that guards the row write.
    pub(crate) fn resync<'a>(
        &self,
        collection: &str,
        ids: impl IntoIterator<Item = &'a str>,
        read_row: impl Fn(&str) -> Option<Value>,
    ) {
        let mut state = self.lock();
        let defs = state.defs_on(collection, IndexEngine::Document);
        if defs.is_empty() {
            return;
        }
        let empty = BTreeSet::new();
        for id in ids {
            let posting_key = (collection.to_string(), id.to_string());
            // A tombstoned bitemporal row keeps its CRDT copy but is gone.
            let row = if state.tombstoned.contains(&posting_key) {
                None
            } else {
                read_row(id)
            };
            let old = state.postings.get(&posting_key).unwrap_or(&empty);
            let ops = plan_index_ops(&defs, id, old, row.as_ref());
            state.apply(collection, id, ops);
        }
    }

    /// Record `ids` of `collection` as deleted while their CRDT copy stays, as
    /// a bitemporal delete leaves them, and remove their entries. Until a
    /// write revives them, no later change to the CRDT copy — a remote
    /// import, a vector merge — gives them entries again, and no unique check
    /// counts them.
    pub(crate) fn tombstone<'a>(&self, collection: &str, ids: impl IntoIterator<Item = &'a str>) {
        let ids: Vec<&str> = ids.into_iter().collect();
        {
            let mut state = self.lock();
            for id in &ids {
                state
                    .tombstoned
                    .insert((collection.to_string(), (*id).to_string()));
            }
        }
        self.resync(collection, ids, |_| None);
    }

    /// Clear the tombstone of a row a document write is about to make live.
    pub(crate) fn revive(&self, collection: &str, id: &str) {
        self.lock()
            .tombstoned
            .remove(&(collection.to_string(), id.to_string()));
    }

    /// Refuse `rows` — `(document id, contents after the write)`, `None` for a
    /// removed row — when one would give a unique index a value another
    /// document already holds, or another row of the same write holds.
    ///
    /// Candidates are found by entry key and confirmed with the coerced SQL
    /// equality on the stored row, which `read_row` supplies.
    pub(crate) fn check_unique(
        &self,
        collection: &str,
        rows: &[(&str, Option<Value>)],
        read_row: impl Fn(&str) -> Option<Value>,
    ) -> Result<(), LiteError> {
        let candidates = unique_candidates(&self.lock(), collection, IndexEngine::Document, rows)?;
        for c in candidates {
            if read_row(&c.other).is_some_and(|stored| c.held_by(&stored)) {
                return Err(c.violation());
            }
        }
        Ok(())
    }

    /// Install `def` with entries built from `rows`, replacing any index of
    /// the same name and its entries. A unique index whose rows share a value
    /// is refused and nothing changes. Returns the number of entries written.
    /// Returns the storage writes of a strict or key-value index for the
    /// caller to commit. A document index returns none: the flush persists it.
    pub(crate) fn install(
        &self,
        def: Arc<IndexDef>,
        rows: impl IntoIterator<Item = (String, Value)>,
    ) -> Result<(u64, Vec<WriteOp>), LiteError> {
        let rows: Vec<(String, Value)> = rows.into_iter().collect();
        if def.unique {
            let mut seen: Vec<(&str, Value)> = Vec::new();
            for (id, row) in &rows {
                for value in index_values(&def, row) {
                    if let Some((other, _)) = seen
                        .iter()
                        .find(|(owner, v)| owner != id && v.eq_coerced(&value))
                    {
                        return Err(unique_violation(&def, &value, other));
                    }
                    seen.push((id, value));
                }
            }
        }
        let mut state = self.lock();
        let durable = def.engine != IndexEngine::Document;
        let (built, ops) = state.collecting(durable, |state| -> Result<u64, LiteError> {
            if let Some(existing) = state.defs.get(&def.name).cloned() {
                state.clear_prefix(&existing.entry_prefix());
            }
            state.clear_prefix(&def.entry_prefix());
            state.put_def(Arc::clone(&def))?;
            let mut count = 0u64;
            for (id, row) in &rows {
                let ops: Vec<IndexWriteOp> = entry_keys(&def, id, row)
                    .into_iter()
                    .map(IndexWriteOp::Put)
                    .collect();
                count += ops.len() as u64;
                state.apply(&def.collection, id, ops);
            }
            Ok(count)
        });
        Ok((built?, ops))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::index::catalog::canonical_field;

    fn def(field: &str, unique: bool) -> Arc<IndexDef> {
        let (path, is_array) = canonical_field(field);
        Arc::new(IndexDef {
            name: format!("idx_{field}"),
            collection: "c".into(),
            path,
            unique,
            case_insensitive: false,
            is_array,
            predicate: None,
            engine: IndexEngine::Document,
        })
    }

    fn row(field: &str, v: Value) -> Value {
        Value::Object(HashMap::from([(field.to_string(), v)]))
    }

    #[test]
    fn an_update_replaces_the_old_entry() {
        let defs = vec![def("n", false)];
        let old_row = row("n", Value::Integer(1));
        let old: BTreeSet<Vec<u8>> = entry_keys(&defs[0], "d", &old_row);
        let ops = plan_index_ops(&defs, "d", &old, Some(&row("n", Value::Integer(2))));
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], IndexWriteOp::Delete(_)));
        assert!(matches!(ops[1], IndexWriteOp::Put(_)));
        assert!(plan_index_ops(&defs, "d", &old, Some(&old_row)).is_empty());
    }

    #[test]
    fn a_unique_index_refuses_a_second_holder() {
        let catalog = IndexCatalog::new();
        let d = def("email", true);
        let stored = row("email", Value::String("a@x".into()));
        catalog
            .install(Arc::clone(&d), [("d1".to_string(), stored.clone())])
            .expect("install");
        let read = |id: &str| (id == "d1").then(|| stored.clone());

        let dup = row("email", Value::String("a@x".into()));
        let err = catalog
            .check_unique("c", &[("d2", Some(dup.clone()))], read)
            .expect_err("duplicate");
        assert!(matches!(err, LiteError::UniqueViolation { .. }), "{err}");
        // The holder itself may keep its value.
        catalog
            .check_unique("c", &[("d1", Some(dup))], read)
            .expect("same document");
    }

    #[test]
    fn a_document_may_repeat_an_element_of_a_unique_array() {
        let catalog = IndexCatalog::new();
        let tags = |t: &[&str]| {
            row(
                "tags",
                Value::Array(t.iter().map(|s| Value::String((*s).into())).collect()),
            )
        };
        let stored = tags(&["a", "a"]);
        catalog
            .install(def("tags[]", true), [("d1".to_string(), stored.clone())])
            .expect("one document repeating an element");
        let read = |id: &str| (id == "d1").then(|| stored.clone());
        catalog
            .check_unique("c", &[("d2", Some(tags(&["b", "b"])))], read)
            .expect("repeats within the written document");
        assert!(
            catalog
                .check_unique("c", &[("d2", Some(tags(&["a"])))], read)
                .is_err(),
            "another document still may not share the element"
        );
    }

    #[test]
    fn install_refuses_rows_that_already_collide() {
        let catalog = IndexCatalog::new();
        let rows = [
            ("d1".to_string(), row("n", Value::Integer(5))),
            ("d2".to_string(), row("n", Value::String("5".into()))),
        ];
        assert!(catalog.install(def("n", true), rows).is_err());
        assert!(catalog.defs().is_empty());
    }
}
