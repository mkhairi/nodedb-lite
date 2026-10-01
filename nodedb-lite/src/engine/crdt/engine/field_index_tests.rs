// SPDX-License-Identifier: BUSL-1.1

//! Document-index upkeep: the index follows every write the engine applies
//! and every path that rewrites a collection's state.

use std::sync::Arc;

use loro::LoroValue;
use nodedb_types::value::Value;

use super::types::{CrdtBatchOp, CrdtEngine, CrdtField};
use crate::index::IndexCatalog;
use crate::index::catalog::{IndexDef, IndexEngine, canonical_field};
use crate::index::rebuild::crdt_rows;

fn text(value: &str) -> LoroValue {
    LoroValue::String(value.into())
}

/// An engine maintaining a document index on `notes.scope`.
fn indexed_engine(peer: u64) -> (CrdtEngine, Arc<IndexCatalog>, Arc<IndexDef>) {
    let mut engine = CrdtEngine::new(peer).unwrap();
    let catalog = Arc::new(IndexCatalog::new());
    engine.set_index_catalog(Arc::clone(&catalog));
    let def = install(&engine, &catalog, "idx_scope", "scope", false);
    (engine, catalog, def)
}

/// Install a document index on `notes.<field>`, built from the rows present.
fn install(
    engine: &CrdtEngine,
    catalog: &IndexCatalog,
    name: &str,
    field: &str,
    case_insensitive: bool,
) -> Arc<IndexDef> {
    let (path, is_array) = canonical_field(field);
    let def = Arc::new(IndexDef {
        name: name.into(),
        collection: "notes".into(),
        path,
        unique: false,
        case_insensitive,
        is_array,
        predicate: None,
        engine: IndexEngine::Document,
    });
    catalog
        .install(Arc::clone(&def), crdt_rows(engine, "notes"))
        .unwrap();
    def
}

fn lookup(catalog: &IndexCatalog, def: &IndexDef, probe: Value) -> Vec<String> {
    let mut ids = catalog.lookup_eq(def, &probe);
    ids.sort();
    ids
}

fn s(value: &str) -> Value {
    Value::String(value.into())
}

/// Every row with a value in the indexed field is found under that value, and
/// every id a lookup returns holds the value it was looked up by.
fn assert_consistent(engine: &CrdtEngine, catalog: &IndexCatalog, def: &IndexDef) {
    let field = def.path.trim_start_matches("$.");
    let rows = crdt_rows(engine, "notes");
    for (id, row) in &rows {
        let Value::Object(fields) = row else { continue };
        let Some(value) = fields.get(field).filter(|v| !matches!(v, Value::Null)) else {
            continue;
        };
        assert!(
            catalog.lookup_eq(def, value).contains(id),
            "{id} holds {value:?} but the index does not find it"
        );
        for other in catalog.lookup_eq(def, value) {
            let held = rows
                .iter()
                .find(|(rid, _)| *rid == other)
                .and_then(|(_, r)| match r {
                    Value::Object(f) => f.get(field).cloned(),
                    _ => None,
                });
            assert_eq!(
                held.as_ref(),
                Some(value),
                "{other} is indexed under a stale value"
            );
        }
    }
}

#[test]
fn index_tracks_put_update_delete() {
    let (mut engine, catalog, def) = indexed_engine(1);

    engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    assert_eq!(lookup(&catalog, &def, s("a")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);

    // An update moves the document from the old value's entry to the new.
    engine
        .upsert("notes", "d1", &[("scope", text("b"))])
        .unwrap();
    assert!(lookup(&catalog, &def, s("a")).is_empty());
    assert_eq!(lookup(&catalog, &def, s("b")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);

    // Partial update, batch upsert and deferred write.
    engine
        .set_fields("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    let d2_fields: [CrdtField<'_>; 1] = [("scope", text("a"))];
    let batch: [CrdtBatchOp<'_>; 1] = [("notes", "d2", &d2_fields[..])];
    engine.batch_upsert(&batch).unwrap();
    engine
        .upsert_deferred("notes", "d3", &[("scope", text("a"))])
        .unwrap();
    assert_eq!(lookup(&catalog, &def, s("a")), ["d1", "d2", "d3"]);
    assert!(lookup(&catalog, &def, s("b")).is_empty());
    assert_consistent(&engine, &catalog, &def);

    // A delete removes the document from its entry.
    engine.delete("notes", "d1").unwrap();
    engine.delete_deferred("notes", "d3").unwrap();
    assert_eq!(lookup(&catalog, &def, s("a")), ["d2"]);
    assert_consistent(&engine, &catalog, &def);

    // A full upsert without the field removes it, so the document leaves
    // the index.
    engine
        .upsert("notes", "d2", &[("title", text("t"))])
        .unwrap();
    assert!(lookup(&catalog, &def, s("a")).is_empty());
    assert_consistent(&engine, &catalog, &def);

    // Null is not indexed. A non-string scalar is, under its typed value.
    engine
        .upsert("notes", "d4", &[("scope", LoroValue::Null)])
        .unwrap();
    engine
        .upsert("notes", "d5", &[("scope", LoroValue::I64(7))])
        .unwrap();
    assert!(lookup(&catalog, &def, Value::Null).is_empty());
    assert_eq!(lookup(&catalog, &def, Value::Integer(7)), ["d5"]);
    assert_consistent(&engine, &catalog, &def);

    // Clearing the collection empties every entry.
    engine.clear_collection("notes").unwrap();
    assert!(lookup(&catalog, &def, Value::Integer(7)).is_empty());
}

#[test]
fn install_builds_from_existing_documents() {
    let mut engine = CrdtEngine::new(1).unwrap();
    let catalog = Arc::new(IndexCatalog::new());
    engine.set_index_catalog(Arc::clone(&catalog));
    engine
        .upsert("notes", "d1", &[("scope", text("Team-A"))])
        .unwrap();
    engine
        .upsert("notes", "d2", &[("title", text("no scope"))])
        .unwrap();

    let def = install(&engine, &catalog, "idx_scope", "scope", false);
    assert_eq!(lookup(&catalog, &def, s("Team-A")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);

    // Installing again under the same name rebuilds. A case-insensitive
    // index matches a probe in any case.
    let def = install(&engine, &catalog, "idx_scope", "scope", true);
    assert_eq!(lookup(&catalog, &def, s("team-a")), ["d1"]);
    assert_eq!(lookup(&catalog, &def, s("TEAM-A")), ["d1"]);

    catalog.forget("idx_scope");
    assert!(catalog.def_named("idx_scope").is_none());
}

#[test]
fn index_rebuilt_after_import_remote() {
    let mut origin = CrdtEngine::new(1).unwrap();
    origin
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    origin
        .upsert("notes", "d2", &[("scope", text("b"))])
        .unwrap();
    let snapshot = origin.export_snapshot("notes").unwrap();

    let (mut replica, catalog, def) = indexed_engine(2);
    replica.import_remote("notes", &snapshot).unwrap();
    assert_eq!(lookup(&catalog, &def, s("a")), ["d1"]);
    assert_eq!(lookup(&catalog, &def, s("b")), ["d2"]);
    assert_consistent(&replica, &catalog, &def);

    // A later remote update moves d1 from a to b.
    let seen = replica
        .state("notes")
        .expect("imported")
        .oplog_version_vector();
    origin
        .upsert("notes", "d1", &[("scope", text("b"))])
        .unwrap();
    let delta = origin.export_delta_from("notes", &seen).unwrap();
    replica.import_remote("notes", &delta).unwrap();
    assert!(lookup(&catalog, &def, s("a")).is_empty());
    assert_eq!(lookup(&catalog, &def, s("b")), ["d1", "d2"]);
    assert_consistent(&replica, &catalog, &def);

    // Replaying the delta contributes nothing and leaves the entries intact.
    replica.import_remote("notes", &delta).unwrap();
    assert_eq!(lookup(&catalog, &def, s("b")), ["d1", "d2"]);
    assert_consistent(&replica, &catalog, &def);

    // A snapshot restored from storage rebuilds the same way.
    let (mut restored, catalog, def) = indexed_engine(1);
    restored
        .import_snapshot("notes", &origin.export_snapshot("notes").unwrap())
        .unwrap();
    assert_eq!(lookup(&catalog, &def, s("b")), ["d1", "d2"]);
    assert_consistent(&restored, &catalog, &def);
}

#[test]
fn index_survives_compaction_and_peer_rotation() {
    let (mut engine, catalog, def) = indexed_engine(1);
    engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    engine
        .upsert("notes", "d1", &[("scope", text("b"))])
        .unwrap();

    engine.compact_history().unwrap();
    assert_eq!(lookup(&catalog, &def, s("b")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);

    engine.rotate_peer_id(2).unwrap();
    assert_eq!(lookup(&catalog, &def, s("b")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);

    // Writes after both still move the document.
    engine
        .upsert("notes", "d1", &[("scope", text("c"))])
        .unwrap();
    assert!(lookup(&catalog, &def, s("b")).is_empty());
    assert_eq!(lookup(&catalog, &def, s("c")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);
}

#[test]
fn index_follows_rollbacks_restores_and_partial_writes() {
    let (mut engine, catalog, def) = indexed_engine(1);

    // A rejected delta rolls its row back by deleting it.
    let rejected = engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    assert!(engine.reject_delta(rejected).is_some());
    assert!(lookup(&catalog, &def, s("a")).is_empty());
    assert_consistent(&engine, &catalog, &def);

    // A restore to an earlier version moves the row back.
    engine
        .upsert("notes", "d2", &[("scope", text("a"))])
        .unwrap();
    let earlier = engine
        .state("notes")
        .expect("written")
        .oplog_version_vector();
    engine
        .upsert("notes", "d2", &[("scope", text("b"))])
        .unwrap();
    engine.restore_to_version("notes", "d2", &earlier).unwrap();
    assert_eq!(lookup(&catalog, &def, s("a")), ["d2"]);
    assert!(lookup(&catalog, &def, s("b")).is_empty());
    assert_consistent(&engine, &catalog, &def);

    // An upsert that fails on a container-valued key has already written
    // the fields before it. The index follows what was applied.
    let block = sonic_rs::from_str("{\"text\": \"x\"}").unwrap();
    engine
        .list_insert("notes", "d2", "blocks", 0, &block)
        .unwrap();
    let failed = engine.upsert(
        "notes",
        "d2",
        &[("scope", text("z")), ("blocks", LoroValue::I64(1))],
    );
    assert!(failed.is_err(), "a scalar must not shadow a container");
    assert_consistent(&engine, &catalog, &def);
}

#[test]
fn index_follows_policy_resolutions() {
    use nodedb_types::sync::compensation::CompensationHint;

    let hint = CompensationHint::UniqueViolation {
        field: "scope".into(),
        conflicting_value: "a".into(),
    };
    let (mut engine, catalog, def) = indexed_engine(1);

    // RenameSuffix rewrites the indexed value in place.
    engine.set_policy(
        "notes",
        nodedb_crdt::CollectionPolicy {
            unique: nodedb_crdt::ConflictPolicy::RenameSuffix,
            ..nodedb_crdt::CollectionPolicy::ephemeral()
        },
    );
    let renamed = engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    assert!(matches!(
        engine.reject_delta_with_policy(renamed, &hint),
        Some(nodedb_crdt::PolicyResolution::AutoResolved(_))
    ));
    assert!(lookup(&catalog, &def, s("a")).is_empty());
    assert_eq!(lookup(&catalog, &def, s("a_1")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);

    // EscalateToDlq deletes the row.
    engine.set_policy("notes", nodedb_crdt::CollectionPolicy::strict());
    let escalated = engine
        .upsert("notes", "d2", &[("scope", text("a"))])
        .unwrap();
    assert!(matches!(
        engine.reject_delta_with_policy(escalated, &hint),
        Some(nodedb_crdt::PolicyResolution::Escalate { .. })
    ));
    assert!(lookup(&catalog, &def, s("a")).is_empty());
    assert_eq!(lookup(&catalog, &def, s("a_1")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);
}

#[test]
fn index_tracks_each_field_of_a_collection() {
    let (mut engine, catalog, def) = indexed_engine(1);
    let kind = install(&engine, &catalog, "idx_kind", "kind", false);
    engine
        .upsert("notes", "d1", &[("scope", text("a")), ("kind", text("x"))])
        .unwrap();

    // Only the second field changes. The first keeps its entry.
    engine
        .upsert("notes", "d1", &[("scope", text("a")), ("kind", text("y"))])
        .unwrap();
    assert_eq!(lookup(&catalog, &def, s("a")), ["d1"]);
    assert!(lookup(&catalog, &kind, s("x")).is_empty());
    assert_eq!(lookup(&catalog, &kind, s("y")), ["d1"]);
    assert_consistent(&engine, &catalog, &def);
}
