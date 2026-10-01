// SPDX-License-Identifier: BUSL-1.1

//! Field-index maintenance: the postings follow every write the engine
//! applies and every path that rewrites a collection's state.

use loro::LoroValue;

use super::types::{CrdtEngine, CrdtField};

const FIELD: &str = "$.scope";

fn text(value: &str) -> LoroValue {
    LoroValue::String(value.into())
}

fn lookup(engine: &CrdtEngine, key: &str) -> Vec<String> {
    engine
        .field_index_lookup("notes", FIELD, key)
        .expect("index registered")
}

#[test]
fn index_tracks_put_update_delete() {
    let mut engine = CrdtEngine::new(1).unwrap();
    engine.register_field_index("notes", FIELD, false);

    engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    assert_eq!(lookup(&engine, "a"), ["d1"]);
    engine.assert_field_indexes_consistent();

    // An update moves the document from the old value's posting to the new.
    engine
        .upsert("notes", "d1", &[("scope", text("b"))])
        .unwrap();
    assert!(lookup(&engine, "a").is_empty());
    assert_eq!(lookup(&engine, "b"), ["d1"]);
    engine.assert_field_indexes_consistent();

    // Partial update, batch upsert and deferred write.
    engine
        .set_fields("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    let d2_fields: [CrdtField<'_>; 1] = [("scope", text("a"))];
    engine
        .batch_upsert(&[("notes", "d2", &d2_fields[..])])
        .unwrap();
    engine
        .upsert_deferred("notes", "d3", &[("scope", text("a"))])
        .unwrap();
    assert_eq!(lookup(&engine, "a"), ["d1", "d2", "d3"]);
    assert!(lookup(&engine, "b").is_empty());
    engine.assert_field_indexes_consistent();

    // A delete removes the document from its posting.
    engine.delete("notes", "d1").unwrap();
    engine.delete_deferred("notes", "d3").unwrap();
    assert_eq!(lookup(&engine, "a"), ["d2"]);
    engine.assert_field_indexes_consistent();

    // A full upsert without the field removes it, so the document leaves
    // the index.
    engine
        .upsert("notes", "d2", &[("title", text("t"))])
        .unwrap();
    assert!(lookup(&engine, "a").is_empty());
    engine.assert_field_indexes_consistent();

    // Null is not indexed. A non-string scalar is, under its key form.
    engine
        .upsert("notes", "d4", &[("scope", LoroValue::Null)])
        .unwrap();
    engine
        .upsert("notes", "d5", &[("scope", LoroValue::I64(7))])
        .unwrap();
    assert!(lookup(&engine, "").is_empty());
    assert_eq!(lookup(&engine, "7"), ["d5"]);
    engine.assert_field_indexes_consistent();

    // Clearing the collection empties every posting.
    engine.clear_collection("notes").unwrap();
    assert!(lookup(&engine, "7").is_empty());
    engine.assert_field_indexes_consistent();
}

#[test]
fn register_builds_from_existing_documents() {
    let mut engine = CrdtEngine::new(1).unwrap();
    engine
        .upsert("notes", "d1", &[("scope", text("Team-A"))])
        .unwrap();
    engine
        .upsert("notes", "d2", &[("title", text("no scope"))])
        .unwrap();
    assert!(
        engine
            .field_index_lookup("notes", FIELD, "Team-A")
            .is_none()
    );

    engine.register_field_index("notes", FIELD, false);
    assert_eq!(lookup(&engine, "Team-A"), ["d1"]);
    engine.assert_field_indexes_consistent();

    // Registering again rebuilds. A case-insensitive index keys lowercased
    // values and lowercases the key looked up.
    engine.register_field_index("notes", FIELD, true);
    assert_eq!(lookup(&engine, "team-a"), ["d1"]);
    assert_eq!(lookup(&engine, "TEAM-A"), ["d1"]);
    engine.assert_field_indexes_consistent();

    engine.drop_field_index("notes", FIELD);
    assert!(
        engine
            .field_index_lookup("notes", FIELD, "team-a")
            .is_none()
    );
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

    let mut replica = CrdtEngine::new(2).unwrap();
    replica.register_field_index("notes", FIELD, false);
    replica.import_remote("notes", &snapshot).unwrap();
    assert_eq!(lookup(&replica, "a"), ["d1"]);
    assert_eq!(lookup(&replica, "b"), ["d2"]);
    replica.assert_field_indexes_consistent();

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
    assert!(lookup(&replica, "a").is_empty());
    assert_eq!(lookup(&replica, "b"), ["d1", "d2"]);
    replica.assert_field_indexes_consistent();

    // Replaying the delta contributes nothing and leaves the postings intact.
    replica.import_remote("notes", &delta).unwrap();
    assert_eq!(lookup(&replica, "b"), ["d1", "d2"]);
    replica.assert_field_indexes_consistent();

    // A snapshot restored from storage rebuilds the same way.
    let mut restored = CrdtEngine::new(1).unwrap();
    restored.register_field_index("notes", FIELD, false);
    restored
        .import_snapshot("notes", &origin.export_snapshot("notes").unwrap())
        .unwrap();
    assert_eq!(lookup(&restored, "b"), ["d1", "d2"]);
    restored.assert_field_indexes_consistent();
}

#[test]
fn index_survives_compaction_and_peer_rotation() {
    let mut engine = CrdtEngine::new(1).unwrap();
    engine.register_field_index("notes", FIELD, false);
    engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    engine
        .upsert("notes", "d1", &[("scope", text("b"))])
        .unwrap();

    engine.compact_history().unwrap();
    assert_eq!(lookup(&engine, "b"), ["d1"]);
    engine.assert_field_indexes_consistent();

    engine.rotate_peer_id(2).unwrap();
    assert_eq!(lookup(&engine, "b"), ["d1"]);
    engine.assert_field_indexes_consistent();

    // Writes after both still move the document.
    engine
        .upsert("notes", "d1", &[("scope", text("c"))])
        .unwrap();
    assert!(lookup(&engine, "b").is_empty());
    assert_eq!(lookup(&engine, "c"), ["d1"]);
    engine.assert_field_indexes_consistent();
}

#[test]
fn index_follows_rollbacks_restores_and_partial_writes() {
    let mut engine = CrdtEngine::new(1).unwrap();
    engine.register_field_index("notes", FIELD, false);

    // A rejected delta rolls its row back by deleting it.
    let rejected = engine
        .upsert("notes", "d1", &[("scope", text("a"))])
        .unwrap();
    assert!(engine.reject_delta(rejected).is_some());
    assert!(lookup(&engine, "a").is_empty());
    engine.assert_field_indexes_consistent();

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
    assert_eq!(lookup(&engine, "a"), ["d2"]);
    assert!(lookup(&engine, "b").is_empty());
    engine.assert_field_indexes_consistent();

    // An upsert that fails on a container-valued key has already written
    // the fields before it. The postings follow what was applied.
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
    assert_eq!(lookup(&engine, "z"), ["d2"]);
    engine.assert_field_indexes_consistent();
}

#[test]
fn index_follows_policy_resolutions() {
    use nodedb_types::sync::compensation::CompensationHint;

    let hint = CompensationHint::UniqueViolation {
        field: "scope".into(),
        conflicting_value: "a".into(),
    };
    let mut engine = CrdtEngine::new(1).unwrap();
    engine.register_field_index("notes", FIELD, false);

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
    assert!(lookup(&engine, "a").is_empty());
    assert_eq!(lookup(&engine, "a_1"), ["d1"]);
    engine.assert_field_indexes_consistent();

    // EscalateToDlq deletes the row.
    engine.set_policy("notes", nodedb_crdt::CollectionPolicy::strict());
    let escalated = engine
        .upsert("notes", "d2", &[("scope", text("a"))])
        .unwrap();
    assert!(matches!(
        engine.reject_delta_with_policy(escalated, &hint),
        Some(nodedb_crdt::PolicyResolution::Escalate { .. })
    ));
    assert!(lookup(&engine, "a").is_empty());
    assert_eq!(lookup(&engine, "a_1"), ["d1"]);
    engine.assert_field_indexes_consistent();
}

#[test]
fn index_tracks_each_field_of_a_collection() {
    let mut engine = CrdtEngine::new(1).unwrap();
    engine.register_field_index("notes", FIELD, false);
    engine.register_field_index("notes", "$.kind", false);
    engine
        .upsert("notes", "d1", &[("scope", text("a")), ("kind", text("x"))])
        .unwrap();

    // Only the second field changes. The first keeps its posting.
    engine
        .upsert("notes", "d1", &[("scope", text("a")), ("kind", text("y"))])
        .unwrap();
    assert_eq!(lookup(&engine, "a"), ["d1"]);
    assert!(
        engine
            .field_index_lookup("notes", "$.kind", "x")
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        engine.field_index_lookup("notes", "$.kind", "y").unwrap(),
        ["d1"]
    );
    engine.assert_field_indexes_consistent();
}
