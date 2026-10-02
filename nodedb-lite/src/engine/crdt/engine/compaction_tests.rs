// SPDX-License-Identifier: BUSL-1.1

//! History compaction against what is already on disk.
//!
//! Compaction replaces a document with a shallow snapshot at its frontier. A
//! collection whose disk form is current at that frontier keeps its marks, so
//! the next flush writes only what changed after it. These assert that the
//! update written then still replays onto the base written before, and that
//! compaction neither plans a rewrite nor runs ahead of an unflushed write.

use std::collections::BTreeMap;

use loro::LoroValue;
use nodedb_crdt::CrdtState;

use super::types::CrdtEngine;
use super::{CrdtWrite, CrdtWriteKind};

const COLLECTION: &str = "items";

/// Run one flush against the engine: plan, then acknowledge every write as
/// committed. Returns the writes so a test can inspect or replay their bytes.
fn flush(engine: &mut CrdtEngine) -> Vec<CrdtWrite> {
    let plan = engine.plan_persistence().unwrap();
    engine.mark_persisted(plan.iter().map(CrdtWrite::persisted));
    plan
}

/// Every row of `collection` with its full value, in id order.
fn rows(state: &CrdtState, collection: &str) -> BTreeMap<String, Option<LoroValue>> {
    state
        .row_ids(collection)
        .into_iter()
        .map(|id| {
            let value = state.read_row(collection, &id);
            (id, value)
        })
        .collect()
}

fn upsert(engine: &mut CrdtEngine, id: &str, val: i64, name: &str) {
    engine
        .upsert(
            COLLECTION,
            id,
            &[
                ("val", LoroValue::I64(val)),
                ("name", LoroValue::String(name.into())),
            ],
        )
        .unwrap();
}

/// The assumption the kept-marks path rests on.
///
/// Disk holds a base and the updates on top of it, persisted up to frontier F.
/// Compaction replaces the in-memory document with a shallow snapshot at F.
/// The next flush exports an update from F out of that shallow document. A
/// restore imports base, then every update, into a fresh document. That must
/// equal the live document in every row and field, at the same version.
///
/// This tests Loro, not engine bookkeeping: the post-compaction update comes
/// from `export_delta_from`, not from a flush plan. The flush path is covered
/// by `compacting_a_flushed_collection_plans_no_rewrite`. If this test fails,
/// keeping the marks across compaction is unsound.
#[test]
fn delta_exported_after_compaction_replays_onto_the_persisted_base() {
    const PEER: u64 = 3;
    let mut engine = CrdtEngine::new(PEER).unwrap();

    // History worth compacting: rows, overwrites, a delete.
    for i in 0..20 {
        upsert(&mut engine, &format!("i{i}"), i, &format!("row {i}"));
    }
    for i in 0..10 {
        upsert(&mut engine, &format!("i{i}"), i * 100, "overwritten");
    }
    engine.delete(COLLECTION, "i19").unwrap();

    let base_plan = flush(&mut engine);
    assert_eq!(base_plan.len(), 1);
    assert!(matches!(
        base_plan[0].kind,
        CrdtWriteKind::Checkpoint { .. }
    ));
    let base = base_plan[0].bytes.clone();

    // One update on top of the base before compaction, as a live store has.
    upsert(&mut engine, "i5", 555, "before compaction");
    let pre_plan = flush(&mut engine);
    assert_eq!(pre_plan.len(), 1);
    assert!(matches!(pre_plan[0].kind, CrdtWriteKind::Delta { seq: 0 }));
    let pre_delta = pre_plan[0].bytes.clone();

    let persisted = engine.state(COLLECTION).unwrap().oplog_version_vector();
    let epoch_before = engine.state_epoch(COLLECTION);
    engine.compact_history().unwrap();
    assert!(
        engine.state_epoch(COLLECTION) > epoch_before,
        "the collection must actually have been compacted for this to test anything"
    );

    // After compaction: overwrite rows written before it, add rows, delete a
    // pre-compaction row, and bring back the row deleted before it.
    upsert(&mut engine, "i0", -1, "after compaction");
    upsert(&mut engine, "i12", -12, "after compaction");
    upsert(&mut engine, "i20", 20, "new after compaction");
    upsert(&mut engine, "i19", 19, "recreated after compaction");
    engine.delete(COLLECTION, "i7").unwrap();

    let post_delta = engine.export_delta_from(COLLECTION, &persisted).unwrap();
    assert!(!post_delta.is_empty());

    // Restore imports through `import_local`, base first, updates in order.
    let restored = CrdtState::new(CrdtEngine::collection_peer_id(PEER, COLLECTION)).unwrap();
    restored.import_local(&base).unwrap();
    restored.import_local(&pre_delta).unwrap();
    restored.import_local(&post_delta).unwrap();

    let live = engine.state(COLLECTION).unwrap();
    assert_eq!(
        rows(&restored, COLLECTION),
        rows(live, COLLECTION),
        "base + updates on disk must replay to exactly the live document"
    );
    assert_eq!(
        restored.oplog_version_vector(),
        live.oplog_version_vector(),
        "the replay must reach the live frontier, or a later update has no base"
    );
}

/// Compacting a collection whose disk form is current plans no write at all.
///
/// Dropping its marks instead makes the next flush export and commit the whole
/// collection as a fresh checkpoint. Across every compacted collection that is
/// one commit of hundreds of MB, more than the file reuses per commit.
///
/// This drives `plan_persistence` after compaction. It then replays the bytes
/// of the next flush onto the base written before compaction.
#[test]
fn compacting_a_flushed_collection_plans_no_rewrite() {
    const PEER: u64 = 1;
    let mut engine = CrdtEngine::new(PEER).unwrap();
    for i in 0..50 {
        upsert(&mut engine, &format!("i{i}"), i, "row");
    }
    let base_plan = flush(&mut engine);
    assert_eq!(base_plan.len(), 1);
    assert!(matches!(
        base_plan[0].kind,
        CrdtWriteKind::Checkpoint { .. }
    ));
    let base = base_plan[0].bytes.clone();

    let epoch_before = engine.state_epoch(COLLECTION);
    engine.compact_history().unwrap();
    assert!(
        engine.state_epoch(COLLECTION) > epoch_before,
        "a flushed collection with new history must still be compacted"
    );

    let exports_before = engine.snapshot_export_count();
    let replan = engine.plan_persistence().unwrap();
    assert!(
        replan.is_empty(),
        "the disk form already replays to the compacted state; compaction must plan nothing, \
         planned {} write(s)",
        replan.len()
    );
    assert_eq!(engine.snapshot_export_count(), exports_before);

    upsert(&mut engine, "i0", 1000, "after compaction");
    let plan = flush(&mut engine);
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(plan[0].kind, CrdtWriteKind::Delta { seq: 0 }),
        "one write after compaction must cost an update, not a checkpoint: {:?}",
        plan[0].kind
    );

    // The update that flush wrote applies on the base written before.
    let restored = CrdtState::new(CrdtEngine::collection_peer_id(PEER, COLLECTION)).unwrap();
    restored.import_local(&base).unwrap();
    restored.import_local(&plan[0].bytes).unwrap();
    assert_eq!(
        rows(&restored, COLLECTION),
        rows(engine.state(COLLECTION).unwrap(), COLLECTION)
    );
}

/// A collection with operations not yet on disk is left for a later call.
///
/// Its next update is exported from the persisted frontier, which lies behind
/// the current one. Compaction at the current frontier can discard history the
/// update needs, so the collection waits for the flush that catches it up.
#[test]
fn compacting_an_unflushed_collection_is_deferred() {
    let mut engine = CrdtEngine::new(1).unwrap();
    for i in 0..10 {
        upsert(&mut engine, &format!("i{i}"), i, "row");
    }
    flush(&mut engine);

    // Written, not flushed.
    upsert(&mut engine, "i10", 10, "unflushed");

    let flushed = engine.flushed_versions.get(COLLECTION).cloned();
    let checkpoint_bytes = engine.checkpoint_bytes.get(COLLECTION).copied();
    let delta_bytes = engine.delta_bytes.get(COLLECTION).copied();
    let next_delta_seq = engine.next_delta_seq.get(COLLECTION).copied();
    let epoch_before = engine.state_epoch(COLLECTION);

    engine.compact_history().unwrap();

    assert_eq!(
        engine.state_epoch(COLLECTION),
        epoch_before,
        "a collection with unflushed operations must not be compacted"
    );
    assert!(
        !engine.compacted_versions.contains_key(COLLECTION),
        "a deferred collection must not be recorded as compacted, or it is never retried"
    );
    assert!(flushed.is_some());
    assert_eq!(engine.flushed_versions.get(COLLECTION), flushed.as_ref());
    assert_eq!(
        engine.checkpoint_bytes.get(COLLECTION).copied(),
        checkpoint_bytes
    );
    assert_eq!(engine.delta_bytes.get(COLLECTION).copied(), delta_bytes);
    assert_eq!(
        engine.next_delta_seq.get(COLLECTION).copied(),
        next_delta_seq
    );

    // The flush that catches it up is an update, not a checkpoint.
    let plan = flush(&mut engine);
    assert_eq!(plan.len(), 1);
    assert!(matches!(plan[0].kind, CrdtWriteKind::Delta { seq: 0 }));

    // Now current on disk, so the next call compacts it.
    engine.compact_history().unwrap();
    let frontier = engine.state(COLLECTION).unwrap().oplog_version_vector();
    assert!(engine.state_epoch(COLLECTION) > epoch_before);
    assert_eq!(engine.compacted_versions.get(COLLECTION), Some(&frontier));
    assert!(
        engine.plan_persistence().unwrap().is_empty(),
        "compacting it once caught up must plan nothing"
    );
}

/// A collection is compacted again only once it has taken `min_ops`
/// operations since its last compaction. Below that the pass leaves it whole
/// and counts it as skipped, so a busy store does not rebuild a large document
/// on every tick to discard a handful of operations.
#[test]
fn compaction_waits_for_min_ops_since_the_last_one() {
    let mut engine = CrdtEngine::new(1).unwrap();
    for i in 0..10 {
        upsert(&mut engine, &format!("i{i}"), i, "row");
    }
    flush(&mut engine);

    let first = engine.compact_history_min_ops(1).unwrap();
    assert_eq!((first.compacted, first.deferred, first.skipped), (1, 0, 0));
    assert!(first.ops_discarded > 0);
    let epoch = engine.state_epoch(COLLECTION);

    // Operations are counted from the version vector, not assumed per upsert.
    let ops = |e: &CrdtEngine| -> u64 {
        let vv = e.state(COLLECTION).unwrap().oplog_version_vector();
        vv.values().map(|end| u64::try_from(*end).unwrap()).sum()
    };
    let at_compaction = ops(&engine);
    upsert(&mut engine, "i10", 10, "a");
    upsert(&mut engine, "i11", 11, "b");
    flush(&mut engine);
    let taken = ops(&engine) - at_compaction;
    let below = engine.compact_history_min_ops(taken + 1).unwrap();
    assert_eq!((below.compacted, below.deferred, below.skipped), (0, 0, 1));
    assert_eq!(below.ops_discarded, 0);
    assert_eq!(
        engine.state_epoch(COLLECTION),
        epoch,
        "a collection below the threshold must not be compacted"
    );

    // One more write reaches it, counted from the last compaction rather
    // than from the skipped pass.
    upsert(&mut engine, "i12", 12, "c");
    flush(&mut engine);
    let reached = engine.compact_history_min_ops(taken + 1).unwrap();
    assert_eq!(
        (reached.compacted, reached.deferred, reached.skipped),
        (1, 0, 0)
    );
    assert_eq!(reached.ops_discarded, ops(&engine) - at_compaction);
    assert!(engine.state_epoch(COLLECTION) > epoch);
}

/// The report counts a collection waiting on a flush as deferred, not
/// skipped, whatever the threshold.
#[test]
fn compaction_report_counts_deferred_collections() {
    let mut engine = CrdtEngine::new(1).unwrap();
    upsert(&mut engine, "i0", 0, "row");
    flush(&mut engine);
    upsert(&mut engine, "i1", 1, "unflushed");

    let report = engine.compact_history_min_ops(1).unwrap();
    assert_eq!(
        (report.compacted, report.deferred, report.skipped),
        (0, 1, 0)
    );
}
