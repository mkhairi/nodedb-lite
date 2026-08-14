//! Compensation handling simulation tests — no Origin required.
//!
//! Tests the full compensation flow: delta rejected → rollback → callback.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use nodedb_client::NodeDb;
use nodedb_lite::engine::crdt::engine::PendingDelta;
use nodedb_lite::sync::*;
use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use nodedb_types::document::Document;
use nodedb_types::sync::compensation::CompensationHint;
use nodedb_types::sync::wire::*;
use nodedb_types::value::Value;

async fn open_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let s = PagedbStorageMem::open_in_memory().await.unwrap();
    NodeDbLite::open(s).await.unwrap()
}

/// The refusal Origin sends when a delta's Loro peer id is owned by another
/// replica: a terminal reject carrying the `peer_id_collision` constraint.
fn peer_id_collision_hint() -> CompensationHint {
    CompensationHint::Custom {
        constraint: CompensationHint::PEER_ID_COLLISION.into(),
        detail: "PEER_ID_COLLISION: peer id 1 on collection 'users' is already owned by \
                 another replica; generate a new peer id and resync"
            .into(),
    }
}

#[tokio::test]
async fn reject_unique_violation_rolls_back_document() {
    let db = open_db().await;

    // Write a document.
    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();

    // Verify it exists.
    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_some()
    );

    // Get the mutation ID.
    let deltas = db.pending_crdt_deltas().unwrap();
    let mid = deltas[0].mutation_id;

    // Simulate Origin rejection.
    db.reject_delta(mid).unwrap();

    // Document should be rolled back.
    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_none(),
        "rejected document should not exist after rollback"
    );
}

#[tokio::test]
async fn compensation_handler_receives_typed_hint() {
    let count = Arc::new(AtomicU32::new(0));
    let last_code = Arc::new(std::sync::Mutex::new(String::new()));

    let count_c = count.clone();
    let code_c = last_code.clone();

    let client = Arc::new(SyncClient::new(SyncConfig::new(
        "wss://localhost/sync",
        "jwt",
    )));

    client.set_compensation_handler(Arc::new(move |event: CompensationEvent| {
        count_c.fetch_add(1, Ordering::Relaxed);
        *code_c.lock().unwrap() = event.hint.code().to_string();
    }));

    // Simulate different rejection types.
    client
        .handle_delta_reject(&DeltaRejectMsg {
            mutation_id: 1,
            reason: "unique".into(),
            compensation: Some(CompensationHint::UniqueViolation {
                field: "email".into(),
                conflicting_value: "a@b.com".into(),
            }),
        })
        .await;
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert_eq!(*last_code.lock().unwrap(), "UNIQUE_VIOLATION");

    client
        .handle_delta_reject(&DeltaRejectMsg {
            mutation_id: 2,
            reason: "fk".into(),
            compensation: Some(CompensationHint::ForeignKeyMissing {
                referenced_id: "user-999".into(),
            }),
        })
        .await;
    assert_eq!(count.load(Ordering::Relaxed), 2);
    assert_eq!(*last_code.lock().unwrap(), "FK_MISSING");

    client
        .handle_delta_reject(&DeltaRejectMsg {
            mutation_id: 3,
            reason: "rls".into(),
            compensation: Some(CompensationHint::PermissionDenied),
        })
        .await;
    assert_eq!(count.load(Ordering::Relaxed), 3);
    assert_eq!(*last_code.lock().unwrap(), "PERMISSION_DENIED");

    client
        .handle_delta_reject(&DeltaRejectMsg {
            mutation_id: 4,
            reason: "rate".into(),
            compensation: Some(CompensationHint::RateLimited {
                retry_after_ms: 5000,
            }),
        })
        .await;
    assert_eq!(count.load(Ordering::Relaxed), 4);
    assert_eq!(*last_code.lock().unwrap(), "RATE_LIMITED");
}

#[tokio::test]
async fn peer_id_collision_is_not_resolved_by_conflict_policy() {
    let db = open_db().await;

    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();
    let mid = db.pending_crdt_deltas().unwrap()[0].mutation_id;

    SyncDelegate::reject_with_policy(&*db, mid, &peer_id_collision_hint());

    // Nothing about the row was refused — only the identity it travelled
    // under. Resolving it as a constraint violation deletes the row and drops
    // the delta, turning an identity fault into silent data loss.
    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_some(),
        "a peer-id collision must not roll back the local row"
    );
    assert!(
        db.pending_crdt_deltas()
            .unwrap()
            .iter()
            .any(|d| d.document_id == "user-alice"),
        "a peer-id collision must not discard the refused write"
    );
}

#[tokio::test]
async fn rotating_the_peer_id_adopts_a_new_identity() {
    let db = open_db().await;
    let before = db.peer_id();

    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();

    SyncDelegate::rotate_peer_id(&*db).await;

    let after = db.peer_id();
    assert_ne!(
        after, before,
        "keeping the refused peer id makes every later write refusable forever"
    );
    assert_ne!(after, 0, "a rotated peer id must be a usable Loro peer id");
}

#[tokio::test]
async fn rotating_the_peer_id_preserves_and_requeues_local_writes() {
    let db = open_db().await;

    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();

    SyncDelegate::rotate_peer_id(&*db).await;

    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_some(),
        "the rotation must carry local rows onto the new identity"
    );
    // Origin has never seen the rebuilt document: its operations are authored
    // by an id Origin has no history for. Rotation is only a recovery if the
    // rows go out again under it.
    assert!(
        db.pending_crdt_deltas()
            .unwrap()
            .iter()
            .any(|d| d.collection == "users" && d.document_id == "user-alice"),
        "every row must be queued for re-push after a rotation"
    );
}

#[tokio::test]
async fn compensation_event_names_the_rejected_collection_and_document() {
    let client = Arc::new(SyncClient::new(SyncConfig::new(
        "wss://localhost/sync",
        "jwt",
    )));

    let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let seen_c = seen.clone();
    client.set_compensation_handler(Arc::new(move |event: CompensationEvent| {
        seen_c
            .lock()
            .unwrap()
            .push((event.collection.clone(), event.document_id.clone()));
    }));

    client
        .build_delta_pushes(
            &[PendingDelta {
                mutation_id: 7,
                collection: "orders".into(),
                document_id: "o1".into(),
                delta_bytes: vec![1, 2, 3],
                seq: 0,
            }],
            42,
        )
        .await;

    client
        .handle_delta_reject(&DeltaRejectMsg {
            mutation_id: 7,
            reason: "unique".into(),
            compensation: Some(CompensationHint::UniqueViolation {
                field: "sku".into(),
                conflicting_value: "abc".into(),
            }),
        })
        .await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one rejection, one compensation event");
    // An event that names neither target is undispatchable: the application
    // cannot roll back, retry, or prompt without knowing what was refused.
    assert!(
        !seen[0].0.is_empty() && !seen[0].1.is_empty(),
        "compensation event must carry its target, got {:?}",
        seen[0]
    );
    assert_eq!(seen[0], ("orders".to_string(), "o1".to_string()));
}

#[tokio::test]
async fn buffered_compensations_drain_to_late_handler() {
    let registry = CompensationRegistry::new();

    // Dispatch before handler is set — should buffer.
    registry.dispatch(CompensationEvent {
        mutation_id: 1,
        collection: "users".into(),
        document_id: "u1".into(),
        hint: CompensationHint::UniqueViolation {
            field: "email".into(),
            conflicting_value: "a@b.com".into(),
        },
    });
    registry.dispatch(CompensationEvent {
        mutation_id: 2,
        collection: "users".into(),
        document_id: "u2".into(),
        hint: CompensationHint::PermissionDenied,
    });

    assert_eq!(registry.buffered_count(), 2);

    // Set handler — should drain buffer.
    let count = Arc::new(AtomicU32::new(0));
    let count_c = count.clone();
    registry.set_handler(Arc::new(move |_: CompensationEvent| {
        count_c.fetch_add(1, Ordering::Relaxed);
    }));

    assert_eq!(count.load(Ordering::Relaxed), 2);
    assert_eq!(registry.buffered_count(), 0);
}

/// NDB-AQL-29: a JWT-authenticated replica whose principal has no write grant
/// is refused on every delta. Resolving that through the conflict policy
/// deleted the local row and dropped the delta, so the replica lost the write
/// Origin had never taken — while its queue drained to zero, which is the one
/// number an operator reads as "replication healthy".
#[tokio::test]
async fn permission_denied_keeps_the_row_and_the_queued_delta() {
    let db = open_db().await;

    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();
    let mid = db.pending_crdt_deltas().unwrap()[0].mutation_id;

    SyncDelegate::reject_with_policy(&*db, mid, &CompensationHint::PermissionDenied);

    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_some(),
        "a missing grant says nothing about the row; deleting it destroys the write"
    );
    assert!(
        db.pending_crdt_deltas()
            .unwrap()
            .iter()
            .any(|d| d.document_id == "user-alice"),
        "the delta must stay queued: a grant added at the Origin makes it land"
    );

    let health = db.health();
    assert_eq!(
        health.engines.pending_deltas, 1,
        "a refused delta is still queued, so it is still pending"
    );
    assert_eq!(
        health.engines.blocked_deltas, 1,
        "health must distinguish a stalled queue from a backlogged one"
    );
    assert_eq!(
        health.engines.dropped_writes, 0,
        "nothing was thrown away, so nothing may be reported as lost"
    );
}

/// The refusal Origin sends when the collection has not been materialized yet
/// — a startup race on a fresh Origin, not a verdict on the row.
#[tokio::test]
async fn an_unrecognised_refusal_retains_the_write_rather_than_destroying_it() {
    let db = open_db().await;

    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();
    let mid = db.pending_crdt_deltas().unwrap()[0].mutation_id;

    SyncDelegate::reject_with_policy(
        &*db,
        mid,
        &CompensationHint::Custom {
            constraint: "collection_not_found".into(),
            detail: "collection 'users' does not exist".into(),
        },
    );

    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_some(),
        "an unclassified refusal must not be read as 'the data is bad'"
    );
    assert_eq!(db.health().engines.blocked_deltas, 1);
}

/// The other half of the same rule: a refusal that IS about the row still
/// compensates. Retaining those would stall the queue forever behind a write
/// Origin can never take.
#[tokio::test]
async fn a_row_level_refusal_still_compensates_and_is_counted_as_a_loss() {
    let db = open_db().await;

    let mut doc = Document::new("user-alice");
    doc.set("username", Value::String("alice".into()));
    db.document_put("users", doc).await.unwrap();
    let mid = db.pending_crdt_deltas().unwrap()[0].mutation_id;

    // Corrupt bytes are the one refusal with no resolution but deletion.
    SyncDelegate::reject_with_policy(&*db, mid, &CompensationHint::IntegrityViolation);

    let health = db.health();
    assert_eq!(
        health.engines.blocked_deltas, 0,
        "a row-level refusal is resolved, not blocked"
    );
    assert_eq!(
        health.engines.dropped_writes, 1,
        "the write left the queue without applying anywhere; health must say so"
    );
    assert!(
        db.document_get("users", "user-alice")
            .await
            .unwrap()
            .is_none(),
        "an integrity violation still rolls the row back"
    );
}

/// A blocked collection stalls at the refusal instead of being re-pushed on
/// every 100 ms tick — and the collections behind it keep flowing.
#[tokio::test]
async fn a_blocked_collection_is_held_back_but_others_keep_flowing() {
    let db = open_db().await;

    let mut blocked = Document::new("u1");
    blocked.set("username", Value::String("alice".into()));
    db.document_put("users", blocked).await.unwrap();
    let mut other = Document::new("o1");
    other.set("total", Value::String("1".into()));
    db.document_put("orders", other).await.unwrap();

    let mid = db
        .pending_crdt_deltas()
        .unwrap()
        .iter()
        .find(|d| d.collection == "users")
        .expect("users delta queued")
        .mutation_id;
    SyncDelegate::reject_with_policy(&*db, mid, &CompensationHint::PermissionDenied);

    let pushable = SyncDelegate::pending_deltas(&*db);
    assert!(
        pushable.iter().all(|d| d.collection != "users"),
        "a refused delta re-sent every tick is a hot loop against a refusal that has not changed"
    );
    assert!(
        pushable.iter().any(|d| d.collection == "orders"),
        "one stalled collection must not stall the rest"
    );

    // A new session is where the refusal may have been lifted.
    SyncDelegate::clear_blocked_deltas(&*db);
    assert!(
        SyncDelegate::pending_deltas(&*db)
            .iter()
            .any(|d| d.collection == "users"),
        "a reconnect must re-attempt what a grant may now allow"
    );
}
