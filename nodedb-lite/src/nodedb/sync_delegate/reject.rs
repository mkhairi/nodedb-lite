//! Reacting to Origin's refusal of a pushed delta.

use nodedb_types::sync::compensation::CompensationHint;

use crate::nodedb::core::NodeDbLite;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

/// The constraint Origin names when a delta's Loro peer id belongs to another
/// replica. It is not an application constraint: no policy can resolve it,
/// because nothing about the row is wrong.
const PEER_ID_COLLISION: &str = "peer_id_collision";

/// Whether a refusal says this replica's producer identity is unusable rather
/// than that its data is.
pub(super) fn is_peer_id_collision(hint: &CompensationHint) -> bool {
    matches!(hint, CompensationHint::Custom { constraint, .. } if constraint == PEER_ID_COLLISION)
}

/// Whether Origin's refusal is a verdict on the row itself.
///
/// Only these four say the write is unacceptable: a value that collides, a
/// parent row that does not exist, a field the schema forbids, bytes that did
/// not survive the wire. Policy resolution answers exactly that question — it
/// renames, defers, or gives up and deletes the local row — so only these may
/// reach it.
///
/// Everything else is a verdict on the *session*: a grant this principal has
/// not been given, a rate limit, a collection Origin has not materialized yet,
/// a refusal a future Origin invents that this build cannot name. Nothing about
/// the row is wrong, and the identical bytes are expected to land once the
/// condition clears. Sending those through policy resolved them as
/// `Escalate` — which deletes the local row and drops the delta — so a missing
/// grant destroyed the write on both sides at once, and did it while the queue
/// drained to zero. Defaulting the unknown case to "the data is bad" is what
/// made that possible, so the default here is the other way round.
fn refusal_is_about_the_row(hint: &CompensationHint) -> bool {
    matches!(
        hint,
        CompensationHint::UniqueViolation { .. }
            | CompensationHint::ForeignKeyMissing { .. }
            | CompensationHint::SchemaViolation { .. }
            | CompensationHint::IntegrityViolation
    )
}

pub(super) fn handle_reject_with_policy_impl<S: StorageEngine>(
    db: &NodeDbLite<S>,
    mutation_id: u64,
    hint: &CompensationHint,
) {
    // A peer-id collision never reaches the policy path. Policy resolution
    // treats an unrecognised refusal as a constraint the row violated: it
    // deletes the local row and drops the delta. Here the row is valid and
    // Origin never saw it — only the identity it travelled under was refused —
    // so that resolution would destroy the write and leave the replica pushing
    // the next one under the same refused id. Recovery is handled by
    // `rotate_peer_id`, which re-queues this delta with the rest.
    if is_peer_id_collision(hint) {
        tracing::warn!(
            mutation_id,
            "Origin refused this delta's Loro peer id as another replica's — \
             rotating the local peer id and resyncing"
        );
        return;
    }

    if !refusal_is_about_the_row(hint) {
        let mut crdt = db.crdt.lock_or_recover();
        crdt.mark_delta_blocked(mutation_id);
        let blocked = crdt.blocked_delta_count();
        drop(crdt);
        tracing::error!(
            mutation_id,
            hint = %hint,
            blocked_deltas = blocked,
            "Origin refused this delta for a reason that is not about the row; \
             keeping it queued and keeping the local row — replication is stalled \
             until the condition clears (health reports blocked_deltas)"
        );
        return;
    }

    let mut crdt = db.crdt.lock_or_recover();
    match crdt.reject_delta_with_policy(mutation_id, hint) {
        Some(nodedb_crdt::PolicyResolution::AutoResolved(action)) => {
            tracing::info!(
                mutation_id,
                action = ?action,
                "SyncDelegate: delta auto-resolved by policy"
            );
        }
        Some(nodedb_crdt::PolicyResolution::Deferred {
            retry_after_ms,
            attempt,
            ..
        }) => {
            tracing::info!(
                mutation_id,
                retry_after_ms,
                attempt,
                "SyncDelegate: delta deferred for retry"
            );
        }
        Some(nodedb_crdt::PolicyResolution::Escalate { .. }) => {
            tracing::warn!(mutation_id, "SyncDelegate: delta escalated to DLQ (policy)");
        }
        Some(nodedb_crdt::PolicyResolution::WebhookRequired { webhook_url, .. }) => {
            tracing::warn!(
                mutation_id,
                webhook_url,
                "SyncDelegate: delta requires webhook (not supported on Lite)"
            );
            let _ = crdt.reject_delta(mutation_id);
        }
        None => {
            tracing::debug!(
                mutation_id,
                "SyncDelegate: reject_with_policy — delta not found"
            );
        }
    }
}
