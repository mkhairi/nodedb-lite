// SPDX-License-Identifier: Apache-2.0
//! `KvOp` arms with no single-node Lite execution path.

use crate::error::LiteError;

/// `KvOp::ResolveWrite` is the resolve-before-propose wire shape
/// Origin's cross-vshard write path uses to decide a policy once and
/// replay it identically on every replica. Lite is single-node with no
/// Raft replay, so its SQL visitor and CRDT sync resolve every write
/// directly and never emit this variant.
pub(super) fn resolve_write() -> LiteError {
    LiteError::Unsupported {
        detail: "KvOp::ResolveWrite is the resolve-before-propose wire shape \
                 of Origin's cross-vshard write path, which Lite's single-node \
                 engine never emits or needs to interpret"
            .into(),
    }
}

/// `KvOp::ResolvedWrite` replays a decision made by Origin's Raft
/// leader, which has no equivalent on the single-node Lite engine.
pub(super) fn resolved_write() -> LiteError {
    LiteError::Unsupported {
        detail: "KvOp::ResolvedWrite replays a decision made by Origin's Raft \
                 leader, which has no equivalent on the single-node Lite engine"
            .into(),
    }
}

/// `KvOp::PredicateUpdate` carries a WHERE predicate for the Data
/// Plane to resolve against current state. Lite's SQL visitor always
/// resolves WHERE to an explicit key list before building a `KvOp`, so
/// it never constructs this variant.
pub(super) fn predicate_update(collection: &str) -> LiteError {
    LiteError::Unsupported {
        detail: format!(
            "KvOp::PredicateUpdate on {collection}: Lite's SQL visitor always \
             resolves WHERE to an explicit key list before building a KvOp"
        ),
    }
}

/// `KvOp::PredicateDelete` mirrors `PredicateUpdate` — see above.
pub(super) fn predicate_delete(collection: &str) -> LiteError {
    LiteError::Unsupported {
        detail: format!(
            "KvOp::PredicateDelete on {collection}: Lite's SQL visitor always \
             resolves WHERE to an explicit key list before building a KvOp"
        ),
    }
}

/// `KvOp::SortedIndexTxnRead` reads a sorted index through a transaction's
/// staging overlay on an Origin core. Lite executes a transaction directly
/// against its single store and keeps no staging overlay to fold in.
pub(super) fn sorted_index_txn_read(collection: &str, index_name: &str) -> LiteError {
    LiteError::Unsupported {
        detail: format!(
            "KvOp::SortedIndexTxnRead on {collection} index {index_name}: Lite keeps \
             no per-transaction staging overlay for a sorted-index read to fold in"
        ),
    }
}

/// A `KvOp::Put` or `KvOp::Delete` carrying sync provenance is a Lite KV
/// push arriving at Origin, gated by Origin's sync idempotency table. Lite
/// is the pushing side and has no such gate. `None` passes.
pub(super) fn refuse_sync_provenance(
    op: &str,
    provenance: &Option<nodedb_types::sync::wire::SyncProvenance>,
) -> Result<(), LiteError> {
    match provenance {
        None => Ok(()),
        Some(_) => Err(LiteError::Unsupported {
            detail: format!(
                "{op} with sync provenance is a Lite KV push that only Origin's sync \
                 idempotency gate admits; Lite never receives one"
            ),
        }),
    }
}

/// A typed [`KvCounterShape`] asks an absent key to become a typed row built
/// from the catalog's column template. Lite stores a KV counter as a bare
/// msgpack number and has no typed-row counter path.
///
/// [`KvCounterShape`]: nodedb_physical::physical_plan::KvCounterShape
pub(super) fn typed_counter_shape(op: &str, collection: &str) -> LiteError {
    LiteError::Unsupported {
        detail: format!(
            "{op} on {collection} with a typed counter shape: Lite stores a KV \
             counter as a bare number and cannot create a typed row for it"
        ),
    }
}
