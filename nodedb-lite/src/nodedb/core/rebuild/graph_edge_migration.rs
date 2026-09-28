// SPDX-License-Identifier: Apache-2.0

//! One-time migration of legacy graph edge CRDT keys.
//!
//! Before the shared edge-preparation helper existed, the batch edge-insert
//! path stored edges under `"{src}--{label}-->{dst}"` CRDT document ids
//! instead of the length-prefixed `EdgeId` `Display` key that
//! `graph_insert_edge` (and the current batch path) both use. Edge delete
//! only ever removes the `EdgeId`-keyed form, so an edge stored under the
//! old key was never removable and resurrected on every CSR rebuild, and its
//! properties were invisible to traversal (which looks properties up by the
//! `EdgeId` key).
//!
//! This migration rewrites every such key, once, guarded by a persisted
//! marker in `Namespace::Meta`. It is also idempotent by construction: a key
//! already in `EdgeId` `Display` form parses successfully and is left
//! untouched.
//!
//! The rewrite for one edge is two CRDT ops on two different document ids
//! (an upsert under the new key, a delete of the old key) — the CRDT engine
//! has no primitive that commits an upsert and a delete of two different
//! rows as a single atomic delta, so a crash between them is possible. The
//! upserts of every migrated edge's new key are issued first, as one
//! `batch_upsert` (one delta per row) call; the old keys are deleted only
//! after every upsert has succeeded. A crash after some deletes but before
//! others leaves the corresponding old keys still present — never a lost
//! edge, since their new-key upsert already landed — and the next open's
//! scan finds those old keys again (they still fail `EdgeId::from_str`),
//! redoes the now-idempotent upsert, and retries the delete. The marker is
//! written only once every upsert and every delete for this run has
//! succeeded.

use nodedb_types::Namespace;
use nodedb_types::document::Document;
use nodedb_types::id::{EdgeId, NodeId};

use crate::engine::crdt::engine::CrdtBatchOp;
use crate::engine::graph::edge::{edge_crdt_fields, edge_id_for};
use crate::error::LiteError;
use crate::nodedb::LockExt;
use crate::nodedb::convert::loro_value_to_document;
use crate::nodedb::core::types::NodeDbLite;
use crate::storage::engine::StorageEngine;

/// Meta key marking that the legacy edge-key migration has completed.
const META_LEGACY_EDGE_KEYS_MIGRATED: &[u8] = b"meta:legacy_edge_keys_migrated";

/// One legacy key rewritten to its `EdgeId` form: the CRDT collection it
/// lives in, the old (legacy) document id, the new (`EdgeId` `Display`) id,
/// and the CRDT field list to upsert under the new id — owned, so it
/// outlives the read-only scan pass that discovers it.
struct LegacyEdgeRewrite {
    collection: String,
    old_key: String,
    new_key: String,
    fields: Vec<(String, loro::LoroValue)>,
}

impl<S: StorageEngine> NodeDbLite<S> {
    /// Rewrite every legacy `"{src}--{label}-->{dst}"` edge CRDT key to its
    /// `EdgeId` `Display` form, across every `__edges__*` collection.
    ///
    /// Skips entirely once the `Namespace::Meta` marker is set. Safe to call
    /// unconditionally on every open: idempotent, and a no-op scan once
    /// migrated.
    ///
    /// The marker is written only after a scan that actually examined at
    /// least one `__edges__*` collection, and only after every rewrite in
    /// this run succeeded. A database with no graph collections yet has
    /// verified nothing, so it keeps rescanning (cheap: there is nothing to
    /// scan) rather than latch "done" before any real edge data has ever
    /// been checked.
    pub(crate) async fn migrate_legacy_edge_keys(&self) -> Result<(), LiteError> {
        let already_done = self
            .storage
            .get(Namespace::Meta, META_LEGACY_EDGE_KEYS_MIGRATED)
            .await?
            .is_some();
        if already_done {
            return Ok(());
        }

        // Pass 1: find every legacy-keyed edge and build its rewrite, under
        // one read-only pass over the CRDT state. Nothing is mutated yet.
        let mut rewrites: Vec<LegacyEdgeRewrite> = Vec::new();
        let mut saw_edge_collection = false;
        {
            let crdt = self.crdt.lock_or_recover();
            let collections = crdt.collection_names();
            for crdt_coll in &collections {
                if !crdt_coll.starts_with("__edges__") {
                    continue;
                }
                saw_edge_collection = true;
                for id in crdt.list_ids(crdt_coll) {
                    // Already in EdgeId Display form: nothing to migrate.
                    if id.parse::<EdgeId>().is_ok() {
                        continue;
                    }
                    let Some(loro_val) = crdt.read(crdt_coll, &id) else {
                        continue;
                    };
                    let doc = loro_value_to_document(&id, &loro_val);
                    let (Some(src), Some(dst), Some(label)) = (
                        doc.get_str("src").map(str::to_owned),
                        doc.get_str("dst").map(str::to_owned),
                        doc.get_str("label").map(str::to_owned),
                    ) else {
                        // Not a recognizable legacy edge document: leave it.
                        continue;
                    };
                    let from = NodeId::from_validated(src);
                    let to = NodeId::from_validated(dst);
                    let Ok(edge_id) = edge_id_for(&from, &to, &label) else {
                        continue;
                    };
                    let new_key = format!("{edge_id}");
                    if new_key == id {
                        continue;
                    }

                    let mut props = Document::new(new_key.clone());
                    for (k, v) in &doc.fields {
                        if k != "src" && k != "dst" && k != "label" {
                            props.fields.insert(k.clone(), v.clone());
                        }
                    }
                    let properties = if props.fields.is_empty() {
                        None
                    } else {
                        Some(props)
                    };
                    // Own the field list immediately: `from`/`to`/`label`
                    // (and the borrows `edge_crdt_fields` hands back into
                    // them) don't outlive this loop iteration, but the
                    // rewrite must survive into the mutation passes below.
                    let fields: Vec<(String, loro::LoroValue)> =
                        edge_crdt_fields(&from, &to, &label, &properties)
                            .into_iter()
                            .map(|(k, v)| (k.to_owned(), v))
                            .collect();

                    rewrites.push(LegacyEdgeRewrite {
                        collection: crdt_coll.clone(),
                        old_key: id,
                        new_key,
                        fields,
                    });
                }
            }
        }

        if rewrites.is_empty() {
            if saw_edge_collection {
                self.storage
                    .put(Namespace::Meta, META_LEGACY_EDGE_KEYS_MIGRATED, &[1u8])
                    .await?;
            }
            return Ok(());
        }

        // Pass 2: upsert every new key as one batch_upsert (one delta per
        // row) call, propagating any failure instead of leaving some edges
        // migrated and others silently not.
        {
            let mut crdt = self.crdt.lock_or_recover();
            let field_refs: Vec<Vec<(&str, loro::LoroValue)>> = rewrites
                .iter()
                .map(|r| {
                    r.fields
                        .iter()
                        .map(|(k, v)| (k.as_str(), v.clone()))
                        .collect()
                })
                .collect();
            let ops: Vec<CrdtBatchOp<'_>> = rewrites
                .iter()
                .zip(field_refs.iter())
                .map(|(r, fields)| (r.collection.as_str(), r.new_key.as_str(), fields.as_slice()))
                .collect();
            crdt.batch_upsert(&ops)?;
        }

        // Pass 3: delete every old key now that its new key is durable. A
        // failure here still propagates (failing the open), but every
        // upsert from pass 2 already landed, so no edge is lost — the next
        // open's scan finds the still-present old keys and retries.
        {
            let mut crdt = self.crdt.lock_or_recover();
            for r in &rewrites {
                crdt.delete(&r.collection, &r.old_key)?;
            }
        }

        if saw_edge_collection {
            self.storage
                .put(Namespace::Meta, META_LEGACY_EDGE_KEYS_MIGRATED, &[1u8])
                .await?;
        }
        Ok(())
    }
}
