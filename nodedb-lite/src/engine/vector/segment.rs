// SPDX-License-Identifier: Apache-2.0

//! Vector segment payloads in node-id order, and the check that a stored
//! segment serves the nodes of the index it is attached to.
//!
//! A segment backs a graph-only checkpoint positionally: slot `i` holds the
//! vector of node `i`. The payload is therefore laid out in node-id order,
//! and each slot's surrogate records the document bound to that node as a
//! [`doc_fingerprint`]. On load, [`segment_serves_index`] compares those
//! fingerprints with the id map, so a segment written for another node
//! order is refused and the index is rebuilt from the durable rows.

use std::collections::HashMap;

use nodedb_vector::segment_backing::VectorSegmentBacking;

use crate::engine::vector::{HnswIndex, IndexIdMap};
use crate::error::LiteError;
use crate::storage::engine::StorageEngine;

/// Where one node's segment vector comes from.
pub(crate) enum NodeSource {
    /// The durable row of the bound document.
    Bound(String),
    /// A node bound to no document (a tombstone): its own vector, kept so
    /// graph traversal through it scores the same after a reload.
    Local(Vec<f32>),
}

/// The surrogate recorded for a node bound to `doc_id`. Never 0, the
/// surrogate of an unbound slot.
pub(crate) fn doc_fingerprint(doc_id: &str) -> u64 {
    // FNV-1a: stable across builds and platforms.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in doc_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash.max(1)
}

/// Where each node of `index` takes its segment vector from, in node-id
/// order, read under the caller's lock on the index map. `None` when a node
/// is bound to no document and its own vector is unreadable.
pub(crate) fn node_sources(index: &HnswIndex, ids: Option<&IndexIdMap>) -> Option<Vec<NodeSource>> {
    (0..index.len() as u32)
        .map(|node| match ids.and_then(|m| m.doc_id(node)) {
            Some(doc_id) => Some(NodeSource::Bound(doc_id.to_owned())),
            None => index
                .get_vector_or_backing(node)
                .map(|v| NodeSource::Local(v.into_owned())),
        })
        .collect()
}

/// The `(dim, vectors, surrogates)` segment payload for `index_key`, one
/// slot per entry of `sources`, in order.
///
/// A bound node's vector is read from its DURABLE row, never from the
/// in-memory index: after a graph-only checkpoint restore the index nodes
/// carry no vector bytes of their own. Returns `None` when there is nothing
/// to publish, or when a bound document has no durable row or a vector of
/// another width. The existing segment is then left alone; it no longer
/// matches the index and is refused on the next load.
pub(crate) async fn segment_payload<S: StorageEngine>(
    storage: &S,
    index_key: &str,
    sources: Vec<NodeSource>,
) -> Result<Option<(usize, Vec<Vec<f32>>, Vec<u64>)>, LiteError> {
    if sources.is_empty() {
        return Ok(None);
    }
    let mut durable: HashMap<String, Vec<f32>> =
        crate::engine::vector::durable::load_collection(storage, index_key)
            .await?
            .into_iter()
            .collect();
    let mut vectors = Vec::with_capacity(sources.len());
    let mut surrogates = Vec::with_capacity(sources.len());
    for source in sources {
        match source {
            NodeSource::Bound(doc_id) => {
                let Some(vector) = durable.remove(&doc_id) else {
                    tracing::warn!(
                        index_key,
                        doc_id,
                        "bound document has no durable vector; segment not written"
                    );
                    return Ok(None);
                };
                surrogates.push(doc_fingerprint(&doc_id));
                vectors.push(vector);
            }
            NodeSource::Local(vector) => {
                surrogates.push(0);
                vectors.push(vector);
            }
        }
    }
    let dim = vectors.first().map_or(0, Vec::len);
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        tracing::warn!(
            index_key,
            "segment vectors differ in width; segment not written"
        );
        return Ok(None);
    }
    Ok(Some((dim, vectors, surrogates)))
}

/// Whether `backing` serves every node of an index of `node_count` nodes:
/// it holds a slot per node, and each bound node's slot carries that node's
/// document fingerprint. With no bindings to compare against, only an empty
/// index is served.
pub(crate) fn segment_serves_index(
    backing: &dyn VectorSegmentBacking,
    node_count: usize,
    ids: Option<&IndexIdMap>,
) -> bool {
    if backing.len() < node_count {
        return false;
    }
    let Some(ids) = ids else {
        return node_count == 0;
    };
    ids.iter().all(|(doc_id, node)| {
        (node as usize) < node_count && backing.get_surrogate(node) == Some(doc_fingerprint(doc_id))
    })
}

/// Attach `backing` to `index` when it serves the index's nodes: one slot
/// per node, each bound node's slot carrying its document's fingerprint, and
/// every slot `index` reads through the backing holding a vector of the
/// index's width. Fails, leaving `index` without the backing, otherwise.
pub(crate) fn attach_verified<B: VectorSegmentBacking + 'static>(
    index: &mut HnswIndex,
    backing: B,
    ids: Option<&IndexIdMap>,
) -> Result<(), LiteError> {
    if !segment_serves_index(&backing, index.len(), ids) {
        return Err(LiteError::Corrupted {
            detail: format!(
                "vector segment of {} slots does not carry the documents bound to the \
                 index's {} nodes in node order",
                backing.len(),
                index.len()
            ),
        });
    }
    index.with_backing(std::sync::Arc::new(backing))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_are_stable_and_never_zero() {
        assert_eq!(doc_fingerprint("a"), doc_fingerprint("a"));
        assert_ne!(doc_fingerprint("a"), doc_fingerprint("b"));
        assert_ne!(doc_fingerprint(""), 0);
    }
}
