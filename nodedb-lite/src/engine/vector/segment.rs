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

/// Build segment vectors in node-id order from durable bound rows.
///
/// Empty sources return `None`. Missing rows and invalid widths return corruption errors.
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
                let vector = durable.remove(&doc_id).ok_or_else(|| LiteError::Corrupted {
                    detail: format!("vector segment '{index_key}' has no durable row for '{doc_id}': restore the vector row"),
                })?;
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
    if dim == 0 {
        return Err(LiteError::Corrupted {
            detail: format!(
                "vector segment '{index_key}' has zero dimensions: restore valid vector rows"
            ),
        });
    }
    if let Some(vector) = vectors.iter().find(|v| v.len() != dim) {
        return Err(LiteError::Corrupted {
            detail: format!(
                "vector segment '{index_key}' expects {dim} dimensions, found {}: restore matching vector rows",
                vector.len()
            ),
        });
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
    #[tokio::test]
    async fn payload_requires_durable_bound_rows_and_consistent_widths() {
        use crate::storage::pagedb_storage::PagedbStorageMem;
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        assert!(
            segment_payload(&storage, "vectors", Vec::new())
                .await
                .unwrap()
                .is_none()
        );
        let missing = segment_payload(
            &storage,
            "vectors",
            vec![NodeSource::Bound("missing".into())],
        )
        .await
        .unwrap_err();
        assert!(matches!(missing, LiteError::Corrupted { .. }));
        assert!(missing.to_string().contains("missing"));
        let zero = segment_payload(&storage, "vectors", vec![NodeSource::Local(Vec::new())])
            .await
            .unwrap_err();
        assert!(matches!(zero, LiteError::Corrupted { .. }));
        let mismatch = segment_payload(
            &storage,
            "vectors",
            vec![
                NodeSource::Local(vec![1.0, 2.0]),
                NodeSource::Local(vec![3.0]),
            ],
        )
        .await
        .unwrap_err();
        assert!(matches!(mismatch, LiteError::Corrupted { .. }));
        storage
            .batch_write(&[crate::engine::vector::durable::put_op(
                "vectors",
                "bound",
                &[4.0, 5.0],
            )])
            .await
            .unwrap();
        let payload = segment_payload(
            &storage,
            "vectors",
            vec![
                NodeSource::Bound("bound".into()),
                NodeSource::Local(vec![1.0, 2.0]),
            ],
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(payload.0, 2);
        assert_eq!(payload.1, vec![vec![4.0, 5.0], vec![1.0, 2.0]]);
        assert_eq!(payload.2, vec![doc_fingerprint("bound"), 0]);
    }
}
