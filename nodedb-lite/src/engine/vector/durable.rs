// SPDX-License-Identifier: Apache-2.0

//! Durable per-document vector storage — the source of truth for vectors.
//!
//! # Why this exists
//!
//! A vector used to live in exactly one place: the in-memory HNSW index, which
//! reached disk only when `flush` wrote the `vec/hnsw/<collection>` segment.
//! That gave vectors a weaker durability guarantee than the documents they
//! belong to — a document is durable the moment its write is acknowledged
//! (versioned put), while its vector survived only if a later flush happened to
//! run. An unclean exit therefore lost every vector written since the last
//! flush, silently, because the write had already reported success.
//!
//! It also made the segment the *only* copy, so a segment that could not be
//! reopened left exactly two options, both wrong: keep a checkpoint whose node
//! vectors are empty placeholders (the first distance computation panics with
//! `dist_to_node: byte-length mismatch`), or drop the index and lose every
//! vector permanently. There was no third option because nothing else held the
//! data — in particular the CRDT holds only `embedding_dim`, never the floats,
//! so the "rebuild from CRDT" the restore path spoke of could never have worked.
//!
//! # The contract
//!
//! Every vector is written here in the same operation that makes its document
//! durable. [`crate::engine::vector::pagedb_backing`] segments become a
//! *derived* index: an accelerator that can always be rebuilt from these rows
//! ([`load_collection`]), never the master copy. That makes a corrupt or
//! unreadable segment a rebuild, not a data-loss event.
//!
//! # Layout
//!
//! `Namespace::Vector`, key `v:<collection>:<doc_id>`, value = little-endian
//! `f32` values with no header. The dimension is implied by the byte length,
//! which is why [`decode`] rejects a length that is not a multiple of 4. The
//! `v:` prefix is disjoint from the other keys in this namespace (`hnsw:<name>`
//! checkpoints and `hnsw_id_map`), so a prefix scan returns vectors only.

use std::collections::HashMap;

use nodedb_types::Namespace;

use crate::error::LiteError;
use crate::storage::engine::{StorageEngine, WriteOp};

/// Bytes per stored element.
const F32_BYTES: usize = 4;

/// Key prefix for per-document vectors, disjoint from `hnsw:*`.
const VEC_PREFIX: &str = "v:";

/// Key-space prefix for one collection's vectors.
pub(crate) fn collection_prefix(collection: &str) -> Vec<u8> {
    format!("{VEC_PREFIX}{collection}:").into_bytes()
}

/// Durable key for one document's vector.
pub(crate) fn key(collection: &str, doc_id: &str) -> Vec<u8> {
    format!("{VEC_PREFIX}{collection}:{doc_id}").into_bytes()
}

/// Encode `vector` as little-endian `f32` bytes.
pub(crate) fn encode(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * F32_BYTES);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode little-endian `f32` bytes back into a vector.
///
/// Returns `None` when `bytes` is not a whole number of `f32`s — a truncated or
/// foreign row is skipped rather than reinterpreted, since a mis-sized vector
/// would panic the distance kernels it is fed to.
pub(crate) fn decode(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(F32_BYTES) {
        return None;
    }
    Some(
        bytes
            .as_chunks::<F32_BYTES>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

/// The write that makes `vector` durable for `doc_id`.
///
/// Returned as a [`WriteOp`] so callers can place it in the SAME batch as the
/// document itself — the whole point is that the two become durable together.
pub(crate) fn put_op(collection: &str, doc_id: &str, vector: &[f32]) -> WriteOp {
    WriteOp::Put {
        ns: Namespace::Vector,
        key: key(collection, doc_id),
        value: encode(vector),
    }
}

/// Remove a document's durable vector.
pub(crate) async fn remove<S: StorageEngine>(
    storage: &S,
    collection: &str,
    doc_id: &str,
) -> Result<(), LiteError> {
    storage
        .delete(Namespace::Vector, &key(collection, doc_id))
        .await
}

/// Every collection that has at least one durable vector.
///
/// Discovery must not depend on `META_HNSW_COLLECTIONS`: that list is written
/// by `flush`, so a database that has taken writes but never flushed has no
/// list at all — and those are exactly the vectors most at risk. Deriving the
/// collection set from the durable rows means a never-flushed database still
/// rebuilds its indexes on open.
pub(crate) async fn list_collections<S: StorageEngine>(
    storage: &S,
) -> Result<Vec<String>, LiteError> {
    let rows = storage
        .scan_prefix(Namespace::Vector, VEC_PREFIX.as_bytes())
        .await?;
    let mut names: Vec<String> = Vec::new();
    for (row_key, _) in rows {
        // `v:<collection>:<doc_id>` — the collection is up to the FIRST ':'
        // after the prefix; document ids may themselves contain ':'.
        let Some(rest) = row_key.get(VEC_PREFIX.len()..) else {
            continue;
        };
        let Some(sep) = rest.iter().position(|&b| b == b':') else {
            continue;
        };
        let Ok(name) = std::str::from_utf8(&rest[..sep]) else {
            continue;
        };
        if !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    }
    Ok(names)
}

/// Load every durable vector for `collection`, as `(doc_id, vector)`.
///
/// This is what makes the HNSW segment rebuildable. Rows that fail to decode
/// are skipped with a warning rather than aborting the load: one unreadable row
/// must not cost the whole index, and skipping is safe because the index is
/// derived — the row stays on disk for a later repair.
pub(crate) async fn load_collection<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<Vec<(String, Vec<f32>)>, LiteError> {
    let prefix = collection_prefix(collection);
    let rows = storage.scan_prefix(Namespace::Vector, &prefix).await?;

    let mut out = Vec::with_capacity(rows.len());
    for (row_key, value) in rows {
        let Some(doc_id) = row_key
            .get(prefix.len()..)
            .and_then(|b| std::str::from_utf8(b).ok())
        else {
            tracing::warn!(
                collection,
                "durable vector row has a non-UTF-8 key; skipping"
            );
            continue;
        };
        let Some(vector) = decode(&value) else {
            tracing::warn!(
                collection,
                doc_id,
                bytes = value.len(),
                "durable vector row is not a whole number of f32s; skipping"
            );
            continue;
        };
        out.push((doc_id.to_owned(), vector));
    }
    Ok(out)
}

/// Document ids with a durable vector row in `collection`, sorted.
///
/// Reads keys only: values pass through the streaming scan undecoded, so a
/// row that [`load_collection`] would skip as malformed still counts here.
/// Ids are cut with [`collection_prefix`], which ends in the `:` delimiter, so
/// `c1` never matches rows of `c10`.
pub(crate) async fn list_doc_ids<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<Vec<String>, LiteError> {
    let prefix = collection_prefix(collection);
    let mut ids: Vec<String> = Vec::new();
    storage
        .scan_prefix_streaming(Namespace::Vector, &prefix, &mut |(row_key, _)| {
            match row_key
                .get(prefix.len()..)
                .and_then(|b| std::str::from_utf8(b).ok())
            {
                Some(doc_id) => ids.push(doc_id.to_owned()),
                None => tracing::warn!(
                    collection,
                    "durable vector row has a non-UTF-8 key; skipping"
                ),
            }
            Ok(true)
        })
        .await?;
    ids.sort_unstable();
    Ok(ids)
}

/// Rebuild `collection`'s HNSW from its durable vectors.
///
/// The single implementation shared by every recovery path — open-time restore
/// and lazy-load both land here, so they cannot drift into different recovery
/// behaviour. Returns the index plus the `"<collection>:<internal_id>" ->
/// (doc_id, internal_id)` entries for it; internal ids are reassigned from
/// zero, so those entries REPLACE any persisted map for this collection.
///
/// `template` carries the `(dim, params)` of the index being replaced so a
/// rebuild cannot silently change how distances are computed. It is taken BY
/// VALUE rather than as an `&HnswIndex` because the index holds a `RefCell`
/// arena — borrowing it across this `await` would make every calling future
/// non-`Send`. When it is `None` (nothing to replace) the params default
/// exactly as `ensure_hnsw` would set them on a first insert. Returns `None`
/// when the collection has no durable vectors.
pub(crate) async fn rebuild_index<S: StorageEngine>(
    storage: &S,
    collection: &str,
    template: Option<(usize, crate::engine::vector::HnswParams)>,
) -> Result<
    Option<(
        crate::engine::vector::HnswIndex,
        HashMap<String, (String, u32)>,
    )>,
    LiteError,
> {
    use crate::engine::vector::{HnswIndex, HnswParams};

    let rows = load_collection(storage, collection).await?;
    if rows.is_empty() {
        return Ok(None);
    }

    let (dim, params) = match template {
        Some(t) => t,
        None => (rows[0].1.len(), HnswParams::default()),
    };
    let mut index = HnswIndex::new(dim, params);
    let mut id_map = HashMap::new();

    for (doc_id, vector) in rows {
        if vector.len() != dim {
            tracing::warn!(
                collection,
                doc_id,
                expected = dim,
                found = vector.len(),
                "durable vector has the wrong dimension for its collection; skipping"
            );
            continue;
        }
        let internal_id = index.len() as u32;
        if let Err(e) = index.insert(vector) {
            tracing::warn!(collection, doc_id, error = %e, "durable vector insert failed; skipping");
            continue;
        }
        id_map.insert(format!("{collection}:{internal_id}"), (doc_id, internal_id));
    }

    Ok(Some((index, id_map)))
}

/// One index's slot → document id bindings, as read from the id map.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type SlotBindings = HashMap<u32, String>;

/// The `(dim, vectors, stamps)` payload serialized into a vector segment.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type SegmentPayload = (usize, Vec<Vec<f32>>, Vec<u64>);

/// Segment stamp of a slot that no document is bound to.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const TOMB: u64 = u64::MAX;

/// Segment stamp of the slot bound to `doc_id`.
///
/// FNV-1a 64 over the UTF-8 bytes of `doc_id` (offset basis
/// `0xcbf29ce484222325`, prime `0x100000001b3`), with the low bit forced
/// to 1. The algorithm is spelled out here, not taken from `std`, because a
/// stamp written by one build is read by every later one: `DefaultHasher`
/// is not stable across Rust releases. The low bit makes a stamp nonzero, so
/// zero always means "written before segments carried stamps". A stamp can
/// equal [`TOMB`]; that is harmless, because a bound slot is compared against
/// its document's stamp and an unbound slot accepts any nonzero stamp.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn stamp(doc_id: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in doc_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash | 1
}

/// Group id-map bindings by index key, for every index key `wanted` accepts.
///
/// `entries` is the flat id-map form: `"{index_key}:{slot}"` → (document id,
/// slot). The index key is everything before the LAST `:`, since an index key
/// can itself contain `:` and a slot never does. An entry whose key does not
/// name its own slot is skipped.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn slot_bindings<'a>(
    entries: impl IntoIterator<Item = (&'a String, &'a (String, u32))>,
    mut wanted: impl FnMut(&str) -> bool,
) -> HashMap<String, SlotBindings> {
    let mut out: HashMap<String, SlotBindings> = HashMap::new();
    for (composite, (doc_id, slot)) in entries {
        let Some((index_key, named)) = composite.rsplit_once(':') else {
            continue;
        };
        if named.parse::<u32>().ok() != Some(*slot) || !wanted(index_key) {
            continue;
        }
        match out.get_mut(index_key) {
            Some(bindings) => {
                bindings.insert(*slot, doc_id.clone());
            }
            None => {
                out.insert(
                    index_key.to_owned(),
                    HashMap::from([(*slot, doc_id.clone())]),
                );
            }
        }
    }
    out
}

/// The first live node of `index` that `bindings` does not bind, if any.
///
/// Insert paths add the node under the index lock and bind it afterwards
/// under the id-map lock. A live unbound node therefore means an insert is
/// between those two steps. A segment stamped from these bindings would mark
/// that slot [`TOMB`], and the next open would refuse the segment.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn unbound_live_slot(
    index: &crate::engine::vector::HnswIndex,
    bindings: &SlotBindings,
) -> Option<u32> {
    (0..index.len() as u32).find(|slot| !index.is_deleted(*slot) && !bindings.contains_key(slot))
}

/// The payload to serialize into `index`'s vector segment, in node order.
///
/// `vectors[i]` is node `i`'s vector, read through the attached segment
/// backing when the node's own storage is empty (graph-only checkpoint
/// restore). Tombstoned nodes keep their vector, so `vectors.len()` equals the
/// node count. `stamps[i]` is [`stamp`] of the document slot `i` is bound to
/// in `bindings`, or [`TOMB`] for an unbound slot.
///
/// Node order is the contract. The graph checkpoint numbers nodes by
/// insertion, and attaching a backing maps its entry `i` to node `i`. A
/// payload in any other order, such as the durable rows' key order, attaches
/// the wrong vector to each node without any error. The stamps let the next
/// open check the mapping before it attaches, see [`verify_segment_stamps`].
///
/// # Errors
///
/// [`LiteError::Storage`] when a node's vector cannot be materialized. The
/// caller must then leave the stored segment as it is and keep it dirty.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn segment_payload_from_index(
    index: &crate::engine::vector::HnswIndex,
    bindings: &SlotBindings,
) -> Result<SegmentPayload, LiteError> {
    let vectors = index.export_vectors().map_err(|e| LiteError::Storage {
        detail: format!("reading HNSW node vectors for the vector segment failed: {e}"),
    })?;
    let stamps = (0..vectors.len())
        .map(|slot| {
            bindings
                .get(&(slot as u32))
                .map_or(TOMB, |doc_id| stamp(doc_id))
        })
        .collect();
    Ok((index.dim(), vectors, stamps))
}

/// Why a stored vector segment cannot back an index.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StampMismatch {
    /// The segment holds fewer slots than the index has nodes.
    ShortSegment { nodes: usize, slots: usize },
    /// The slot carries a zero stamp: the segment predates stamping.
    Unstamped { slot: u32 },
    /// The slot's stamp is not that of the document bound to it.
    WrongDocument { slot: u32, doc_id: String },
    /// No document is bound to the slot, but its node is live.
    UnboundLiveNode { slot: u32 },
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Display for StampMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortSegment { nodes, slots } => {
                write!(f, "segment has {slots} slots for {nodes} nodes")
            }
            Self::Unstamped { slot } => write!(
                f,
                "slot {slot} has no stamp; the segment predates node-order stamps"
            ),
            Self::WrongDocument { slot, doc_id } => write!(
                f,
                "slot {slot} is bound to {doc_id:?} but its stamp belongs to another document"
            ),
            Self::UnboundLiveNode { slot } => {
                write!(f, "slot {slot} is unbound but its node is live")
            }
        }
    }
}

/// Check that `backing` was written for `index` in its node order.
///
/// Run before `HnswIndex::with_backing`, which checks only the dimension and
/// the slot count. For each node `i`, the backing's stamp at `i` must be:
///
/// - nonzero: a zero stamp is a segment written before stamping existed.
/// - [`stamp`] of the document `bindings` binds to slot `i`, when bound.
/// - any nonzero value when slot `i` is unbound, and then node `i` must be
///   deleted.
///
/// Slots past the node count are not checked. A segment can legitimately be
/// longer than the graph it backs.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn verify_segment_stamps(
    index: &crate::engine::vector::HnswIndex,
    bindings: &SlotBindings,
    backing: &dyn nodedb_vector::segment_backing::VectorSegmentBacking,
) -> Result<(), StampMismatch> {
    let nodes = index.len();
    if backing.len() < nodes {
        return Err(StampMismatch::ShortSegment {
            nodes,
            slots: backing.len(),
        });
    }
    for slot in 0..nodes as u32 {
        let sid = backing.get_surrogate(slot).unwrap_or(0);
        if sid == 0 {
            return Err(StampMismatch::Unstamped { slot });
        }
        match bindings.get(&slot) {
            Some(doc_id) => {
                if sid != stamp(doc_id) {
                    return Err(StampMismatch::WrongDocument {
                        slot,
                        doc_id: doc_id.clone(),
                    });
                }
            }
            None => {
                if !index.is_deleted(slot) {
                    return Err(StampMismatch::UnboundLiveNode { slot });
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_vector() {
        let v = vec![1.0_f32, -2.5, 0.0, 3.25];
        assert_eq!(decode(&encode(&v)).expect("decodes"), v);
    }

    /// A truncated row must be REJECTED, not reinterpreted. A vector whose
    /// length is not a whole number of f32s would reach the distance kernels
    /// with the wrong byte length and panic there instead.
    #[test]
    fn rejects_a_truncated_row() {
        let mut bytes = encode(&[1.0_f32, 2.0]);
        bytes.pop();
        assert!(decode(&bytes).is_none());
        assert!(decode(&[]).is_none());
    }

    /// The per-document prefix must not collide with the other keys living in
    /// `Namespace::Vector`, or a rebuild scan would pick up checkpoints.
    #[test]
    fn key_prefix_is_disjoint_from_checkpoint_keys() {
        let k = key("entries", "abc");
        assert!(k.starts_with(&collection_prefix("entries")));
        assert!(!k.starts_with(b"hnsw:"));
        assert_ne!(k.as_slice(), b"hnsw_id_map");
    }

    /// One collection's scan prefix must not match another's.
    #[test]
    fn collection_prefixes_do_not_alias() {
        let entries = collection_prefix("entries");
        assert!(!key("entries_archive", "x").starts_with(&entries));
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod stamps {
        use super::super::*;
        use crate::engine::vector::pagedb_backing::{PagedbBacking, build_ndvs_bytes};
        use crate::engine::vector::{HnswIndex, HnswParams};

        /// Three nodes, `[1,0,0]`, `[0,1,0]`, `[0,0,1]`.
        fn index() -> HnswIndex {
            let mut index = HnswIndex::new(3, HnswParams::default());
            for v in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                index.insert(v.to_vec()).unwrap();
            }
            index
        }

        fn bindings(pairs: &[(u32, &str)]) -> SlotBindings {
            pairs
                .iter()
                .map(|(slot, doc)| (*slot, (*doc).to_string()))
                .collect()
        }

        fn backing(index: &HnswIndex, stamps: &[u64]) -> PagedbBacking {
            let vectors = index.export_vectors().unwrap();
            let bytes = build_ndvs_bytes(index.dim(), &vectors, stamps).unwrap();
            PagedbBacking::from_bytes(bytes.into_boxed_slice()).unwrap()
        }

        #[test]
        fn stamp_is_stable_and_never_zero() {
            // FNV-1a 64 of "a" is 0xaf63dc4c8601ec8c; the low bit is then set.
            assert_eq!(stamp("a"), 0xaf63_dc4c_8601_ec8d);
            assert_eq!(stamp(""), 0xcbf2_9ce4_8422_2325);
            for id in ["", "a", "b", "doc:1"] {
                assert_ne!(stamp(id), 0);
            }
            assert_ne!(stamp("a"), stamp("b"));
        }

        #[test]
        fn a_segment_written_from_the_index_passes() {
            let index = index();
            let bound = bindings(&[(0, "b"), (1, "a"), (2, "c")]);
            let (dim, vectors, stamps) = segment_payload_from_index(&index, &bound).unwrap();
            assert_eq!(dim, 3);
            assert_eq!(vectors[0], vec![1.0, 0.0, 0.0], "payload is in node order");
            let bytes = build_ndvs_bytes(dim, &vectors, &stamps).unwrap();
            let seg = PagedbBacking::from_bytes(bytes.into_boxed_slice()).unwrap();
            assert_eq!(verify_segment_stamps(&index, &bound, &seg), Ok(()));
        }

        #[test]
        fn a_bound_slot_with_another_documents_stamp_is_refused() {
            let index = index();
            let bound = bindings(&[(0, "b"), (1, "a"), (2, "c")]);
            // Key order ("a", "b", "c") instead of node order ("b", "a", "c").
            let seg = backing(&index, &[stamp("a"), stamp("b"), stamp("c")]);
            assert_eq!(
                verify_segment_stamps(&index, &bound, &seg),
                Err(StampMismatch::WrongDocument {
                    slot: 0,
                    doc_id: "b".into()
                })
            );
        }

        #[test]
        fn an_unbound_live_slot_is_refused() {
            let index = index();
            let bound = bindings(&[(0, "b"), (2, "c")]);
            let seg = backing(&index, &[stamp("b"), TOMB, stamp("c")]);
            assert_eq!(
                verify_segment_stamps(&index, &bound, &seg),
                Err(StampMismatch::UnboundLiveNode { slot: 1 })
            );
        }

        #[test]
        fn an_unbound_deleted_slot_passes() {
            let mut index = index();
            index.delete(1);
            let bound = bindings(&[(0, "b"), (2, "c")]);
            let seg = backing(&index, &[stamp("b"), TOMB, stamp("c")]);
            assert_eq!(verify_segment_stamps(&index, &bound, &seg), Ok(()));
        }

        #[test]
        fn a_zero_stamp_is_refused() {
            let mut index = index();
            index.delete(1);
            let bound = bindings(&[(0, "b"), (2, "c")]);
            // An empty stamp slice is how every pre-stamp segment was written:
            // the surrogate block is all zeros.
            let seg = backing(&index, &[]);
            assert_eq!(
                verify_segment_stamps(&index, &bound, &seg),
                Err(StampMismatch::Unstamped { slot: 0 })
            );
            let seg = backing(&index, &[stamp("b"), 0, stamp("c")]);
            assert_eq!(
                verify_segment_stamps(&index, &bound, &seg),
                Err(StampMismatch::Unstamped { slot: 1 }),
                "an unbound slot still needs a nonzero stamp"
            );
        }

        #[test]
        fn a_short_segment_is_refused() {
            let index = index();
            let bound = bindings(&[(0, "b"), (1, "a"), (2, "c")]);
            let bytes = build_ndvs_bytes(
                3,
                &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                &[stamp("b"), stamp("a")],
            )
            .unwrap();
            let seg = PagedbBacking::from_bytes(bytes.into_boxed_slice()).unwrap();
            assert_eq!(
                verify_segment_stamps(&index, &bound, &seg),
                Err(StampMismatch::ShortSegment { nodes: 3, slots: 2 })
            );
        }

        #[test]
        fn bindings_group_by_the_index_key_before_the_last_colon() {
            let entries: HashMap<String, (String, u32)> = [
                ("docs:0".to_string(), ("a".to_string(), 0)),
                ("docs:emb:0".to_string(), ("b".to_string(), 0)),
                ("docs:emb:1".to_string(), ("c".to_string(), 1)),
                ("docs:7".to_string(), ("stale".to_string(), 3)),
            ]
            .into_iter()
            .collect();
            let grouped = slot_bindings(entries.iter(), |_| true);
            assert_eq!(grouped.get("docs"), Some(&bindings(&[(0, "a")])));
            assert_eq!(
                grouped.get("docs:emb"),
                Some(&bindings(&[(0, "b"), (1, "c")]))
            );
            let only = slot_bindings(entries.iter(), |k| k == "docs:emb");
            assert_eq!(only.len(), 1);
        }
    }
}
