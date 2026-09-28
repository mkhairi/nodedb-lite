// SPDX-License-Identifier: Apache-2.0

//! Reading and writing the durable per-document vector rows.

use nodedb_types::Namespace;

use crate::error::LiteError;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::layout::{ROW_PREFIX, index_prefix, key, parse_key};

/// Bytes per stored element.
const F32_BYTES: usize = 4;

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

/// Every index key that has at least one durable vector: base keys
/// (`"{collection}"`) and named keys (`"{collection}:{field}"`) alike.
///
/// Discovery must not depend on `META_HNSW_COLLECTIONS`: that list is written
/// by `flush`, so a database that has taken writes but never flushed has no
/// list at all — and those are exactly the vectors most at risk. Deriving the
/// index set from the durable rows means a never-flushed database still
/// rebuilds its indexes on open.
pub(crate) async fn list_collections<S: StorageEngine>(
    storage: &S,
) -> Result<Vec<String>, LiteError> {
    let rows = storage
        .scan_prefix(Namespace::Vector, ROW_PREFIX.as_bytes())
        .await?;
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut names: Vec<String> = Vec::new();
    for (row_key, _) in &rows {
        let Some((index_key, _)) = parse_key(row_key) else {
            continue;
        };
        if seen.insert(index_key) {
            names.push(index_key.to_owned());
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
    let prefix = index_prefix(collection);
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

/// Rebuild `collection`'s HNSW from its durable vectors.
///
/// The single implementation shared by every recovery path — open-time restore
/// and lazy-load both land here, so they cannot drift into different recovery
/// behaviour. Returns the index plus its node ↔ document bindings; node ids
/// are reassigned from zero, so those bindings REPLACE any persisted ones for
/// this collection.
///
/// `template` carries the `(dim, params)` of the index being replaced so a
/// rebuild cannot silently change how distances are computed. It is taken BY
/// VALUE rather than as an `&HnswIndex` because the index holds a `RefCell`
/// arena — borrowing it across this `await` would make every calling future
/// non-`Send`. When it is `None` (nothing to replace) the params default
/// exactly as `resident::lock_resident_or_create` sets them on a first
/// insert. Returns `None` when the collection has no durable vectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) async fn rebuild_index<S: StorageEngine>(
    storage: &S,
    collection: &str,
    template: Option<(usize, crate::engine::vector::HnswParams)>,
) -> Result<
    Option<(
        crate::engine::vector::HnswIndex,
        crate::engine::vector::IndexIdMap,
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
    let mut id_map = crate::engine::vector::IndexIdMap::default();

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
        // Durable rows are one per id, so no binding is displaced here.
        id_map.bind(&doc_id, internal_id);
    }

    Ok(Some((index, id_map)))
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
}
