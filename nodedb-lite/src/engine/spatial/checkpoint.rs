//! Checkpoint serialization and restoration for [`SpatialIndexManager`].
//!
//! Persists the in-memory spatial state to `Namespace::Spatial` so that a
//! cold open can load the index without rebuilding from CRDT documents. A
//! flush writes only the R-trees, doc-maps, and catalog entries that changed
//! since this handle last wrote them.
//!
//! ## Key layout under `Namespace::Spatial`
//!
//! | Key                                    | Value                                               |
//! |----------------------------------------|-----------------------------------------------------|
//! | `spatial:_collections`                 | MessagePack `Vec<(String, String)>` — (collection, field) pairs |
//! | `spatial:{collection}:{field}:docmap`  | MessagePack `Vec<(String, u64)>` — doc_id → entry_id |
//! | `spatial:_next_id`                     | MessagePack `u64` — next entry ID                  |
//!
//! The R-tree blob (`spatial:{collection}:{field}:rtree`) is stored in a pagedb
//! segment when `as_spatial_segment_ext()` is available, or falls back to the
//! `Namespace::Spatial` KV path (e.g. WASM). In both cases the
//! bytes stored are the CRC32C-wrapped R-tree checkpoint produced by
//! `crate::storage::checksum::wrap`.

use std::collections::HashMap;

use nodedb_spatial::rtree::RTree;
use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::nodedb::flush_gens::{ArtifactFlush, FlushArtifact, FlushGens, spatial_rtree_key};
use crate::storage::engine::{StorageEngine, WriteOp};

/// Catalog key: the sorted `(collection, field)` list.
const COLLECTIONS_KEY: &[u8] = b"spatial:_collections";

/// Next entry id key.
const NEXT_ID_KEY: &[u8] = b"spatial:_next_id";

/// One R-tree checkpoint a flush writes after the catalog batch.
pub(crate) struct SpatialTreeFlush {
    /// Mark flushed once this tree's own write succeeds.
    pub(crate) planned: ArtifactFlush,
    pub(crate) collection: String,
    pub(crate) field: String,
    /// The R-tree checkpoint, not yet CRC32C-wrapped.
    pub(crate) bytes: Vec<u8>,
}

/// The spatial writes one flush plans.
pub(crate) struct SpatialFlush {
    /// Puts for the dirty doc-maps and the changed catalog entries, written
    /// as one batch.
    pub(crate) ops: Vec<WriteOp>,
    /// One plan per dirty collection doc-map put in `ops`.
    pub(crate) docmaps: Vec<ArtifactFlush>,
    /// Catalog entries put in `ops`, recorded as written once `ops` commits.
    pub(crate) meta: Vec<(&'static [u8], Vec<u8>)>,
    /// Dirty R-trees, each written after `ops`.
    pub(crate) trees: Vec<SpatialTreeFlush>,
}

/// Serialize the spatial state a flush must write.
///
/// A doc-map or R-tree is written only when it is dirty or `full` is set.
/// Each generation is captured by `FlushGens::plan`, so call this under the
/// manager lock the state is read under. `spatial:_collections` and
/// `spatial:_next_id` are written only when they differ from the value this
/// handle last wrote, or `full` is set.
///
/// Every registered tree is listed in the catalog, sorted, so the encoded
/// list compares equal across ticks while the set is unchanged. A tree whose
/// checkpoint fails to serialize is logged and stays dirty.
pub(crate) fn serialize_spatial(
    indices: &HashMap<(String, String), RTree>,
    doc_to_entry: &HashMap<(String, String), u64>,
    next_id: u64,
    full: bool,
    gens: &FlushGens,
) -> NodeDbResult<SpatialFlush> {
    let mut ops: Vec<WriteOp> = Vec::new();
    let mut docmaps: Vec<ArtifactFlush> = Vec::new();
    let mut meta: Vec<(&'static [u8], Vec<u8>)> = Vec::new();
    let mut trees: Vec<SpatialTreeFlush> = Vec::new();

    // ── Collection list ───────────────────────────────────────────────────────
    let mut index_keys: Vec<(String, String)> = indices.keys().cloned().collect();
    index_keys.sort();
    let keys_bytes = zerompk::to_msgpack_vec(&index_keys)
        .map_err(|e| NodeDbError::serialization("msgpack", e))?;
    if full || gens.meta_changed(COLLECTIONS_KEY, &keys_bytes) {
        ops.push(WriteOp::Put {
            ns: Namespace::Spatial,
            key: COLLECTIONS_KEY.to_vec(),
            value: keys_bytes.clone(),
        });
        meta.push((COLLECTIONS_KEY, keys_bytes));
    }

    // ── Next entry ID ─────────────────────────────────────────────────────────
    let next_id_bytes =
        zerompk::to_msgpack_vec(&next_id).map_err(|e| NodeDbError::serialization("msgpack", e))?;
    if full || gens.meta_changed(NEXT_ID_KEY, &next_id_bytes) {
        ops.push(WriteOp::Put {
            ns: Namespace::Spatial,
            key: NEXT_ID_KEY.to_vec(),
            value: next_id_bytes.clone(),
        });
        meta.push((NEXT_ID_KEY, next_id_bytes));
    }

    // ── Per-collection doc-map (always on B+ tree) ────────────────────────────
    // The doc-map is filtered by collection only, so every field of a
    // collection stores the same bytes. It is tracked and written per
    // collection.
    let mut collections: Vec<&str> = index_keys.iter().map(|(c, _)| c.as_str()).collect();
    collections.dedup();
    for collection in collections {
        let Some(planned) = gens.plan(FlushArtifact::SpatialDocMap, collection, full) else {
            continue;
        };
        let mut pairs: Vec<(String, u64)> = doc_to_entry
            .iter()
            .filter(|((coll, _doc_id), _)| coll == collection)
            .map(|((_coll, doc_id), &entry_id)| (doc_id.clone(), entry_id))
            .collect();
        pairs.sort();
        let docmap_bytes = zerompk::to_msgpack_vec(&pairs)
            .map_err(|e| NodeDbError::serialization("msgpack", e))?;
        for (_, field) in index_keys.iter().filter(|(c, _)| c == collection) {
            ops.push(WriteOp::Put {
                ns: Namespace::Spatial,
                key: format!("spatial:{collection}:{field}:docmap").into_bytes(),
                value: docmap_bytes.clone(),
            });
        }
        docmaps.push(planned);
    }

    // ── Per-index R-tree bytes ────────────────────────────────────────────────
    for (collection, field) in &index_keys {
        let Some(tree) = indices.get(&(collection.clone(), field.clone())) else {
            continue;
        };
        let key = spatial_rtree_key(collection, field);
        let Some(planned) = gens.plan(FlushArtifact::SpatialRtree, &key, full) else {
            continue;
        };
        match tree.checkpoint_to_bytes(None) {
            Ok(bytes) => trees.push(SpatialTreeFlush {
                planned,
                collection: collection.clone(),
                field: field.clone(),
                bytes,
            }),
            Err(e) => {
                // Not planned as written, so it stays dirty and the next
                // flush tries again.
                tracing::error!(
                    collection = %collection,
                    field = %field,
                    error = %e,
                    "spatial index checkpoint failed"
                );
            }
        }
    }

    Ok(SpatialFlush {
        ops,
        docmaps,
        meta,
        trees,
    })
}

/// Write the spatial state `serialize_spatial` planned.
///
/// The doc-maps and catalog entries go in one batch, skipped when empty, and
/// are marked written once it commits. A failed batch returns the error with
/// nothing marked. Each R-tree is then written on its own and marked flushed
/// only after its write succeeds. A failed segment write is logged and the
/// tree stays dirty, so the next flush retries it.
pub(crate) async fn write_serialized_spatial<S>(
    storage: &S,
    flush: SpatialFlush,
    gens: &FlushGens,
) -> NodeDbResult<()>
where
    S: StorageEngine,
{
    let SpatialFlush {
        ops,
        docmaps,
        meta,
        trees,
    } = flush;

    // ── Commit catalog + docmap to B+ tree ────────────────────────────────────
    if !ops.is_empty() {
        storage
            .batch_write(&ops)
            .await
            .map_err(NodeDbError::storage)?;
    }
    for planned in &docmaps {
        gens.mark_flushed(planned);
        gens.record_write(planned);
    }
    for (key, value) in meta {
        gens.mark_meta_written(key, value);
    }

    // ── Per-index R-tree bytes: segment when available, KV fallback ───────────
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(seg) = storage.as_spatial_segment_ext() {
        for tree in &trees {
            let wrapped = crate::storage::checksum::wrap(&tree.bytes);
            match seg
                .write_spatial_segment(&tree.collection, &tree.field, &wrapped)
                .await
            {
                Ok(()) => {
                    gens.mark_flushed(&tree.planned);
                    gens.record_write(&tree.planned);
                }
                Err(e) => {
                    tracing::error!(
                        collection = %tree.collection,
                        field = %tree.field,
                        error = %e,
                        "spatial R-tree segment write failed; \
                         the tree stays dirty and the next flush retries it"
                    );
                }
            }
        }
        return Ok(());
    }

    // Legacy KV path (WASM fallback).
    let mut rtree_ops: Vec<WriteOp> = Vec::with_capacity(trees.len());
    for tree in &trees {
        let rtree_key = format!("spatial:{}:{}:rtree", tree.collection, tree.field);
        rtree_ops.push(WriteOp::Put {
            ns: Namespace::Spatial,
            key: rtree_key.into_bytes(),
            value: crate::storage::checksum::wrap(&tree.bytes),
        });
    }
    if !rtree_ops.is_empty() {
        storage
            .batch_write(&rtree_ops)
            .await
            .map_err(NodeDbError::storage)?;
    }
    for tree in &trees {
        gens.mark_flushed(&tree.planned);
        gens.record_write(&tree.planned);
    }

    Ok(())
}

/// Restore spatial state from storage on cold open.
///
/// Returns `(checkpoints, doc_to_entry, next_id)`.
/// Returns an empty state if no checkpoint is found.
/// Whether the last flush recorded an empty spatial catalog, meaning the store
/// had no spatial index when it last wrote one.
///
/// Every flushing handle writes `spatial:_collections` once, even when empty,
/// so this tells "nothing to index" apart from "never checkpointed". A missing
/// or undecodable key reads as `false`.
pub(crate) async fn catalog_records_no_index<S: StorageEngine>(storage: &S) -> NodeDbResult<bool> {
    let Some(bytes) = storage.get(Namespace::Spatial, COLLECTIONS_KEY).await? else {
        return Ok(false);
    };
    Ok(zerompk::from_msgpack::<Vec<(String, String)>>(&bytes).is_ok_and(|keys| keys.is_empty()))
}

pub(crate) async fn restore_spatial<S>(
    storage: &S,
) -> NodeDbResult<(
    Vec<(String, String, Vec<u8>)>,
    HashMap<(String, String), u64>,
    u64,
)>
where
    S: StorageEngine,
{
    // ── Read collection list ──────────────────────────────────────────────────
    let Some(keys_bytes) = storage
        .get(Namespace::Spatial, b"spatial:_collections")
        .await?
    else {
        return Ok((Vec::new(), HashMap::new(), 1));
    };

    let Ok(index_keys) = zerompk::from_msgpack::<Vec<(String, String)>>(&keys_bytes) else {
        tracing::warn!("spatial checkpoint: failed to decode collection list — starting fresh");
        return Ok((Vec::new(), HashMap::new(), 1));
    };

    if index_keys.is_empty() {
        return Ok((Vec::new(), HashMap::new(), 1));
    }

    // ── Read next_id ──────────────────────────────────────────────────────────
    let next_id = if let Some(bytes) = storage.get(Namespace::Spatial, b"spatial:_next_id").await? {
        zerompk::from_msgpack::<u64>(&bytes).unwrap_or(1)
    } else {
        1
    };

    // ── Per-index R-tree bytes and doc-map ────────────────────────────────────
    let mut checkpoints: Vec<(String, String, Vec<u8>)> = Vec::new();
    let mut doc_to_entry: HashMap<(String, String), u64> = HashMap::new();

    for (collection, field) in &index_keys {
        // Try pagedb segment first (non-WASM), then fall back to KV blob.
        let rtree_envelope: Option<Vec<u8>> = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Some(seg) = storage.as_spatial_segment_ext() {
                    match seg.open_spatial_segment(collection, field).await {
                        Ok(Some(boxed)) => Some(boxed.into_vec()),
                        Ok(None) => {
                            // Segment absent — fall through to KV blob.
                            None
                        }
                        Err(e) => {
                            tracing::warn!(
                                collection = %collection,
                                field = %field,
                                error = %e,
                                "spatial segment open failed — falling back to KV blob"
                            );
                            None
                        }
                    }
                } else {
                    None
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                None
            }
        };

        // If segment was not found (absent or not supported), try KV blob.
        let envelope = if let Some(env) = rtree_envelope {
            env
        } else {
            let rtree_key = format!("spatial:{collection}:{field}:rtree");
            match storage.get(Namespace::Spatial, rtree_key.as_bytes()).await {
                Ok(Some(env)) => env,
                Ok(None) => continue,
                Err(_) => continue,
            }
        };

        match crate::storage::checksum::unwrap(&envelope) {
            Some(bytes) => {
                checkpoints.push((collection.clone(), field.clone(), bytes));
            }
            None => {
                tracing::error!(
                    collection = %collection,
                    field = %field,
                    "spatial R-tree CRC32C mismatch — discarding"
                );
                // Best-effort cleanup of the stale KV blob (segment path has
                // no stale entry to clean in this branch).
                let rtree_key = format!("spatial:{collection}:{field}:rtree");
                let _ = storage
                    .delete(Namespace::Spatial, rtree_key.as_bytes())
                    .await;
                continue;
            }
        }

        // ── Restore doc_id → entry_id mapping ────────────────────────────────
        let docmap_key = format!("spatial:{collection}:{field}:docmap");
        if let Ok(Some(docmap_bytes)) = storage.get(Namespace::Spatial, docmap_key.as_bytes()).await
            && let Ok(pairs) = zerompk::from_msgpack::<Vec<(String, u64)>>(&docmap_bytes)
        {
            for (doc_id, entry_id) in pairs {
                doc_to_entry.insert((collection.clone(), doc_id), entry_id);
            }
        }
    }

    tracing::debug!(
        index_count = checkpoints.len(),
        doc_entry_count = doc_to_entry.len(),
        next_id,
        "spatial checkpoint restored"
    );

    Ok((checkpoints, doc_to_entry, next_id))
}
