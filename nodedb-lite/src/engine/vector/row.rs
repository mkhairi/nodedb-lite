// SPDX-License-Identifier: Apache-2.0

//! The CRDT row a vector is attached to.
//!
//! A vector is an attachment to the row with the same collection and id. A
//! vector write merges its own fields into that row and never replaces the
//! row's other fields. A vector delete removes those fields again, and
//! removes the whole row only when nothing but the vector lives in it.

use loro::LoroValue;
use nodedb_types::collection_config::PrimaryEngine;
use nodedb_types::sync::wire::CollectionDescriptor;

use crate::engine::crdt::CrdtEngine;
use crate::engine::vector::AttachedVectors;
use crate::error::LiteError;
use crate::nodedb::collection::CollectionMeta;
use crate::storage::engine::StorageEngine;

/// The vector's dimension, written on every vector row.
pub(crate) const EMBEDDING_DIM_FIELD: &str = "embedding_dim";

/// The named vector a row carries, written by named-vector inserts.
pub(crate) const VECTOR_FIELD_TAG: &str = "__field";

/// Every row field a vector write owns. None of them is document data.
pub(crate) const VECTOR_ROW_FIELDS: &[&str] = &[EMBEDDING_DIM_FIELD, VECTOR_FIELD_TAG];

/// Whether `collection` is declared vector-primary: its rows exist for their
/// vectors, so deleting the vector deletes the row.
///
/// Reads the persisted collection descriptor. A collection with no persisted
/// descriptor is not declared vector-primary. Fails when reading storage
/// fails.
pub(crate) async fn is_vector_primary<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<bool, LiteError> {
    let key = format!("collection:{collection}");
    let Some(bytes) = storage
        .get(nodedb_types::Namespace::Meta, key.as_bytes())
        .await?
    else {
        return Ok(false);
    };
    let Ok(meta) = sonic_rs::from_slice::<CollectionMeta>(&bytes) else {
        return Ok(false);
    };
    let Some(descriptor) = meta
        .descriptor_json
        .as_deref()
        .and_then(|json| sonic_rs::from_str::<CollectionDescriptor>(json).ok())
    else {
        return Ok(false);
    };
    Ok(matches!(descriptor.primary, PrimaryEngine::Vector) || descriptor.vector_primary.is_some())
}

/// Which of a row's vectors a detach removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorSlot<'a> {
    /// The base vector, in index `"{collection}"`.
    Base,
    /// A named vector, in index `"{collection}:{field}"`.
    Named(&'a str),
}

/// Detach one vector from `doc_id`'s row in `collection`, after its index
/// binding is gone. `remaining` lists the vectors still attached to the row.
///
/// Only fields the detached vector owns alone go; document fields stay:
/// - No vector remains: `embedding_dim` and `__field` go. The whole row goes
///   when `vector_primary` is set or no document field remains.
/// - A vector remains: `embedding_dim` stays with it. Detaching the last
///   named vector removes `__field`; detaching the named vector `__field`
///   names retags it to a remaining named vector. Detaching the base vector
///   changes nothing.
///
/// Emits at most one CRDT delta. Returns its mutation ID, or 0 when nothing
/// changed.
pub(crate) fn detach_vector_row(
    crdt: &mut CrdtEngine,
    collection: &str,
    doc_id: &str,
    detached: VectorSlot<'_>,
    remaining: &AttachedVectors,
    vector_primary: bool,
) -> Result<u64, LiteError> {
    let Some(row) = crdt.read(collection, doc_id) else {
        return Ok(0);
    };
    if !remaining.any() {
        if vector_primary || !has_document_fields(&row) {
            return crdt.delete(collection, doc_id);
        }
        let (_, mutation_id) = crdt.remove_fields(collection, doc_id, VECTOR_ROW_FIELDS)?;
        return Ok(mutation_id);
    }
    let VectorSlot::Named(field) = detached else {
        return Ok(0);
    };
    match remaining.named.first() {
        None => {
            let (_, mutation_id) = crdt.remove_fields(collection, doc_id, &[VECTOR_FIELD_TAG])?;
            Ok(mutation_id)
        }
        Some(other) if field_tag(&row) == Some(field) => crdt.set_fields(
            collection,
            doc_id,
            &[(VECTOR_FIELD_TAG, LoroValue::String(other.as_str().into()))],
        ),
        Some(_) => Ok(0),
    }
}

/// The named vector `row`'s `__field` records.
fn field_tag(row: &LoroValue) -> Option<&str> {
    match row {
        LoroValue::Map(map) => match map.get(VECTOR_FIELD_TAG) {
            Some(LoroValue::String(tag)) => Some(tag.as_str()),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `row` holds any field a vector write does not own.
fn has_document_fields(row: &LoroValue) -> bool {
    match row {
        LoroValue::Map(map) => map.keys().any(|k| !VECTOR_ROW_FIELDS.contains(&k.as_str())),
        _ => false,
    }
}
