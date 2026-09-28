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
//! `Namespace::Vector`, one row per `(index key, document id)`; see
//! [`layout`] for the key. The value is little-endian `f32` values with no
//! header. The dimension is implied by the byte length, which is why
//! `decode` rejects a length that is not a multiple of 4. Databases written
//! in the earlier key layout are rewritten once on open by [`migrate`].

mod layout;
pub(crate) mod migrate;
mod rows;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use rows::rebuild_index;
pub(crate) use rows::{list_collections, load_collection, put_op, remove};
