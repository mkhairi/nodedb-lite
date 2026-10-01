// SPDX-License-Identifier: Apache-2.0

//! `NodeDb` trait implementation for `NodeDbLite`, split by concern.
//!
//! - `dispatch`: the single `impl NodeDb for NodeDbLite<S>` block.
//! - `vector` / `graph` / `document` / `sql_lifecycle` / `crdt_list`:
//!   inherent helpers the dispatch block delegates to, one file per domain.

mod crdt_list;
mod dispatch;
mod document;
mod document_batch;
mod document_vector;
mod graph;
mod graph_traversal;
mod graph_walk;
mod sql_lifecycle;
mod vector;
#[cfg(test)]
mod vector_tests;

pub use document_batch::BatchItem;
