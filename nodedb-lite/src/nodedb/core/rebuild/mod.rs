// SPDX-License-Identifier: Apache-2.0

//! Cold-start index rebuild helpers for `NodeDbLite`.
//!
//! Separate sub-modules handle each index family so no single file exceeds
//! the 500-line limit:
//!
//! - `text`  — FTS + Spatial rebuild (CRDT + DocumentHistory)
//! - `graph` — CSR adjacency rebuild (CRDT + Namespace::Graph KV + GraphHistory)
//! - `graph_edge_migration` — one-time legacy edge CRDT key rewrite

pub(super) mod graph;
pub(super) mod graph_edge_migration;
pub(super) mod text;
