// SPDX-License-Identifier: Apache-2.0

//! Cold-start index rebuild helpers for `NodeDbLite`.
//!
//! Each module handles one index family.
//!
//! - `text` — Ordinary and authoritative FTS recovery.
//! - `columnar_text` — Columnar FTS and system geohash recovery.
//! - `sparse` — Sparse recovery from CRDT and DocumentHistory.
//! - `spatial` — Spatial recovery from CRDT geometry fields.
//! - `graph` — CSR adjacency rebuild (CRDT + Namespace::Graph KV + GraphHistory)
//! - `graph_edge_migration` — one-time legacy edge CRDT key rewrite

mod columnar_text;
pub(super) mod graph;
pub(super) mod graph_edge_migration;
mod sparse;
mod spatial;
pub(super) mod text;
