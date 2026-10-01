// SPDX-License-Identifier: Apache-2.0

//! Integration tests for NodeDB-Lite.
//!
//! Tests the full stack: StorageEngine → Engines → NodeDbLite → NodeDb trait.
//! Performance/scale workloads live in `nodedb-bench/benches/`.

#[path = "integration/mod.rs"]
mod integration;
