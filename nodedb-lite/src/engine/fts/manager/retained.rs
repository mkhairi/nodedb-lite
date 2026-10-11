// SPDX-License-Identifier: Apache-2.0

//! Retained-index leases, charging nothing on `aql-dev`.
//!
//! Upstream Lite charges each index a conservative per-token-occurrence lease
//! that survives retraction, because upstream nodedb-fts does not account its
//! memtable. On `aql-dev` it does: every index is built by `resident_index`
//! with a governor-bound memtable that charges the bytes of each posting and
//! term key it holds, and releases them on removal (nodedb `aql-dev`, "charge
//! resident postings to the memory governor"). That covers live writes,
//! candidate indexes and checkpoint restore. A lease on top charged the same
//! postings again: 75.8 MB for 9.0 MB resident in
//! `fts_charge_tracks_resident_postings_without_a_second_lease`, and it was
//! never released. On a 89,799-object store it refused the open below a
//! 40 GB budget.
//!
//! The type and its methods stay so callers match upstream and a rebase onto
//! upstream Lite conflicts in this file only.

use crate::engine::fts::LiteFtsIndex;
use crate::error::LiteError;
use nodedb_mem::MemoryGovernor;
use nodedb_types::Surrogate;
use std::sync::Arc;

#[derive(Default)]
pub(super) struct RetainedIndex {}

impl RetainedIndex {
    pub(super) fn reserve_base(
        &mut self,
        _governor: Arc<MemoryGovernor>,
        _key: &str,
    ) -> Result<(), LiteError> {
        Ok(())
    }

    pub(super) fn reserve_write(
        &mut self,
        _governor: Arc<MemoryGovernor>,
        _key: &str,
        _surrogate: Surrogate,
        _tokens: &[String],
    ) -> Result<(), LiteError> {
        Ok(())
    }

    pub(super) fn restored(
        _governor: Arc<MemoryGovernor>,
        _key: &str,
        _index: &LiteFtsIndex,
    ) -> Self {
        Self {}
    }
}
