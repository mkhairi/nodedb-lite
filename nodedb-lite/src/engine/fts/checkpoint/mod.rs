// SPDX-License-Identifier: Apache-2.0

//! Checkpoint serialization and restoration for
//! [`FtsCollectionManager`](crate::engine::fts::FtsCollectionManager).
//!
//! Persists the full in-memory FTS state so that a cold open can load the
//! index without re-tokenizing source documents. `format` documents the
//! storage layout.

mod format;
mod restore;
mod serialize;
mod trust;
mod write;

pub(crate) use format::COLLECTIONS_KEY;
pub(crate) use restore::restore_fts;
#[cfg(test)]
pub(crate) use serialize::serialize_fts;
pub(crate) use serialize::{FtsFlush, catalog_changed, plan_fts};
pub(crate) use trust::{
    CHECKPOINT_STATE_KEY, checkpoint_compatible, encode_checkpoint_complete,
    encode_checkpoint_incomplete, persist_checkpoint_complete, persist_checkpoint_incomplete,
};
pub(crate) use write::write_planned_fts;
