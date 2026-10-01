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

pub(crate) use restore::restore_fts;
pub(crate) use serialize::serialize_fts;
pub(crate) use trust::{
    checkpoint_compatible, persist_checkpoint_complete, persist_checkpoint_incomplete,
};
pub(crate) use write::write_serialized_fts;
