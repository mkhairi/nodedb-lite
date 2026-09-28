// SPDX-License-Identifier: Apache-2.0

//! Per-collection FTS manager for Lite, split by responsibility:
//! `registry` (state, keys, errors, checkpoint access), `index` (writes),
//! `search` (BM25 reads), `phrase` (exact phrase reads).

pub mod index;
pub mod phrase;
pub mod registry;
pub mod search;

pub use index::whole_document_text;
pub(crate) use registry::resident_index;
#[cfg(test)]
pub(crate) use registry::test_governor;
pub use registry::{FtsCollectionManager, FtsResult, index_key};
