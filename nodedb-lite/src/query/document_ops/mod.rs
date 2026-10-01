// SPDX-License-Identifier: Apache-2.0
#[cfg(test)]
mod field_index_tests;
#[cfg(test)]
mod index_select_tests;
pub(crate) mod index_spec;
pub mod indexes;
pub mod reads;
pub mod sets;
pub mod writes;

use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

/// A collection is "strict" iff the strict engine has a schema for it.
/// Schemaless collections flow through the CRDT engine instead.
pub(crate) fn is_strict<S: StorageEngine>(engine: &LiteQueryEngine<S>, collection: &str) -> bool {
    engine.strict.schema(collection).is_some()
}
