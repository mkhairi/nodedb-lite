// SPDX-License-Identifier: Apache-2.0

//! Vector similarity search over one HNSW index, shared by `NodeDbLite` and
//! `LiteDataPlaneVisitor`.

mod filter;
mod hydrate;
pub(crate) mod lazy_load;
mod run;
#[cfg(test)]
mod tests;

pub(crate) use run::run_vector_search;
