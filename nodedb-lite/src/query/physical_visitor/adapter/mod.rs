// SPDX-License-Identifier: Apache-2.0

//! Physical operation routing and admission context.

mod array;
mod columnar;
mod crdt;
mod document;
mod document_index;
mod graph;
mod graph_resolve;
mod kv;
mod meta;
pub(super) mod policy;
mod query;
mod spatial;
mod timeseries;

mod surrogates;
mod visitor;

pub(crate) use surrogates::execute_surrogate_scan;
pub(crate) use visitor::{LiteDataPlaneVisitor, LitePhysicalFut};
