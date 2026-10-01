// SPDX-License-Identifier: Apache-2.0

//! Physical visitor admission context and operation routing.

use std::future::Future;
use std::pin::Pin;

use nodedb_physical::PhysicalTaskVisitor;
use nodedb_physical::physical_plan::{
    ArrayOp, ColumnarOp, CrdtOp, DocumentOp, GraphOp, KvOp, MetaOp, QueryOp, SpatialOp, TextOp,
    TimeseriesOp, VectorOp,
};
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::{array, columnar, crdt, document, graph, kv, meta, query, spatial, timeseries};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type LitePhysicalFut<'a> =
    Pin<Box<dyn Future<Output = Result<QueryResult, LiteError>> + Send + 'a>>;

#[cfg(target_arch = "wasm32")]
pub(crate) type LitePhysicalFut<'a> =
    Pin<Box<dyn Future<Output = Result<QueryResult, LiteError>> + 'a>>;

pub(crate) struct LiteDataPlaneVisitor<'a, S: StorageEngine> {
    pub(crate) engine: &'a LiteQueryEngine<S>,
    pub(crate) permit: Option<&'a crate::engine::fts::coordinator::TextMutationPermit>,
}

impl<'a, S: StorageEngine> LiteDataPlaneVisitor<'a, S> {
    pub(crate) fn new(engine: &'a LiteQueryEngine<S>) -> Self {
        Self {
            engine,
            permit: None,
        }
    }
}

impl<'a, S: StorageEngine + 'a> PhysicalTaskVisitor for LiteDataPlaneVisitor<'a, S> {
    type Output = LitePhysicalFut<'a>;
    type Error = LiteError;

    fn vector(&mut self, op: &VectorOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        match self.permit {
            Some(permit) => {
                super::super::vector_op::execute_vector_op_admitted(self.engine, Some(permit), op)
            }
            None => super::super::vector_op::execute_vector_op(self.engine, op),
        }
    }

    fn array(&mut self, op: &ArrayOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        array::dispatch(self.engine, op)
    }

    fn text(&mut self, op: &TextOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        match self.permit {
            Some(permit) => {
                super::super::text_op::execute_text_op_admitted(self.engine, Some(permit), op)
            }
            None => super::super::text_op::execute_text_op(self.engine, op),
        }
    }

    fn document(&mut self, op: &DocumentOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        document::dispatch(self.engine, self.permit, op)
    }

    fn kv(&mut self, op: &KvOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        kv::dispatch(self.engine, op)
    }

    fn crdt(&mut self, op: &CrdtOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        crdt::dispatch(self.engine, self.permit, op)
    }

    fn meta(&mut self, op: &MetaOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        meta::dispatch(self.engine, self.permit, op)
    }

    fn columnar(&mut self, op: &ColumnarOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        columnar::dispatch(self.engine, self.permit, op)
    }

    fn timeseries(&mut self, op: &TimeseriesOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        timeseries::dispatch(self.engine, op)
    }

    fn spatial(&mut self, op: &SpatialOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        spatial::dispatch(self.engine, op)
    }

    fn graph(&mut self, op: &GraphOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        graph::dispatch(self.engine, op)
    }

    fn query(&mut self, op: &QueryOp) -> Result<LitePhysicalFut<'a>, LiteError> {
        query::dispatch(self.engine, op)
    }

    fn cluster_array(
        &mut self,
        _op: &nodedb_physical::physical_plan::ClusterArrayOp,
    ) -> Result<LitePhysicalFut<'a>, LiteError> {
        unreachable!(
            "ClusterArray plans are coordinator-only; Lite never sets \
             cluster_enabled so its SQL planner cannot produce this variant"
        )
    }

    fn cluster_event(
        &mut self,
        _op: &nodedb_physical::physical_plan::ClusterEventOp,
    ) -> Result<LitePhysicalFut<'a>, LiteError> {
        // ClusterEvent (topic publish / stream consume) is a coordinator-only
        // cluster operation constructed by the event/CDC/topic subsystem — not
        // just the SQL planner — so unlike `cluster_array` it is not provably
        // unreachable here. Lite is single-node and links to Origin via Loro; it
        // never runs cluster event plans, so reject it explicitly (matching every
        // other unsupported-in-Lite op) rather than panic on a reachable path.
        Err(LiteError::Unsupported {
            detail: "ClusterEvent (topic publish / stream consume) is a \
                     coordinator-only cluster operation; unsupported on the \
                     single-node Lite engine"
                .into(),
        })
    }
}
