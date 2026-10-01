// SPDX-License-Identifier: Apache-2.0

//! Vector-primary collection DML: vector_primary_insert/
//! vector_primary_delete/vector_primary_update/vector_primary_truncate.

use nodedb_sql::{
    VectorPrimaryDeleteVisitArgs, VectorPrimaryInsertVisitArgs, VectorPrimaryUpdateVisitArgs,
};

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::visitor::vector_primary::{
    lower_vector_primary_delete, lower_vector_primary_insert, lower_vector_primary_truncate,
    lower_vector_primary_update,
};
use crate::storage::engine::StorageEngine;

use super::LiteFut;

pub(super) fn vector_primary_insert<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    args: VectorPrimaryInsertVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    lower_vector_primary_insert(engine, permit, args)
}

pub(super) fn vector_primary_delete<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    args: VectorPrimaryDeleteVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    lower_vector_primary_delete(engine, permit, args)
}

pub(super) fn vector_primary_truncate<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
    field: &str,
    restart_identity: bool,
) -> Result<LiteFut<'a>, LiteError> {
    lower_vector_primary_truncate(engine, permit, collection, field, restart_identity)
}

pub(super) fn vector_primary_update<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    permit: &'a crate::engine::fts::coordinator::TextMutationPermit,
    args: VectorPrimaryUpdateVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    lower_vector_primary_update(engine, permit, args)
}
