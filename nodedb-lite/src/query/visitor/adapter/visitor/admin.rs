// SPDX-License-Identifier: Apache-2.0

//! Collection admin: truncate/create_index/drop_index.

use nodedb_sql::types::query::EngineType;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::document_ops::index_spec::{IndexSpecPlan, prepare_index_spec, write_index_spec};
use crate::query::engine::LiteQueryEngine;
use crate::query::visitor::adapter::basic::{lower_create_index, lower_drop_index};
use crate::storage::engine::StorageEngine;

use super::LiteFut;

/// `SqlPlan::Truncate`: clear the collection through the engine the planner
/// resolved, then apply `RESTART IDENTITY`.
pub(super) fn truncate<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    engine_type: EngineType,
    restart_identity: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let collection = collection.to_string();
    Ok(Box::pin(async move {
        let result =
            crate::query::truncate::truncate_engine(engine, &collection, engine_type).await?;
        crate::query::truncate::restart_identity(engine, &collection, restart_identity);
        Ok(result)
    }))
}

/// `SqlPlan::CreateIndex`: refuse a duplicate, then index the collection.
///
/// A schemaless collection persists the spec and builds in-memory postings,
/// which index lookups read. It writes no Meta index entries. Any other
/// collection backfills the Meta index entries as before.
pub(super) fn create_index<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    index_name: Option<&str>,
    collection: &str,
    field: &str,
    unique: bool,
    if_not_exists: bool,
    case_insensitive: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let index_name = index_name.map(str::to_string);
    let collection = collection.to_string();
    let field = field.to_string();
    Ok(Box::pin(async move {
        // Refuse a duplicate before any write, so a refused statement has
        // no side effect.
        let plan = prepare_index_spec(
            engine,
            index_name.as_deref(),
            &collection,
            &field,
            unique,
            case_insensitive,
            if_not_exists,
        )
        .await?;
        match plan {
            IndexSpecPlan::NotSchemaless => {
                lower_create_index(engine, &collection, &field, unique, case_insensitive)?.await
            }
            IndexSpecPlan::Exists => Ok(QueryResult::empty()),
            IndexSpecPlan::New(pending) => {
                write_index_spec(engine, pending).await?;
                Ok(QueryResult::empty())
            }
        }
    }))
}

pub(super) fn drop_index<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    index_name: &str,
    collection: Option<&str>,
    _if_exists: bool,
) -> Result<LiteFut<'a>, LiteError> {
    lower_drop_index(engine, index_name, collection)
}
