// SPDX-License-Identifier: Apache-2.0

//! Collection admin: truncate/create_index/drop_index.

use nodedb_sql::types::query::EngineType;

use crate::error::LiteError;
use crate::query::document_ops::indexes::{self, CreateIndexRequest};
use crate::query::engine::LiteQueryEngine;
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

/// `SqlPlan::CreateIndex`. The planner refuses partial (`WHERE`) indexes;
/// those arrive through the DDL path, which reaches the same handler.
pub(super) fn create_index<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    index_name: Option<&str>,
    collection: &str,
    field: &str,
    unique: bool,
    if_not_exists: bool,
    case_insensitive: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let name = index_name.map(str::to_string);
    let collection = collection.to_string();
    let field = field.to_string();
    Ok(Box::pin(async move {
        indexes::create_index(
            engine,
            CreateIndexRequest {
                name: name.as_deref(),
                collection: &collection,
                field: &field,
                unique,
                case_insensitive,
                predicate: None,
                if_not_exists,
            },
        )
        .await
    }))
}

/// `SqlPlan::DropIndex`. Index names are unique across the database, so the
/// collection is not needed to find the index.
pub(super) fn drop_index<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    index_name: &str,
    _collection: Option<&str>,
    if_exists: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let name = index_name.to_string();
    Ok(Box::pin(async move {
        indexes::drop_index(engine, &name, if_exists).await
    }))
}
