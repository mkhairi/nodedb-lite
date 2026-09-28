// SPDX-License-Identifier: Apache-2.0

//! Lowerings for direct-to-engine CRUD ops: `Scan`, `PointGet`, `Insert`,
//! `Upsert`, `Update`, `Delete`, `ConstantResult`. These dispatch straight to
//! `LiteQueryEngine` methods without intermediate planning helpers.

use nodedb_sql::ScanVisitArgs;
use nodedb_sql::types::filter::Filter;
use nodedb_sql::types::query::EngineType;
use nodedb_sql::types::{SqlValue, WriteRoute};
use nodedb_sql::types_expr::SqlExpr;

use crate::error::LiteError;
use crate::index::IndexEngine;
use crate::query::document_ops::index_reads::index_range_fetch;
use crate::query::engine::LiteQueryEngine;
use crate::query::visitor::index_range::index_range;
use crate::query::visitor::scan_post::{ScanPostArgs, apply_scan_post_processing};
use crate::storage::engine::StorageEngine;

use super::visitor::LiteFut;

pub(super) fn lower_constant_result<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    columns: &[String],
    values: &[SqlValue],
) -> Result<LiteFut<'a>, LiteError> {
    let columns = columns.to_vec();
    let values = values.to_vec();
    Ok(Box::pin(async move {
        engine.execute_constant_result(&columns, &values).await
    }))
}

pub(super) fn lower_scan<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    args: &ScanVisitArgs<'_>,
) -> Result<LiteFut<'a>, LiteError> {
    let collection = args.collection.to_string();
    let engine_type = args.engine;
    let filters = args.filters.to_vec();
    let sort_keys = args.sort_keys.to_vec();
    let window_functions = args.window_functions.to_vec();
    let projection = args.projection.to_vec();
    let limit = args.limit;
    let offset = args.offset;
    let distinct = args.distinct;
    let index_engine = match engine_type {
        EngineType::DocumentSchemaless => Some(IndexEngine::Document),
        EngineType::DocumentStrict => Some(IndexEngine::Strict),
        EngineType::KeyValue => Some(IndexEngine::KeyValue),
        _ => None,
    };
    let range =
        index_engine.and_then(|kind| index_range(&engine.indexes, &collection, kind, &filters));
    Ok(Box::pin(async move {
        // A bounded indexed field lets the index list the candidate rows;
        // every filter below still applies to them.
        let indexed = match &range {
            Some(r) => {
                index_range_fetch(engine, &r.def, r.lower.as_ref(), r.upper.as_ref()).await?
            }
            None => None,
        };
        let raw = match indexed {
            Some(raw) => raw,
            None => engine.execute_scan(&collection, &engine_type).await?,
        };
        apply_scan_post_processing(
            raw,
            ScanPostArgs {
                filters: &filters,
                sort_keys: &sort_keys,
                window_specs: &window_functions,
                projection: &projection,
                sequences: engine.sequences(),
                limit,
                offset,
                distinct,
            },
        )
    }))
}

pub(super) fn lower_point_get<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    engine_type: EngineType,
    key_column: &str,
    key_value: &SqlValue,
) -> Result<LiteFut<'a>, LiteError> {
    let collection = collection.to_string();
    let key_column = key_column.to_string();
    let key_value = key_value.clone();
    Ok(Box::pin(async move {
        engine
            .execute_point_get(&collection, &engine_type, &key_column, &key_value)
            .await
    }))
}

pub(super) fn lower_insert<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    engine_type: EngineType,
    route: WriteRoute,
    rows: &[Vec<(String, SqlValue)>],
    if_absent: bool,
    primary_key: Option<&str>,
) -> Result<LiteFut<'a>, LiteError> {
    let collection = collection.to_string();
    let rows = rows.to_vec();
    let primary_key = primary_key.map(str::to_string);
    Ok(Box::pin(async move {
        engine
            .execute_insert(
                &collection,
                &engine_type,
                route,
                &rows,
                if_absent,
                primary_key.as_deref(),
            )
            .await
    }))
}

pub(super) fn lower_update<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    engine_type: EngineType,
    assignments: &[(String, SqlExpr)],
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<LiteFut<'a>, LiteError> {
    let collection = collection.to_string();
    let assignments = assignments.to_vec();
    let filters = filters.to_vec();
    let target_keys = target_keys.to_vec();
    Ok(Box::pin(async move {
        engine
            .execute_update(
                &collection,
                &engine_type,
                &assignments,
                &filters,
                &target_keys,
            )
            .await
    }))
}

pub(super) fn lower_delete<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    engine_type: EngineType,
    filters: &[Filter],
    target_keys: &[SqlValue],
) -> Result<LiteFut<'a>, LiteError> {
    let collection = collection.to_string();
    let filters = filters.to_vec();
    let target_keys = target_keys.to_vec();
    Ok(Box::pin(async move {
        engine
            .execute_delete(&collection, &engine_type, &filters, &target_keys)
            .await
    }))
}
