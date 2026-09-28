// SPDX-License-Identifier: Apache-2.0

//! `UPDATE ... FROM` lowering.

use std::collections::HashMap;

use nodedb_sql::types::SqlPlan;
use nodedb_sql::types::filter::Filter;
use nodedb_sql::types::query::EngineType;
use nodedb_sql::types_expr::SqlExpr;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::document_ops::sets::{collect_ids_pub, fetch_document_value_pub};
use crate::query::document_ops::writes::point_update;
use crate::query::engine::LiteQueryEngine;
use crate::query::value_utils::value_to_string;
use crate::storage::engine::StorageEngine;

use crate::query::visitor::adapter::LiteFut;

use super::rows::{convert_assignments, resolve_updates_with_source, result_to_maps};

// ── UpdateFrom ───────────────────────────────────────────────────────────────

/// `UPDATE target SET ... FROM source WHERE target.col = source.col`.
#[allow(clippy::too_many_arguments)]
pub(in crate::query::visitor) fn lower_update_from<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    _engine_type: EngineType,
    source: &SqlPlan,
    target_join_col: &str,
    source_join_col: &str,
    assignments: &[(String, SqlExpr)],
    _target_filters: &[Filter],
    _returning: bool,
) -> Result<LiteFut<'a>, LiteError> {
    let target = collection.to_string();
    let source = source.clone();
    let t_join = target_join_col.to_string();
    let s_join = source_join_col.to_string();
    let updates = convert_assignments(assignments)?;

    Ok(Box::pin(async move {
        let source_result = engine.execute_plan(&source).await?;
        let source_maps = result_to_maps(source_result);

        let mut source_index: HashMap<String, HashMap<String, Value>> = HashMap::new();
        for row in source_maps {
            if let Some(key_val) = row.get(&s_join) {
                source_index.insert(value_to_string(key_val), row);
            }
        }

        let target_ids = collect_ids_pub(engine, &target).await?;
        let mut affected: u64 = 0;

        for doc_id in &target_ids {
            let target_val = fetch_document_value_pub(engine, &target, doc_id).await?;
            let join_key = match target_val.get(&t_join).map(value_to_string) {
                Some(k) => k,
                None => continue,
            };

            let source_val = match source_index.get(&join_key) {
                Some(v) => v.clone(),
                None => continue,
            };

            let resolved = resolve_updates_with_source(&updates, &source_val)?;
            point_update(engine, &target, doc_id, &resolved).await?;
            affected += 1;
        }

        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: affected,
            command: Some("UPDATE".into()),
        })
    }))
}
