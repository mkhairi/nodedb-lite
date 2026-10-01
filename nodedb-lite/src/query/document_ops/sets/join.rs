// SPDX-License-Identifier: Apache-2.0
use super::super::writes::point_update_admitted;
use super::DocumentJoin;
use super::UpdateValue;
use super::actions::qualify_updates_with_source;
use super::source::{build_join_map, collect_ids, extract_field_str, fetch_document_value};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
/// UpdateFromJoin: update target rows matched by an equi-join with a source collection.
///
/// Execution within one Lite (single-node) transaction:
/// 1. Scan source collection and build a hash map keyed by `source_join_col`.
/// 2. Scan target collection.
/// 3. For each target row whose `target_join_col` value exists in the hash map,
///    apply the `updates` assignments (merged document: target fields + source
///    fields qualified as `<source_alias>.<field>`).
/// 4. All writes go through `point_update` so CRDT vs strict routing is preserved.
pub async fn update_from_join<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    target_collection: &str,
    source_collection: &str,
    source_alias: &str,
    target_join_col: &str,
    source_join_col: &str,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    update_from_join_coordinated(
        engine,
        None,
        DocumentJoin {
            target_collection,
            source_collection,
            source_alias,
            target_join_col,
            source_join_col,
        },
        updates,
    )
    .await
}

pub(crate) async fn update_from_join_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    join: DocumentJoin<'_>,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return update_from_join_admitted(engine, permit, join, updates).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = update_from_join_admitted(engine, guard.permit(), join, updates).await;
    guard.finish(result)
}

pub(crate) async fn update_from_join_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    join: DocumentJoin<'_>,
    updates: &[(String, UpdateValue)],
) -> Result<QueryResult, LiteError> {
    let DocumentJoin {
        target_collection,
        source_collection,
        source_alias,
        target_join_col,
        source_join_col,
    } = join;
    // Step 1: build join map from source collection.
    let source_map = build_join_map(engine, source_collection, source_join_col).await?;

    // Step 2: scan target collection to find matching rows.
    let target_ids = collect_ids(engine, target_collection).await?;

    let mut affected_n: u64 = 0;
    for doc_id in &target_ids {
        let target_val = fetch_document_value(engine, target_collection, doc_id).await?;
        let join_key = extract_field_str(&target_val, target_join_col);
        let join_key = match join_key {
            Some(k) => k,
            None => continue,
        };

        let source_val = match source_map.get(&join_key) {
            Some(v) => v,
            None => continue,
        };

        // Build merged document: target fields + source fields qualified by alias.
        let effective_updates = qualify_updates_with_source(updates, source_val, source_alias)?;

        point_update_admitted(
            engine,
            permit,
            target_collection,
            doc_id,
            &effective_updates,
        )
        .await?;
        affected_n += 1;
    }

    Ok(QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: affected_n,
        command: Some("UPDATE".into()),
    })
}

#[cfg(test)]
mod tests {
    use crate::NodeDbLite;
    use crate::PagedbStorageMem;

    async fn make_db() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        NodeDbLite::open(storage).await.unwrap()
    }

    /// update_from_join on two empty collections returns 0 rows_affected without error.
    #[tokio::test]
    async fn update_from_join_empty_collections() {
        let db = make_db().await;
        let result = super::update_from_join(
            &db.query_engine,
            "target_ufj",
            "source_ufj",
            "s",
            "tid",
            "sid",
            &[],
        )
        .await
        .unwrap();
        assert_eq!(result.rows_affected, 0);
    }

    #[tokio::test]
    async fn update_from_join_borrows_existing_admission() -> Result<(), Box<dyn std::error::Error>>
    {
        use nodedb_types::value::Value;
        let db = make_db().await;
        for collection in ["source", "target"] {
            let bytes =
                zerompk::to_msgpack_vec(&Value::Object(std::collections::HashMap::from([
                    ("id".into(), Value::String("one".into())),
                    ("body".into(), Value::String("original".into())),
                ])))?;
            crate::query::document_ops::writes::point_put(
                &db.query_engine,
                collection,
                "one",
                &bytes,
            )
            .await?;
        }
        let updates = vec![(
            "body".into(),
            super::super::UpdateValue::Literal(zerompk::to_msgpack_vec(&Value::String(
                "joined".into(),
            ))?),
        )];
        let guard = db.query_engine.fts_state.admit_mutation().await;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            super::update_from_join_coordinated(
                &db.query_engine,
                Some(guard.permit()),
                super::DocumentJoin {
                    target_collection: "target",
                    source_collection: "source",
                    source_alias: "s",
                    target_join_col: "id",
                    source_join_col: "id",
                },
                &updates,
            ),
        )
        .await?;
        let result = guard.finish(result)?;
        assert_eq!(result.rows_affected, 1);
        let fields =
            super::super::source::fetch_document_value(&db.query_engine, "target", "one").await?;
        assert_eq!(fields.get("body"), Some(&Value::String("joined".into())));
        Ok(())
    }
}
