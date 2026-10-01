// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::truncate::{clear_spatial, truncated};
use crate::storage::engine::StorageEngine;
use crate::storage::engine::WriteOp;
use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

/// Truncate: delete ALL documents in a collection, then the overlays they
/// fed: the FTS index, the sparse-vector postings, the R-tree entries, and
/// every vector bucket of the collection. Answers with the bare `TRUNCATE`
/// tag, never a row count.
pub async fn truncate<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    truncate_coordinated(engine, None, collection).await
}

pub(crate) async fn truncate_coordinated<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: Option<&crate::engine::fts::coordinator::TextMutationPermit>,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    if let Some(permit) = permit {
        return truncate_admitted(engine, permit, collection).await;
    }
    let guard = engine.fts_state.admit_mutation().await;
    let result = truncate_admitted(engine, guard.permit(), collection).await;
    guard.finish(result)
}

pub(crate) async fn truncate_admitted<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    permit: &crate::engine::fts::coordinator::TextMutationPermit,
    collection: &str,
) -> Result<QueryResult, LiteError> {
    let _permit = permit;
    crate::engine::fts::checkpoint::persist_checkpoint_incomplete(engine.storage.as_ref()).await?;
    if is_strict(engine, collection) {
        let prefix = format!("{collection}:");
        let all_entries = engine
            .storage
            .scan_prefix(Namespace::Strict, prefix.as_bytes())
            .await?;
        let mut ops: Vec<WriteOp> = Vec::with_capacity(all_entries.len());
        for (key, _) in all_entries {
            ops.push(WriteOp::Delete {
                ns: Namespace::Strict,
                key,
            });
        }
        // Each deleted row takes its index entries with it in the same batch.
        engine.strict.commit(collection, ops).await?;
    } else {
        let ids = {
            let mut crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
            let ids = crdt.list_ids(collection);
            crdt.clear_collection(collection)
                .map_err(|e| LiteError::Storage {
                    detail: e.to_string(),
                })?;
            ids
        };
        let mut sparse = engine
            .sparse_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?;
        for id in &ids {
            sparse.remove_document_all_fields(collection, id);
        }
    }
    {
        let mut manager = engine
            .fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?;
        manager.remove_collection_postings(collection);
        manager.ensure_declaration_indices()?;
    }
    clear_spatial(engine, collection)?;
    crate::query::physical_visitor::clear_collection_indexes(&engine.vector_state, collection)
        .await?;
    Ok(truncated())
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn truncate_keeps_declared_fields_and_marks_checkpoint_incomplete()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::engine::fts::catalog::SearchDeclaration;
        use nodedb_types::Value;
        use std::collections::BTreeMap;
        let storage = crate::PagedbStorageMem::open_in_memory().await?;
        let db = crate::NodeDbLite::open(storage).await?;
        let declaration = SearchDeclaration {
            name: "fts_docs".into(),
            fields: vec!["body".into()],
            analyzer: "standard".into(),
            fuzzy: false,
        };
        {
            let mut manager = db
                .query_engine
                .fts_state
                .manager
                .lock()
                .map_err(|_| crate::error::LiteError::LockPoisoned)?;
            let record = manager.next_declaration_record("docs", &Some(declaration.clone()))?;
            manager.load_declarations(BTreeMap::from([("docs".into(), record)]));
            manager.ensure_declaration_indices()?;
        }
        let bytes = zerompk::to_msgpack_vec(&Value::Object(std::collections::HashMap::from([(
            "body".into(),
            Value::String("searchable".into()),
        )])))?;
        super::super::point_put(&db.query_engine, "docs", "one", &bytes).await?;
        let guard = db.query_engine.fts_state.admit_mutation().await;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            super::truncate_coordinated(&db.query_engine, Some(guard.permit()), "docs"),
        )
        .await?;
        guard.finish(result)?;
        let revisions = {
            let manager = db
                .query_engine
                .fts_state
                .manager
                .lock()
                .map_err(|_| crate::error::LiteError::LockPoisoned)?;
            assert_eq!(manager.declaration_for("docs"), Some(&declaration));
            assert!(
                manager
                    .checkpoint_data()
                    .0
                    .contains_key(&crate::engine::fts::manager::index_key("docs", "body"))
            );
            manager.declaration_revisions()
        };
        assert!(
            !crate::engine::fts::checkpoint::checkpoint_compatible(
                db.query_engine.storage.as_ref(),
                &revisions
            )
            .await?
        );
        assert!(
            !db.query_engine
                .crdt
                .lock()
                .map_err(|_| crate::error::LiteError::LockPoisoned)?
                .exists("docs", "one")
        );
        Ok(())
    }
}
