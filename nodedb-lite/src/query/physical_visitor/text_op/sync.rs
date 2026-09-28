// SPDX-License-Identifier: Apache-2.0

//! `FtsIndexDoc` / `FtsDeleteDoc`: index frames keyed by an Origin surrogate.

use std::sync::Arc;

use nodedb_types::Surrogate;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::adapter::LitePhysicalFut;

fn affected(rows_affected: u64) -> QueryResult {
    QueryResult {
        columns: vec![],
        rows: vec![],
        rows_affected,
        command: None,
    }
}

/// `TextOp::FtsIndexDoc`: index `text` into the whole-document index.
pub(super) fn fts_index_doc<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    surrogate: Surrogate,
    text: &str,
) -> LitePhysicalFut<'a> {
    let collection = collection.to_owned();
    let text = text.to_owned();
    let fts_state = Arc::clone(&engine.fts_state);
    #[cfg(not(target_arch = "wasm32"))]
    let fts_outbound = engine.fts_outbound.as_ref().map(Arc::clone);
    Box::pin(async move {
        // On Lite the surrogate space is internal to FtsCollectionManager.
        // We use `text` as the string doc_id (stable across frames for the
        // same document). We also register the Origin surrogate → Lite doc_id
        // mapping so FtsDeleteDoc can resolve it precisely.
        let mut mgr = fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?;
        mgr.index_document(&collection, &text, &text)?;
        mgr.register_origin_surrogate(surrogate, &text);
        drop(mgr);
        // Stage for durable sync outbound (SQL path — no await needed).
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(q) = fts_outbound {
            q.stage_index(&collection, &text, text.clone());
        }
        Ok(affected(1))
    })
}

/// `TextOp::FtsDeleteDoc`: remove the document an Origin surrogate names.
pub(super) fn fts_delete_doc<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    surrogate: Surrogate,
) -> LitePhysicalFut<'a> {
    let collection = collection.to_owned();
    let fts_state = Arc::clone(&engine.fts_state);
    #[cfg(not(target_arch = "wasm32"))]
    let fts_outbound = engine.fts_outbound.as_ref().map(Arc::clone);
    Box::pin(async move {
        let mut mgr = fts_state
            .manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?;
        let removed_doc_id = mgr.remove_by_origin_surrogate(&collection, surrogate)?;
        drop(mgr);
        // Stage delete for durable sync outbound (SQL path — no await needed).
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(q), Some(doc_id)) = (fts_outbound, removed_doc_id.as_deref()) {
            q.stage_delete(&collection, doc_id);
        }
        Ok(affected(u64::from(removed_doc_id.is_some())))
    })
}
