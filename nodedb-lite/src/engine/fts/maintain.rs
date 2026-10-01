// SPDX-License-Identifier: Apache-2.0

//! The one place a document write reaches the text index.
//!
//! Every write path — the `NodeDb` trait, SQL DML, rows pushed from Origin,
//! and imported CRDT deltas — keeps the text index current through these
//! functions. `outbound` is where the whole-document text is staged for
//! Origin. A write that came from Origin passes `None`, so nothing echoes
//! back to it.

use std::collections::HashMap;
use std::sync::Mutex;

use nodedb_types::Value;

use crate::engine::crdt::CrdtEngine;
use crate::engine::fts::FtsState;
use crate::error::LiteError;
use crate::nodedb::convert::loro_value_to_document;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

/// Where whole-document text is staged for Origin.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use crate::sync::FtsOutbound;

/// Targets without sync have no outbound queue. No value of this type
/// exists, so every caller there passes `None`.
#[cfg(target_arch = "wasm32")]
pub(crate) struct FtsOutbound<S>(std::convert::Infallible, std::marker::PhantomData<S>);

#[cfg(target_arch = "wasm32")]
impl<S> FtsOutbound<S> {
    fn stage_index(
        &self,
        _collection: &str,
        _doc_id: &str,
        _text: String,
    ) -> Result<(), LiteError> {
        match self.0 {}
    }

    fn stage_delete(&self, _collection: &str, _doc_id: &str) -> Result<(), LiteError> {
        match self.0 {}
    }
}

/// Index a schemaless document whole and per string field, replacing what
/// it held before, and stage its whole-document text on `outbound`.
/// Source and local index changes precede outbound Backpressure. Errors do not roll them back.
pub(crate) fn index_document<S: StorageEngine>(
    fts: &FtsState,
    outbound: Option<&FtsOutbound<S>>,
    collection: &str,
    doc_id: &str,
    fields: &HashMap<String, Value>,
) -> Result<(), LiteError> {
    let text = fts
        .manager
        .lock_or_recover()
        .index_document_fields(collection, doc_id, fields)?;
    if let Some(q) = outbound {
        q.stage_index(collection, doc_id, text)?;
    }
    Ok(())
}

/// Remove a document from every text index of its collection, and stage the
/// removal on `outbound`.
/// Source and local index changes precede outbound Backpressure. Errors do not roll them back.
pub(crate) fn remove_document<S: StorageEngine>(
    fts: &FtsState,
    outbound: Option<&FtsOutbound<S>>,
    collection: &str,
    doc_id: &str,
) -> Result<(), LiteError> {
    fts.manager
        .lock_or_recover()
        .remove_document_fields(collection, doc_id)?;
    if let Some(q) = outbound {
        q.stage_delete(collection, doc_id)?;
    }
    Ok(())
}

/// Bring the text entries of CRDT documents in line with their current
/// state: a document that exists is re-indexed, one that does not is
/// removed.
///
/// System collections (`__` prefix) carry no text index, as in the
/// cold-start rebuild.
pub(crate) fn reindex_crdt_documents<'a, S: StorageEngine>(
    fts: &FtsState,
    crdt: &Mutex<CrdtEngine>,
    outbound: Option<&FtsOutbound<S>>,
    collection: &str,
    doc_ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), LiteError> {
    if collection.starts_with("__") {
        return Ok(());
    }
    for doc_id in doc_ids {
        let current = crdt
            .lock_or_recover()
            .read(collection, doc_id)
            .map(|loro_val| loro_value_to_document(doc_id, &loro_val).fields);
        match current {
            Some(fields) => index_document(fts, outbound, collection, doc_id, &fields)?,
            None => remove_document(fts, outbound, collection, doc_id)?,
        }
    }
    Ok(())
}
