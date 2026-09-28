// SPDX-License-Identifier: Apache-2.0

//! Inverted text index maintenance for document writes.
//!
//! A schemaless document is indexed twice over: its whole-document text
//! (searched when a caller names no field) and each top-level string field
//! under `"{collection}:{field}"`. The work is done by `engine::fts::maintain`,
//! shared with every other write path.

use std::collections::HashMap;

use nodedb_types::Value;

use crate::engine::fts::maintain;
use crate::engine::fts::maintain::FtsOutbound;
use crate::error::LiteError;
use crate::nodedb::core::types::NodeDbLite;
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// The queue a local text-index change is staged on for Origin, or
    /// `None` when there is none or the sync gate keeps the document local.
    fn fts_outbound_for(
        &self,
        collection: &str,
        fields: Option<&HashMap<String, Value>>,
    ) -> Option<&FtsOutbound<S>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let gated_out = fields.is_some_and(|f| !self.should_sync_doc(collection, f));
            if gated_out {
                return None;
            }
            self.fts_outbound.as_deref()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (collection, fields);
            None
        }
    }

    /// Update the inverted text index after a local document write, and stage
    /// the whole-document text for Origin — unless the sync gate keeps the
    /// document local-only.
    ///
    /// A failure here fails the write: nothing re-indexes the gap afterwards.
    pub(crate) fn index_document_text(
        &self,
        collection: &str,
        doc_id: &str,
        fields: &HashMap<String, Value>,
    ) -> Result<(), LiteError> {
        maintain::index_document(
            &self.fts_state,
            self.fts_outbound_for(collection, Some(fields)),
            collection,
            doc_id,
            fields,
        )
    }

    /// Remove a document from every text index of its collection, and stage
    /// the removal for Origin.
    pub(crate) fn remove_document_text(
        &self,
        collection: &str,
        doc_id: &str,
    ) -> Result<(), LiteError> {
        maintain::remove_document(
            &self.fts_state,
            self.fts_outbound_for(collection, None),
            collection,
            doc_id,
        )
    }

    /// Bring the text entries of `doc_ids` in line with their current CRDT
    /// state without staging anything for Origin. Used for changes that came
    /// from Origin or a peer.
    pub(crate) fn reindex_documents_local<'a>(
        &self,
        collection: &str,
        doc_ids: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), LiteError> {
        maintain::reindex_crdt_documents(
            &self.fts_state,
            &self.crdt,
            None::<&FtsOutbound<S>>,
            collection,
            doc_ids,
        )
    }
}
