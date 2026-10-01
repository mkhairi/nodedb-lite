// SPDX-License-Identifier: Apache-2.0

//! Sparse cold recovery remains independent of search field declarations.

use crate::engine::{
    document::history::ops::current_document_page,
    fts::rebuild::{PAGE_BYTES, PAGE_RECORDS},
};
use crate::{
    error::LiteError,
    nodedb::{convert::loro_value_to_document, core::types::NodeDbLite, lock_ext::LockExt},
    storage::engine::StorageEngine,
};

impl<S: StorageEngine> NodeDbLite<S> {
    pub(super) async fn rebuild_sparse_documents(&self) -> Result<(), LiteError> {
        let collections = self.crdt.lock_or_recover().collection_names();
        for collection in collections {
            if collection.starts_with("__")
                || self.is_authoritative_text_collection(&collection).await?
            {
                continue;
            }
            let mut cursor: Option<String> = None;
            loop {
                let ids = self.crdt.lock_or_recover().live_ids_page(
                    &collection,
                    cursor.as_deref(),
                    PAGE_RECORDS,
                    PAGE_BYTES,
                )?;
                if ids.is_empty() {
                    break;
                }
                for id in ids {
                    let fields = self
                        .crdt
                        .lock_or_recover()
                        .read(&collection, &id)
                        .map(|value| loro_value_to_document(&id, &value).fields);
                    if let Some(fields) = fields {
                        self.sparse_state
                            .manager
                            .lock_or_recover()
                            .index_document_fields(&collection, &id, &fields);
                    }
                    cursor = Some(id);
                }
            }
        }
        for collection in self.list_bitemporal_collections().await? {
            let mut cursor: Option<Vec<u8>> = None;
            loop {
                let page = current_document_page(
                    &*self.storage,
                    &collection,
                    cursor.as_deref(),
                    PAGE_RECORDS,
                    PAGE_BYTES,
                )
                .await?;
                let Some(last_key) = page.last_key else {
                    break;
                };
                for (_, id, fields) in page.entries {
                    self.sparse_state
                        .manager
                        .lock_or_recover()
                        .index_document_fields(&collection, &id, &fields);
                }
                cursor = Some(last_key);
            }
        }
        Ok(())
    }
}
