// SPDX-License-Identifier: Apache-2.0

//! Ordinary document candidates read applied CRDT rows one at a time.

use super::{PAGE_BYTES, PAGE_RECORDS, check_row_budget};
use crate::engine::{
    crdt::CrdtEngine,
    fts::{FtsState, manager::CollectionReplacement},
};
use crate::{
    error::LiteError,
    nodedb::{convert::loro_value_to_document, lock_ext::LockExt},
};
use std::sync::Mutex;

pub(super) fn build_ordinary(
    crdt: &Mutex<CrdtEngine>,
    fts: &FtsState,
    collection: &str,
    candidate: &mut CollectionReplacement,
) -> Result<(), LiteError> {
    let mut cursor: Option<String> = None;
    loop {
        let ids = crdt.lock_or_recover().live_ids_page(
            collection,
            cursor.as_deref(),
            PAGE_RECORDS,
            PAGE_BYTES,
        )?;
        if ids.is_empty() {
            break;
        }
        for id in ids {
            let fields = crdt
                .lock_or_recover()
                .read(collection, &id)
                .map(|value| loro_value_to_document(&id, &value).fields);
            if let Some(fields) = fields {
                check_row_budget(collection, &id, &fields, PAGE_BYTES)?;
                let surrogate = fts
                    .manager
                    .lock_or_recover()
                    .reserve_surrogate(collection, &id)?;
                candidate.index_document_fields(&id, surrogate, &fields)?;
            }
            cursor = Some(id);
        }
    }
    Ok(())
}
