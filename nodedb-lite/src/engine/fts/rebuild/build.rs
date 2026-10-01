// SPDX-License-Identifier: Apache-2.0

//! Build collection replacements from exactly one authoritative row source.

use super::{PAGE_BYTES, PAGE_RECORDS, ordinary::build_ordinary};
use crate::engine::fts::{
    FtsState, catalog::SearchDeclarationRecord, coordinator::TextMutationPermit,
    manager::CollectionReplacement,
};
use crate::engine::{
    crdt::CrdtEngine,
    document::history::ops::{current_document_page, is_bitemporal},
    strict::StrictEngine,
};
use crate::{error::LiteError, nodedb::lock_ext::LockExt, storage::engine::StorageEngine};
use nodedb_types::{Value, columnar::ColumnType};
use std::{collections::HashMap, sync::Mutex};

pub(crate) async fn build_collection_replacement<S: StorageEngine>(
    storage: &S,
    crdt: &Mutex<CrdtEngine>,
    strict: &StrictEngine<S>,
    fts: &FtsState,
    collection: &str,
    record: SearchDeclarationRecord,
    _permit: &TextMutationPermit,
) -> Result<CollectionReplacement, LiteError> {
    let mut candidate = fts
        .manager
        .lock_or_recover()
        .begin_replacement(collection, record)?;
    if let Some(schema) = strict.schema(collection) {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            let page = strict
                .text_rows_page(collection, cursor.as_deref(), PAGE_RECORDS, PAGE_BYTES)
                .await?;
            if page.entries.is_empty() {
                break;
            }
            for (key, values) in page.entries {
                let id = crate::engine::index_integration::row_id(&schema.columns, &values);
                let fields: HashMap<String, Value> = schema
                    .columns
                    .iter()
                    .zip(values)
                    .filter(|(column, _)| column.column_type == ColumnType::String)
                    .map(|(column, value)| (column.name.clone(), value))
                    .collect();
                let surrogate = fts
                    .manager
                    .lock_or_recover()
                    .reserve_surrogate(collection, &id)?;
                candidate.index_document_fields(&id, surrogate, &fields)?;
                cursor = Some(key);
            }
        }
    } else if is_bitemporal(storage, collection).await? {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            let page = current_document_page(
                storage,
                collection,
                cursor.as_deref(),
                PAGE_RECORDS,
                PAGE_BYTES,
            )
            .await?;
            let Some(last_key) = page.last_key else {
                break;
            };
            for (_, id, fields) in page.entries {
                let surrogate = fts
                    .manager
                    .lock_or_recover()
                    .reserve_surrogate(collection, &id)?;
                candidate.index_document_fields(&id, surrogate, &fields)?;
            }
            cursor = Some(last_key);
        }
    } else {
        build_ordinary(crdt, fts, collection, &mut candidate)?;
    }
    Ok(candidate)
}
