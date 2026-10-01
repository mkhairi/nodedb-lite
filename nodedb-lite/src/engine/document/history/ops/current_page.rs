// SPDX-License-Identifier: Apache-2.0

//! Current live documents from bounded authoritative pointer pages.

use super::super::{
    key::{coll_prefix, versioned_doc_key},
    value::decode_value,
};
use crate::{error::LiteError, storage::engine::StorageEngine};
use nodedb_types::{Namespace, Value};
use std::collections::HashMap;

pub(crate) struct CurrentDocumentPage {
    pub entries: Vec<(Vec<u8>, String, HashMap<String, Value>)>,
    pub last_key: Option<Vec<u8>>,
}

pub(crate) async fn current_document_page<S: StorageEngine>(
    storage: &S,
    collection: &str,
    after_key: Option<&[u8]>,
    max_records: usize,
    max_bytes: usize,
) -> Result<CurrentDocumentPage, LiteError> {
    let prefix = coll_prefix(collection);
    let pointers = storage
        .scan_prefix_from_budgeted(
            Namespace::LatestVersion,
            &prefix,
            after_key,
            max_records,
            max_bytes,
        )
        .await?;
    let mut result = CurrentDocumentPage {
        entries: Vec::new(),
        last_key: None,
    };
    let mut used = 0usize;
    for (key, pointer) in pointers.entries {
        let id = std::str::from_utf8(&key[prefix.len()..])
            .map_err(|error| invalid(collection, format!("pointer ID is not UTF-8: {error}")))?;
        let timestamp = std::str::from_utf8(&pointer)
            .map_err(|error| invalid(collection, format!("pointer is not UTF-8: {error}")))?
            .trim()
            .parse::<i64>()
            .map_err(|error| {
                invalid(collection, format!("pointer timestamp for '{id}': {error}"))
            })?;
        let history_key = versioned_doc_key(collection, id, timestamp)?;
        let pointer_bytes = key.len().saturating_add(pointer.len());
        let remaining = max_bytes.saturating_sub(used).saturating_sub(pointer_bytes);
        let history = match storage
            .scan_prefix_from_budgeted(Namespace::DocumentHistory, &history_key, None, 1, remaining)
            .await
        {
            Err(LiteError::Backpressure { .. }) if result.last_key.is_some() => break,
            outcome => outcome?,
        };
        let Some((stored_key, value)) = history
            .entries
            .into_iter()
            .next()
            .filter(|(stored_key, _)| stored_key == &history_key)
        else {
            return Err(invalid(
                collection,
                format!("pointer for '{id}' names missing history row {timestamp}"),
            ));
        };
        used = used
            .saturating_add(pointer_bytes)
            .saturating_add(stored_key.len())
            .saturating_add(value.len());
        let version = decode_value(&value)?;
        if version.is_live() {
            let fields = if version.body.is_empty() {
                HashMap::new()
            } else {
                match nodedb_types::json_msgpack::value_from_msgpack(&version.body).map_err(
                    |error| invalid(collection, format!("current document '{id}': {error}")),
                )? {
                    Value::Object(fields) => fields,
                    other => {
                        return Err(invalid(
                            collection,
                            format!(
                                "current document '{id}' is {}, expected object",
                                other.type_name()
                            ),
                        ));
                    }
                }
            };
            result.entries.push((key.clone(), id.to_owned(), fields));
        }
        result.last_key = Some(key);
    }
    Ok(result)
}

fn invalid(collection: &str, detail: String) -> LiteError {
    LiteError::Serialization {
        detail: format!("current history for '{collection}': {detail}"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::write::versioned_put;
    use super::*;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    #[tokio::test]
    async fn current_pages_use_latest_pointer_without_valid_time_filtering() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let body = nodedb_types::json_msgpack::value_to_msgpack(&Value::Object(HashMap::from([(
            "title".into(),
            Value::String("current".into()),
        )])))
        .unwrap();
        versioned_put(&storage, "docs", "id", &body, 10, Some(1000), Some(2000))
            .await
            .unwrap();
        let page = current_document_page(&storage, "docs", None, 1, 4096)
            .await
            .unwrap();
        assert_eq!(page.entries[0].2["title"], Value::String("current".into()));
        assert!(
            current_document_page(&storage, "docs", page.last_key.as_deref(), 1, 4096)
                .await
                .unwrap()
                .last_key
                .is_none()
        );
    }
}
