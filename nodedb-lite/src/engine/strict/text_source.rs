// SPDX-License-Identifier: Apache-2.0

//! Bounded authoritative strict rows for text rebuilds.

use super::{crud::decode_tuple, engine::StrictEngine};
use crate::{error::LiteError, storage::engine::StorageEngine};
use nodedb_types::{Namespace, Value};

pub(crate) struct StrictTextPage {
    pub entries: Vec<(Vec<u8>, Vec<Value>)>,
}

impl<S: StorageEngine> StrictEngine<S> {
    pub(crate) async fn text_rows_page(
        &self,
        collection: &str,
        after_key: Option<&[u8]>,
        max_records: usize,
        max_bytes: usize,
    ) -> Result<StrictTextPage, LiteError> {
        let state = self.get_state(collection)?;
        let prefix = format!("{collection}:");
        let page = self
            .storage
            .scan_prefix_from_budgeted(
                Namespace::Strict,
                prefix.as_bytes(),
                after_key,
                max_records,
                max_bytes,
            )
            .await?;
        let mut entries = Vec::with_capacity(page.entries.len());
        for (key, bytes) in page.entries {
            entries.push((key, decode_tuple(&state, &bytes)?));
        }
        Ok(StrictTextPage { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::pagedb_storage::PagedbStorageMem;
    use nodedb_types::columnar::{ColumnDef, ColumnType, StrictSchema};
    use std::sync::Arc;

    #[tokio::test]
    async fn strict_text_pages_preserve_primary_key_order_and_current_projection() {
        let storage = Arc::new(PagedbStorageMem::open_in_memory().await.unwrap());
        let engine = StrictEngine::new(storage);
        let schema = StrictSchema::new(vec![
            ColumnDef::required("id", ColumnType::Int64).with_primary_key(),
            ColumnDef::nullable("body", ColumnType::String),
        ])
        .unwrap();
        engine.create_collection("rows", schema).await.unwrap();
        for id in [1, 2] {
            engine
                .insert("rows", &[Value::Integer(id), Value::String("alpha".into())])
                .await
                .unwrap();
        }
        let first = engine.text_rows_page("rows", None, 1, 4096).await.unwrap();
        assert_eq!(first.entries[0].1[0], Value::Integer(1));
        let second = engine
            .text_rows_page("rows", Some(&first.entries[0].0), 1, 4096)
            .await
            .unwrap();
        assert_eq!(second.entries[0].1[0], Value::Integer(2));
        assert!(
            engine
                .text_rows_page("rows", Some(&second.entries[0].0), 1, 4096)
                .await
                .unwrap()
                .entries
                .is_empty()
        );
    }
}
