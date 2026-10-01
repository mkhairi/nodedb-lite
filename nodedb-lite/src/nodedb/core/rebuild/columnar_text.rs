// SPDX-License-Identifier: Apache-2.0

//! Legacy automatic columnar text and system geohash recovery.

use crate::{error::LiteError, nodedb::core::types::NodeDbLite, storage::engine::StorageEngine};

impl<S: StorageEngine> NodeDbLite<S> {
    pub(super) async fn rebuild_columnar_text(&self) -> Result<(), LiteError> {
        for collection in self.columnar.collection_names() {
            let Some(schema) = self.columnar.schema(&collection) else {
                continue;
            };
            let profile = self.columnar.profile(&collection);
            for values in self.columnar.list_rows(&collection).await? {
                let row_id = crate::engine::index_integration::row_id(&schema.columns, &values);
                crate::engine::index_integration::index_row_text(
                    &collection,
                    &row_id,
                    &schema.columns,
                    &values,
                    &self.fts_state.manager,
                )?;
                crate::engine::index_integration::index_geohash(
                    &collection,
                    &row_id,
                    &schema,
                    profile.as_ref(),
                    &values,
                    &self.fts_state.manager,
                )?;
            }
        }
        Ok(())
    }
}
