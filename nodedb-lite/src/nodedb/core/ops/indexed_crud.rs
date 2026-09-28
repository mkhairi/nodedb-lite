// SPDX-License-Identifier: Apache-2.0

//! Indexed CRUD for strict and columnar collections: combines base-engine
//! writes with secondary-index maintenance (vector, spatial, text, B-tree)
//! and HTAP materialized-view replication.

use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::nodedb::core::types::NodeDbLite;
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    // -- Indexed CRUD for strict/columnar collections --

    /// Insert a row into a strict collection and update secondary indexes.
    ///
    /// Combines `StrictEngine.insert()` with `index_row()` for geometry,
    /// vector, and text columns.
    pub async fn strict_insert(
        &self,
        collection: &str,
        values: &[nodedb_types::value::Value],
    ) -> NodeDbResult<()> {
        let schema = self.strict.schema(collection).ok_or_else(|| {
            NodeDbError::storage(format!("strict collection '{collection}' not found"))
        })?;

        // Insert into storage. `StrictEngine` is interior-mutable; await directly.
        self.strict
            .insert(collection, values)
            .await
            .map_err(NodeDbError::storage)?;

        // Build a row_id string from the PK value for index keying.
        let row_id = crate::engine::index_integration::row_id(&schema.columns, values);

        // Update secondary indexes.
        crate::engine::index_integration::index_row(
            collection,
            &row_id,
            &schema.columns,
            values,
            &self.spatial,
            &self.fts_state.manager,
        )?;
        crate::engine::index_integration::index_row_vectors(
            &self.vector_state,
            collection,
            &row_id,
            &schema.columns,
            values,
        )
        .await?;

        // Replicate to materialized columnar views (HTAP CDC).
        self.htap
            .replicate_insert(collection, values, &self.columnar);

        Ok(())
    }

    /// Delete a row from a strict collection and clean up text indexes.
    pub async fn strict_delete(
        &self,
        collection: &str,
        pk: &nodedb_types::value::Value,
    ) -> NodeDbResult<bool> {
        let schema = self.strict.schema(collection).ok_or_else(|| {
            NodeDbError::storage(format!("strict collection '{collection}' not found"))
        })?;

        // The same row id the insert indexed the row under.
        let row_id = crate::engine::index_integration::pk_row_id(pk);

        // Remove text and vector index entries before deleting the row.
        crate::engine::index_integration::deindex_row_text(
            collection,
            &row_id,
            &self.fts_state.manager,
        )?;
        crate::engine::index_integration::deindex_row_vectors(
            &self.vector_state,
            collection,
            &row_id,
            &schema.columns,
        )
        .await?;

        // Replicate delete to materialized columnar views (HTAP CDC).
        self.htap.replicate_delete(collection, pk, &self.columnar);

        self.strict
            .delete(collection, pk)
            .await
            .map_err(NodeDbError::storage)
    }

    /// Insert a row into a columnar collection and update secondary indexes.
    pub async fn columnar_insert(
        &self,
        collection: &str,
        values: &[nodedb_types::value::Value],
    ) -> NodeDbResult<()> {
        let schema = self.columnar.schema(collection).ok_or_else(|| {
            NodeDbError::storage(format!("columnar collection '{collection}' not found"))
        })?;

        self.columnar
            .insert(collection, values)
            .map_err(NodeDbError::storage)?;

        let row_id = crate::engine::index_integration::row_id(&schema.columns, values);

        crate::engine::index_integration::index_row(
            collection,
            &row_id,
            &schema.columns,
            values,
            &self.spatial,
            &self.fts_state.manager,
        )?;
        crate::engine::index_integration::index_row_vectors(
            &self.vector_state,
            collection,
            &row_id,
            &schema.columns,
            values,
        )
        .await?;

        // Spatial profile: compute geohash for Point geometries and store
        // in the text index for prefix-based proximity queries.
        crate::engine::index_integration::index_geohash(
            collection,
            &row_id,
            &schema,
            self.columnar.profile(collection).as_ref(),
            values,
            &self.fts_state.manager,
        )?;
        Ok(())
    }

    /// Apply a CRDT field-level update to a strict collection row.
    ///
    /// Used during sync: a remote delta specifies field changes for a row.
    /// This reads the current tuple, patches the fields, and writes back.
    pub async fn strict_crdt_patch(
        &self,
        collection: &str,
        pk: &nodedb_types::value::Value,
        field_updates: &std::collections::HashMap<String, nodedb_types::value::Value>,
    ) -> NodeDbResult<()> {
        let schema = self.strict.schema(collection).ok_or_else(|| {
            NodeDbError::storage(format!("strict collection '{collection}' not found"))
        })?;

        // Read existing tuple.
        let existing = self
            .strict
            .get(collection, pk)
            .await
            .map_err(NodeDbError::storage)?
            .ok_or_else(|| NodeDbError::storage("row not found for CRDT patch"))?;

        // Re-encode as tuple bytes for the adapter.
        let encoder = nodedb_strict::TupleEncoder::new(&schema);
        let tuple_bytes = encoder
            .encode(&existing)
            .map_err(|e| NodeDbError::storage(e.to_string()))?;

        // Apply the CRDT patch.
        let patched = crate::engine::strict::crdt_adapter::apply_crdt_set(
            &tuple_bytes,
            &schema,
            field_updates,
        )
        .map_err(NodeDbError::storage)?;

        // Decode patched tuple back to values and update.
        let decoder = nodedb_strict::TupleDecoder::new(&schema);
        let new_values = decoder
            .extract_all(&patched)
            .map_err(|e| NodeDbError::storage(e.to_string()))?;

        // Write back via the standard update path.
        self.strict
            .update_by_values(collection, pk, &new_values)
            .await
            .map_err(NodeDbError::storage)?;

        // Re-index the patched row's text so no pre-patch term keeps matching.
        crate::engine::index_integration::index_row_text(
            collection,
            &crate::engine::index_integration::row_id(&schema.columns, &new_values),
            &schema.columns,
            &new_values,
            &self.fts_state.manager,
        )?;

        Ok(())
    }
}
