// SPDX-License-Identifier: Apache-2.0

//! Indexed CRUD for strict and columnar collections: combines base-engine
//! writes with secondary-index maintenance (vector, spatial, text, B-tree)
//! and HTAP materialized-view replication.

use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::nodedb::core::types::NodeDbLite;
use crate::nodedb::lock_ext::LockExt;
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
        let row_id = pk_to_string(&schema.columns, values);

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

        // Update secondary B-tree indexes on non-PK columns.
        {
            use crate::engine::strict::secondary_index::SecondaryIndex;
            let mut sec = self.secondary_indices.lock_or_recover();
            for (i, col) in schema.columns.iter().enumerate() {
                if col.primary_key || i >= values.len() {
                    continue;
                }
                let key = format!("{collection}:{}", col.name);
                sec.entry(key)
                    .or_insert_with(|| SecondaryIndex::new(&col.name))
                    .insert(&values[i], &row_id);
            }
        }

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
        let row_id = pk_value_to_string(pk);

        // Remove text and vector index entries before deleting the row.
        crate::engine::index_integration::deindex_row_text(
            collection,
            &row_id,
            &schema.columns,
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

        let row_id = pk_to_string(&schema.columns, values);

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
        if let Some(profile) = self.columnar.profile(collection)
            && let Some((_idx, geom)) = crate::engine::columnar::spatial_profile::extract_geometry(
                &schema, &profile, values,
            )
            && let Some(hash) = crate::engine::columnar::spatial_profile::compute_geohash(&geom)
        {
            self.fts_state
                .manager
                .lock_or_recover()
                .index_field(collection, "_geohash", &row_id, &hash)?;
        }
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

        Ok(())
    }
}

/// Build a string row ID from PK column values (for index keying).
fn pk_to_string(
    columns: &[nodedb_types::columnar::ColumnDef],
    values: &[nodedb_types::value::Value],
) -> String {
    let mut parts = Vec::new();
    for (i, col) in columns.iter().enumerate() {
        if col.primary_key
            && let Some(val) = values.get(i)
        {
            parts.push(pk_value_to_string(val));
        }
    }
    parts.join(":")
}

/// The row-id text of one primary-key value.
fn pk_value_to_string(val: &nodedb_types::value::Value) -> String {
    use nodedb_types::value::Value;
    match val {
        Value::Integer(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Uuid(s) => s.clone(),
        other => format!("{other:?}"),
    }
}
