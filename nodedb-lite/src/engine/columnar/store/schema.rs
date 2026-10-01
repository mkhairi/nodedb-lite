// SPDX-License-Identifier: Apache-2.0

//! Collection schema creation, alteration, and removal.

use super::lifecycle::{META_COLUMNAR_COLLECTIONS, META_COLUMNAR_SCHEMA_PREFIX};
use super::segments::remove_segment_bytes;
use super::state::{CollectionState, ColumnarEngine, SegmentMeta};
use crate::error::LiteError;
use crate::storage::engine::{StorageEngine, WriteOp};
use nodedb_columnar::mutation::MutationEngine;
use nodedb_types::Namespace;
use nodedb_types::columnar::{ColumnarProfile, ColumnarSchema};
use std::sync::{Arc, Mutex};

impl<S: StorageEngine> ColumnarEngine<S> {
    /// Create a new columnar collection.
    pub async fn create_collection(
        &self,
        name: &str,
        schema: ColumnarSchema,
        profile: ColumnarProfile,
        bitemporal: bool,
    ) -> Result<(), LiteError> {
        // Snapshot existing names + dup check under read lock.
        let mut names: Vec<String> = {
            let guard = self
                .collections
                .read()
                .map_err(|_| LiteError::LockPoisoned)?;
            if guard.contains_key(name) {
                return Err(LiteError::BadRequest {
                    detail: format!("columnar collection '{name}' already exists"),
                });
            }
            guard.keys().cloned().collect()
        };
        names.push(name.to_string());

        // Persist schema + collection list (lock dropped).
        #[derive(serde::Serialize, zerompk::ToMessagePack)]
        struct StoredSchema<'a> {
            schema: &'a ColumnarSchema,
            profile: &'a ColumnarProfile,
            bitemporal: bool,
        }
        let meta_key = format!("{META_COLUMNAR_SCHEMA_PREFIX}{name}");
        let schema_bytes = zerompk::to_msgpack_vec(&StoredSchema {
            schema: &schema,
            profile: &profile,
            bitemporal,
        })
        .map_err(|e| LiteError::Serialization {
            detail: e.to_string(),
        })?;

        let names_bytes =
            zerompk::to_msgpack_vec(&names).map_err(|e| LiteError::Serialization {
                detail: e.to_string(),
            })?;

        self.storage
            .batch_write(&[
                WriteOp::Put {
                    ns: Namespace::Meta,
                    key: meta_key.into_bytes(),
                    value: schema_bytes,
                },
                WriteOp::Put {
                    ns: Namespace::Meta,
                    key: META_COLUMNAR_COLLECTIONS.to_vec(),
                    value: names_bytes,
                },
            ])
            .await?;

        let mutation = MutationEngine::new(name.to_string(), schema);
        let state = CollectionState {
            mutation,
            profile,
            bitemporal,
            segments: Vec::new(),
            next_segment_id: 1,
        };

        let mut guard = self
            .collections
            .write()
            .map_err(|_| LiteError::LockPoisoned)?;
        if guard.contains_key(name) {
            return Err(LiteError::BadRequest {
                detail: format!(
                    "columnar collection '{name}' was created concurrently by another writer"
                ),
            });
        }
        guard.insert(name.to_string(), Arc::new(Mutex::new(state)));

        Ok(())
    }

    /// Drop a columnar collection and all its data.
    pub async fn drop_collection(&self, name: &str) -> Result<(), LiteError> {
        // Snapshot state and remove from map under write lock.
        let (segments, remaining_names): (Vec<SegmentMeta>, Vec<String>) = {
            let mut guard = self
                .collections
                .write()
                .map_err(|_| LiteError::LockPoisoned)?;
            let state_arc = guard
                .remove(name)
                .ok_or_else(|| LiteError::collection_not_found("columnar", name))?;
            let segments = {
                let s = state_arc.lock().map_err(|_| LiteError::LockPoisoned)?;
                s.segments.clone()
            };
            let names: Vec<String> = guard.keys().cloned().collect();
            (segments, names)
        };

        // Delete large segment bytes via the segment ext (or KV fallback).
        for seg in &segments {
            if seg.fully_deleted_at_ms.is_none() {
                remove_segment_bytes(&*self.storage, name, seg.segment_id).await?;
            }
        }

        // Remove small B+ tree entries (delete bitmaps, metadata, schema).
        let mut ops = Vec::new();
        for seg in &segments {
            ops.push(WriteOp::Delete {
                ns: Namespace::Columnar,
                key: format!("{name}:del:{}", seg.segment_id).into_bytes(),
            });
        }
        ops.push(WriteOp::Delete {
            ns: Namespace::Columnar,
            key: format!("{name}:meta").into_bytes(),
        });
        ops.push(WriteOp::Delete {
            ns: Namespace::Meta,
            key: format!("{META_COLUMNAR_SCHEMA_PREFIX}{name}").into_bytes(),
        });

        let names_bytes =
            zerompk::to_msgpack_vec(&remaining_names).map_err(|e| LiteError::Serialization {
                detail: e.to_string(),
            })?;
        ops.push(WriteOp::Put {
            ns: Namespace::Meta,
            key: META_COLUMNAR_COLLECTIONS.to_vec(),
            value: names_bytes,
        });

        self.storage.batch_write(&ops).await?;
        Ok(())
    }

    /// Add a column to an existing columnar collection.
    pub async fn alter_add_column(
        &self,
        name: &str,
        column: nodedb_types::columnar::ColumnDef,
    ) -> Result<(), LiteError> {
        if !column.nullable && column.default.is_none() {
            return Err(LiteError::BadRequest {
                detail: format!(
                    "ALTER ADD COLUMN '{}': non-nullable column must have a DEFAULT",
                    column.name
                ),
            });
        }

        let state_arc = self.lookup(name)?;

        // Snapshot current schema + profile under inner lock.
        let (mut schema, profile) = {
            let s = Self::lock_state(&state_arc)?;
            if s.mutation
                .schema()
                .columns
                .iter()
                .any(|c| c.name == column.name)
            {
                return Err(LiteError::BadRequest {
                    detail: format!("column '{}' already exists in '{name}'", column.name),
                });
            }
            (s.mutation.schema().clone(), s.profile.clone())
        };

        schema.columns.push(column);
        schema.version = schema.version.saturating_add(1);

        // Persist updated schema (lock dropped).
        #[derive(serde::Serialize, zerompk::ToMessagePack)]
        struct StoredSchema<'a> {
            schema: &'a ColumnarSchema,
            profile: &'a ColumnarProfile,
        }
        let meta_key = format!("{META_COLUMNAR_SCHEMA_PREFIX}{name}");
        let schema_bytes = zerompk::to_msgpack_vec(&StoredSchema {
            schema: &schema,
            profile: &profile,
        })
        .map_err(|e| LiteError::Serialization {
            detail: e.to_string(),
        })?;

        self.storage
            .put(Namespace::Meta, meta_key.as_bytes(), &schema_bytes)
            .await?;

        // Swap in the new MutationEngine.
        let mut s = Self::lock_state(&state_arc)?;
        s.mutation = MutationEngine::new(name.to_string(), schema);

        Ok(())
    }

    /// Get the schema for a collection (returns a clone).
    pub fn schema(&self, name: &str) -> Option<ColumnarSchema> {
        let guard = self.collections.read().ok()?;
        let state_arc = guard.get(name)?;
        let s = state_arc.lock().ok()?;
        Some(s.mutation.schema().clone())
    }

    /// Get the profile for a collection (returns a clone).
    pub fn profile(&self, name: &str) -> Option<ColumnarProfile> {
        let guard = self.collections.read().ok()?;
        let state_arc = guard.get(name)?;
        let s = state_arc.lock().ok()?;
        Some(s.profile.clone())
    }

    /// List all columnar collection names.
    pub fn collection_names(&self) -> Vec<String> {
        self.collections
            .read()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }
}
