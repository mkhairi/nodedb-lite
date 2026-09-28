//! Index integration for strict and columnar collections.
//!
//! Provides helper functions to maintain secondary indexes (R-tree, HNSW,
//! text/BM25) when rows are inserted or deleted from strict or columnar
//! collections. These indexes enable spatial queries, vector search, and
//! full-text search over typed collections.
//!
//! Uses the existing per-collection index infrastructure in NodeDbLite
//! (HNSW indices, spatial manager, text indices) — the same indexes used
//! by schemaless document collections.
//!
//! Callers: `NodeDbLite` calls `index_row` and `index_row_vectors` after
//! `StrictEngine.insert()` or `ColumnarEngine.insert()` to maintain secondary
//! indexes, and `deindex_row_text` and `deindex_row_vectors` before
//! `delete()` to remove a row's entries. Vector columns reach their HNSW
//! index through `engine::vector::resident`, which loads an evicted index
//! back first.

use std::sync::{Arc, Mutex};

use nodedb_types::columnar::ColumnType;
use nodedb_types::geometry::Geometry;
use nodedb_types::value::Value;

use crate::engine::fts::FtsCollectionManager;
use crate::engine::spatial::SpatialIndexManager;
use crate::engine::vector::VectorState;
use crate::engine::vector::nodes::{encode_sidecar, unbind_node, upsert_node};
use crate::engine::vector::resident::lock_resident;
use crate::error::LiteError;
use crate::nodedb::lock_ext::LockExt;
use crate::storage::engine::StorageEngine;

/// Index a row from a strict or columnar collection into its spatial and
/// text indexes: GEOMETRY → R-tree, STRING → text/BM25. VECTOR columns go
/// through [`index_row_vectors`].
///
/// `collection` is the collection name (used as the index key).
/// `row_id` is a string identifier for the row (typically the PK value).
/// `columns` are the column definitions from the schema.
/// `values` are the row's values in schema order.
pub fn index_row(
    collection: &str,
    row_id: &str,
    columns: &[nodedb_types::columnar::ColumnDef],
    values: &[Value],
    spatial: &Mutex<SpatialIndexManager>,
    fts: &Mutex<FtsCollectionManager>,
) -> Result<(), LiteError> {
    for (i, col) in columns.iter().enumerate() {
        if i >= values.len() {
            break;
        }
        let val = &values[i];

        match &col.column_type {
            ColumnType::Geometry => {
                index_geometry(collection, &col.name, row_id, val, spatial);
            }
            ColumnType::String => {
                index_text(collection, &col.name, row_id, val, fts)?;
            }
            _ => {} // No secondary index for other types.
        }
    }
    Ok(())
}

/// Remove a row's text entries from inverted indexes.
///
/// Only handles String columns (BM25 inverted index). R-tree spatial removal
/// requires the original geometry (uses `SpatialIndexManager.remove_document`
/// directly). HNSW uses soft-delete (tombstone) which doesn't need the original
/// vector — call `HnswIndex.delete(node_id)` directly when the node ID is known.
pub fn deindex_row_text(
    collection: &str,
    row_id: &str,
    columns: &[nodedb_types::columnar::ColumnDef],
    fts: &Mutex<FtsCollectionManager>,
) -> Result<(), LiteError> {
    for col in columns {
        if matches!(col.column_type, ColumnType::String) {
            remove_text(collection, &col.name, row_id, fts)?;
        }
    }
    Ok(())
}

/// Index a geometry value into the spatial R-tree.
fn index_geometry(
    collection: &str,
    field: &str,
    doc_id: &str,
    value: &Value,
    spatial: &Mutex<SpatialIndexManager>,
) {
    let geom = match value {
        Value::Geometry(g) => g.clone(),
        Value::String(s) => {
            // Try parsing as GeoJSON.
            match sonic_rs::from_str::<Geometry>(s) {
                Ok(g) => g,
                Err(_) => return,
            }
        }
        _ => return,
    };

    let mut spatial = spatial.lock_or_recover();
    spatial.index_document(collection, field, doc_id, &geom);
}

/// Index a row's VECTOR columns into their HNSW indexes, keyed
/// `"{collection}:{column}"`, and bind each node to `row_id`, replacing the
/// node `row_id` held before, and encode it into the codec sidecar when the
/// index config calls for one. The vector is made durable first, as every
/// other vector insert does. A NULL value indexes nothing.
///
/// Fails with `DataException` for a value that is not a vector of the
/// column's width, and propagates a storage or index error.
pub(crate) async fn index_row_vectors<S: StorageEngine>(
    vector_state: &Arc<VectorState<S>>,
    collection: &str,
    row_id: &str,
    columns: &[nodedb_types::columnar::ColumnDef],
    values: &[Value],
) -> Result<(), LiteError> {
    for (col, value) in columns.iter().zip(values) {
        let ColumnType::Vector(dim) = &col.column_type else {
            continue;
        };
        let Some(vector) = vector_value(collection, &col.name, value, *dim as usize)? else {
            continue;
        };
        let index_key = format!("{collection}:{}", col.name);
        let op = crate::engine::vector::durable::put_op(&index_key, row_id, &vector);
        vector_state
            .storage
            .batch_write(std::slice::from_ref(&op))
            .await?;
        let node = upsert_node(vector_state, &index_key, row_id, &vector).await?;
        encode_sidecar(vector_state, &index_key, node, &vector)?;
    }
    Ok(())
}

/// Remove a row's VECTOR column entries: the durable vector, and the live
/// node bound to `row_id`, loading an evicted index back first. A tombstone
/// an evicted index missed would bring the vector back.
pub(crate) async fn deindex_row_vectors<S: StorageEngine>(
    vector_state: &Arc<VectorState<S>>,
    collection: &str,
    row_id: &str,
    columns: &[nodedb_types::columnar::ColumnDef],
) -> Result<(), LiteError> {
    for col in columns {
        if !matches!(col.column_type, ColumnType::Vector(_)) {
            continue;
        }
        let index_key = format!("{collection}:{}", col.name);
        crate::engine::vector::durable::remove(&*vector_state.storage, &index_key, row_id).await?;
        let mut indices = lock_resident(vector_state, &index_key).await?;
        unbind_node(
            vector_state,
            indices.get_mut(&index_key),
            &index_key,
            row_id,
        );
    }
    Ok(())
}

/// The vector a VECTOR column value holds: an array of numbers or packed
/// little-endian f32 bytes, exactly `dim` wide. `None` for NULL.
fn vector_value(
    collection: &str,
    column: &str,
    value: &Value,
    dim: usize,
) -> Result<Option<Vec<f32>>, LiteError> {
    let vector: Vec<f32> = match value {
        Value::Null => return Ok(None),
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::Float(f) => Ok(*f as f32),
                Value::Integer(n) => Ok(*n as f32),
                other => Err(LiteError::DataException {
                    detail: format!(
                        "vector column '{column}' of '{collection}' holds a non-numeric \
                         component: {other:?}"
                    ),
                }),
            })
            .collect::<Result<_, _>>()?,
        Value::Bytes(b) => {
            let (chunks, rest) = b.as_chunks::<4>();
            if !rest.is_empty() {
                return Err(LiteError::DataException {
                    detail: format!(
                        "vector column '{column}' of '{collection}' holds {} bytes, \
                         not a multiple of 4",
                        b.len()
                    ),
                });
            }
            chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
        }
        other => {
            return Err(LiteError::DataException {
                detail: format!(
                    "vector column '{column}' of '{collection}' holds a non-vector value: \
                     {other:?}"
                ),
            });
        }
    };
    nodedb_vector::error::check_dim(dim, vector.len())?;
    Ok(Some(vector))
}

/// Index a string value into the inverted text index (BM25).
fn index_text(
    collection: &str,
    field: &str,
    doc_id: &str,
    value: &Value,
    fts: &Mutex<FtsCollectionManager>,
) -> Result<(), LiteError> {
    let text = match value {
        Value::String(s) => s.as_str(),
        _ => return Ok(()),
    };
    fts.lock_or_recover()
        .index_field(collection, field, doc_id, text)
}

/// Remove a document from the text index.
fn remove_text(
    collection: &str,
    field: &str,
    doc_id: &str,
    fts: &Mutex<FtsCollectionManager>,
) -> Result<(), LiteError> {
    fts.lock_or_recover()
        .remove_field(collection, field, doc_id)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nodedb_mem::{EngineId, EngineLimits, GovernorConfig, MemoryGovernor};
    use nodedb_types::columnar::ColumnDef;
    use nodedb_types::{DatabaseId, TenantId};

    use super::*;
    use crate::engine::fts::FtsCollectionManager;

    /// Build a real, uncapped governor for these index-integration tests.
    fn test_governor() -> Arc<MemoryGovernor> {
        let per_engine = usize::MAX / EngineId::ALL.len();
        Arc::new(
            MemoryGovernor::new(GovernorConfig {
                global_ceiling: per_engine * EngineId::ALL.len(),
                engine_limits: EngineLimits::uniform(per_engine),
            })
            .expect("test governor"),
        )
    }

    /// Build a `ScopedMemory` bound to the spatial engine for these tests.
    fn test_spatial_memory() -> nodedb_mem::ScopedMemory {
        nodedb_mem::ScopedMemory::new(
            test_governor(),
            DatabaseId::DEFAULT,
            TenantId::new(0),
            EngineId::Spatial,
        )
    }

    #[test]
    fn index_row_routes_geometry() {
        let columns = vec![
            ColumnDef::required("id", ColumnType::Int64),
            ColumnDef::nullable("geom", ColumnType::Geometry),
        ];
        let values = vec![
            Value::Integer(1),
            Value::Geometry(Geometry::Point {
                coordinates: [10.0, 20.0],
            }),
        ];

        let spatial = Mutex::new(SpatialIndexManager::new(test_spatial_memory()));
        let text = Mutex::new(FtsCollectionManager::new(test_governor()));

        index_row("test", "1", &columns, &values, &spatial, &text)
            .expect("index update must succeed");

        let spatial = spatial.lock().expect("lock");
        assert!(!spatial.is_empty());
    }

    async fn vector_state() -> Arc<VectorState<crate::storage::pagedb_storage::PagedbStorageMem>> {
        let storage = Arc::new(
            crate::storage::pagedb_storage::PagedbStorageMem::open_in_memory()
                .await
                .expect("in-memory pagedb"),
        );
        let memory = crate::query::engine::test_scoped_memory(
            &crate::query::engine::test_governor(),
            EngineId::Vector,
        );
        Arc::new(VectorState::new(storage, 64, memory))
    }

    fn vector_columns() -> Vec<ColumnDef> {
        vec![
            ColumnDef::required("id", ColumnType::Int64),
            ColumnDef::nullable("emb", ColumnType::Vector(3)),
        ]
    }

    #[tokio::test]
    async fn index_row_vectors_indexes_binds_and_deindexes() {
        let state = vector_state().await;
        let values = vec![
            Value::Integer(1),
            Value::Array(vec![
                Value::Float(1.0),
                Value::Float(2.0),
                Value::Float(3.0),
            ]),
        ];

        index_row_vectors(&state, "test", "1", &vector_columns(), &values)
            .await
            .expect("index update must succeed");
        {
            let hnsw = state.hnsw_indices.lock().expect("lock");
            assert_eq!(hnsw.get("test:emb").map(|i| i.live_count()), Some(1));
        }
        assert_eq!(
            state
                .vector_id_map
                .lock()
                .expect("lock")
                .doc_id("test:emb", 0),
            Some("1")
        );

        deindex_row_vectors(&state, "test", "1", &vector_columns())
            .await
            .expect("deindex must succeed");
        let hnsw = state.hnsw_indices.lock().expect("lock");
        assert_eq!(hnsw.get("test:emb").map(|i| i.live_count()), Some(0));
    }

    #[tokio::test]
    async fn index_row_vectors_refuses_a_value_of_the_wrong_width() {
        let state = vector_state().await;
        let values = vec![
            Value::Integer(1),
            Value::Array(vec![Value::Float(1.0), Value::Float(2.0)]),
        ];
        let err = index_row_vectors(&state, "test", "1", &vector_columns(), &values)
            .await
            .expect_err("a 2-wide value in a 3-wide column must fail");
        assert!(matches!(err, LiteError::DataException { .. }), "{err:?}");
        assert!(state.hnsw_indices.lock().expect("lock").is_empty());
    }

    #[test]
    fn index_row_routes_text() {
        let columns = vec![
            ColumnDef::required("id", ColumnType::Int64),
            ColumnDef::required("body", ColumnType::String),
        ];
        let values = vec![
            Value::Integer(1),
            Value::String("hello world search test".into()),
        ];

        let spatial = Mutex::new(SpatialIndexManager::new(test_spatial_memory()));
        let text = Mutex::new(FtsCollectionManager::new(test_governor()));

        index_row("test", "1", &columns, &values, &spatial, &text)
            .expect("index update must succeed");

        let text = text.lock().expect("lock");
        assert!(
            !text.is_empty(),
            "text FTS index should have entries after indexing a String column"
        );
    }

    #[test]
    fn index_row_skips_non_indexable() {
        let columns = vec![
            ColumnDef::required("id", ColumnType::Int64),
            ColumnDef::required("count", ColumnType::Int64),
        ];
        let values = vec![Value::Integer(1), Value::Integer(42)];

        let spatial = Mutex::new(SpatialIndexManager::new(test_spatial_memory()));
        let text = Mutex::new(FtsCollectionManager::new(test_governor()));

        index_row("test", "1", &columns, &values, &spatial, &text)
            .expect("index update must succeed");

        // No indexes should be populated for Int64 columns.
        assert!(spatial.lock().expect("lock").is_empty());
        assert!(
            text.lock().expect("lock").is_empty(),
            "no text index for Int64 columns"
        );
    }
}
