//! CRUD operations for strict document collections.

use std::collections::HashMap;

use nodedb_strict::arrow_extract::extract_column_to_arrow;
use nodedb_types::Namespace;
use nodedb_types::columnar::SchemaOps;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::runtime::now_millis_i64;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::engine::{CollectionState, StrictEngine, strict_err_to_lite};
use super::history::{history_key, history_value};

impl<S: StorageEngine> StrictEngine<S> {
    // -- Write path --

    /// Insert a row into a strict collection.
    ///
    /// Validates schema, encodes as Binary Tuple, writes to storage keyed by PK.
    /// Returns an error if the PK already exists.
    ///
    /// For bitemporal collections, also writes an initial history entry so that
    /// the row's birth time is recorded in `Namespace::StrictHistory`.
    pub async fn insert(&self, collection: &str, values: &[Value]) -> Result<(), LiteError> {
        self.insert_rows(collection, std::slice::from_ref(&values.to_vec()))
            .await
    }

    /// Insert rows as [`Self::insert`] does, all or none: a primary key that
    /// exists or repeats within `rows`, or a unique index the rows would
    /// break, refuses every row.
    pub async fn insert_rows(
        &self,
        collection: &str,
        rows: &[Vec<Value>],
    ) -> Result<(), LiteError> {
        let state = self.get_state(collection)?;
        let mut ops = Vec::with_capacity(rows.len());
        let mut keys: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for values in rows {
            let tuple = state.encoder.encode(values).map_err(strict_err_to_lite)?;
            let key = state.storage_key(collection, values);
            if !keys.insert(key.clone())
                || self.storage.get(Namespace::Strict, &key).await?.is_some()
            {
                return Err(LiteError::BadRequest {
                    detail: format!("duplicate primary key in collection '{collection}'"),
                });
            }
            ops.push(WriteOp::Put {
                ns: Namespace::Strict,
                key,
                value: tuple,
            });
        }
        let history_ops = state.schema.bitemporal.then(|| ops.clone());
        self.commit(collection, ops).await?;

        // For bitemporal collections, write each row's birth history entry.
        // The current row's system_from_ms is stored at slot 0 of the tuple;
        // we read it back from `values[0]` (the `__system_from_ms` column).
        if let Some(history_ops) = history_ops {
            for (values, op) in rows.iter().zip(&history_ops) {
                let WriteOp::Put { key, value, .. } = op else {
                    continue;
                };
                let system_from_ms = extract_system_from_values(values);
                // u64::MAX encodes "no system_to yet" (row is still current).
                let hist_key =
                    history_key(collection, system_from_ms, &key[collection.len() + 1..]);
                let hist_value = history_value(value, i64::MAX);
                self.storage
                    .put(Namespace::StrictHistory, &hist_key, &hist_value)
                    .await?;
            }
        }
        Ok(())
    }

    /// Insert multiple rows atomically.
    pub async fn insert_batch(
        &self,
        collection: &str,
        rows: &[Vec<Value>],
    ) -> Result<(), LiteError> {
        let state = self.get_state(collection)?;

        let mut ops = Vec::with_capacity(rows.len());
        for values in rows {
            let tuple = state.encoder.encode(values).map_err(strict_err_to_lite)?;
            let key = state.storage_key(collection, values);
            ops.push(WriteOp::Put {
                ns: Namespace::Strict,
                key,
                value: tuple,
            });
        }

        self.commit(collection, ops).await
    }

    /// Update a row by PK. Reads the existing tuple, patches the specified
    /// fields, and writes the modified tuple back.
    ///
    /// `updates` maps column names to new values. Columns not in the map
    /// retain their existing values.
    pub async fn update(
        &self,
        collection: &str,
        pk: &Value,
        updates: &HashMap<String, Value>,
    ) -> Result<bool, LiteError> {
        let changed = self
            .update_many(collection, &[(pk.clone(), updates.clone())])
            .await?;
        Ok(changed == 1)
    }

    /// Update rows by PK, each `(pk, updates)` as [`Self::update`] does, in
    /// one storage batch: a unique index the statement would break refuses
    /// every row. Rows that do not exist are skipped. Returns the number of
    /// rows updated.
    pub async fn update_many(
        &self,
        collection: &str,
        changes: &[(Value, HashMap<String, Value>)],
    ) -> Result<u64, LiteError> {
        let state = self.get_state(collection)?;
        let mut ops = Vec::new();
        // (key suffix, old tuple) of each superseded version, and
        // (final key, new tuple) of each new one, for bitemporal history.
        let mut superseded: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut born: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut updated = 0u64;
        for (pk, updates) in changes {
            let key = state.storage_key_from_pk(collection, pk);
            let Some(existing) = self.storage.get(Namespace::Strict, &key).await? else {
                continue;
            };
            let mut values = decode_tuple(&state, &existing)?;
            for (col_name, new_value) in updates {
                let col_idx =
                    state
                        .schema
                        .column_index(col_name)
                        .ok_or_else(|| LiteError::BadRequest {
                            detail: format!(
                                "unknown column '{col_name}' in collection '{collection}'"
                            ),
                        })?;
                // Validate the new value against the column type.
                if !matches!(new_value, Value::Null)
                    && !state.schema.columns[col_idx].column_type.accepts(new_value)
                {
                    return Err(LiteError::BadRequest {
                        detail: format!(
                            "column '{}': type mismatch",
                            state.schema.columns[col_idx].name
                        ),
                    });
                }
                values[col_idx] = new_value.clone();
            }
            let new_tuple = state.encoder.encode(&values).map_err(strict_err_to_lite)?;
            // An update that changes the PK moves the row to its new key.
            let new_key = state.storage_key(collection, &values);
            if state.schema.bitemporal {
                superseded.push((key[collection.len() + 1..].to_vec(), existing));
                born.push((new_key.clone(), new_tuple.clone()));
            }
            if new_key != key {
                ops.push(WriteOp::Delete {
                    ns: Namespace::Strict,
                    key,
                });
            }
            ops.push(WriteOp::Put {
                ns: Namespace::Strict,
                key: new_key,
                value: new_tuple,
            });
            updated += 1;
        }

        // For bitemporal collections, record each old version's supersession
        // before overwriting. The system_to of the old version is now().
        for (suffix, old_tuple) in &superseded {
            let system_to_ms = now_millis_i64();
            self.record_history_supersession(collection, suffix, old_tuple, system_to_ms)
                .await?;
        }

        self.commit(collection, ops).await?;

        // For bitemporal collections, write each new version's birth entry.
        for (final_key, new_tuple) in &born {
            let new_system_from_ms = now_millis_i64();
            let hist_key = history_key(
                collection,
                new_system_from_ms,
                &final_key[collection.len() + 1..],
            );
            let hist_value = history_value(new_tuple, i64::MAX);
            self.storage
                .put(Namespace::StrictHistory, &hist_key, &hist_value)
                .await?;
        }

        Ok(updated)
    }

    /// Update a row by replacing with complete new values (for CRDT adapter).
    pub async fn update_by_values(
        &self,
        collection: &str,
        pk: &Value,
        new_values: &[Value],
    ) -> Result<bool, LiteError> {
        let state = self.get_state(collection)?;
        let key = state.storage_key_from_pk(collection, pk);

        if self.storage.get(Namespace::Strict, &key).await?.is_none() {
            return Ok(false);
        }

        let new_tuple = state
            .encoder
            .encode(new_values)
            .map_err(strict_err_to_lite)?;
        self.commit(
            collection,
            vec![WriteOp::Put {
                ns: Namespace::Strict,
                key,
                value: new_tuple,
            }],
        )
        .await?;
        Ok(true)
    }

    /// Delete a row by PK. Returns true if the row existed.
    ///
    /// For bitemporal collections, the old row's history entry is finalized
    /// with `system_to_ms = now()` before the current row is removed.
    /// History rows are retained for audit until an explicit `TemporalPurge`.
    pub async fn delete(&self, collection: &str, pk: &Value) -> Result<bool, LiteError> {
        let state = self.get_state(collection)?;
        let key = state.storage_key_from_pk(collection, pk);

        let existing = self.storage.get(Namespace::Strict, &key).await?;
        match existing {
            None => return Ok(false),
            Some(old_tuple) => {
                if state.schema.bitemporal {
                    let system_to_ms = now_millis_i64();
                    self.record_history_supersession(
                        collection,
                        &key[collection.len() + 1..],
                        &old_tuple,
                        system_to_ms,
                    )
                    .await?;
                }
                self.commit(
                    collection,
                    vec![WriteOp::Delete {
                        ns: Namespace::Strict,
                        key,
                    }],
                )
                .await?;
            }
        }
        Ok(true)
    }

    // -- Read path --

    /// Point lookup by PK. Returns the row as a Vec<Value>, or None.
    pub async fn get(&self, collection: &str, pk: &Value) -> Result<Option<Vec<Value>>, LiteError> {
        let state = self.get_state(collection)?;
        let key = state.storage_key_from_pk(collection, pk);

        match self.storage.get(Namespace::Strict, &key).await? {
            Some(bytes) => decode_tuple(&state, &bytes).map(Some),
            None => Ok(None),
        }
    }

    /// Point lookup with column projection. Only decodes the requested columns.
    pub async fn get_projected(
        &self,
        collection: &str,
        pk: &Value,
        columns: &[&str],
    ) -> Result<Option<Vec<Value>>, LiteError> {
        let state = self.get_state(collection)?;
        let key = state.storage_key_from_pk(collection, pk);

        match self.storage.get(Namespace::Strict, &key).await? {
            Some(bytes) => {
                let mut values = Vec::with_capacity(columns.len());
                for col_name in columns {
                    let val = state
                        .decoder
                        .extract_by_name(&bytes, col_name)
                        .map_err(strict_err_to_lite)?;
                    values.push(val);
                }
                Ok(Some(values))
            }
            None => Ok(None),
        }
    }

    /// Scan all rows in a collection. Returns raw tuple bytes for Arrow extraction.
    pub async fn scan_raw(&self, collection: &str) -> Result<Vec<Vec<u8>>, LiteError> {
        let _state = self.get_state(collection)?;
        let prefix = format!("{collection}:");
        let entries = self
            .storage
            .scan_prefix(Namespace::Strict, prefix.as_bytes())
            .await?;
        Ok(entries.into_iter().map(|(_, v)| v).collect())
    }

    /// Scan all rows and extract a single column into an Arrow array.
    pub async fn scan_column_to_arrow(
        &self,
        collection: &str,
        col_idx: usize,
    ) -> Result<arrow::array::ArrayRef, LiteError> {
        let state = self.get_state(collection)?;
        let tuples = self.scan_raw(collection).await?;
        let refs: Vec<&[u8]> = tuples.iter().map(|t| t.as_slice()).collect();

        extract_column_to_arrow(&state.schema, &state.decoder, &refs, col_idx)
            .map_err(strict_err_to_lite)
    }

    /// Scan all rows and extract multiple columns into Arrow arrays.
    pub async fn scan_columns_to_arrow(
        &self,
        collection: &str,
        col_indices: &[usize],
    ) -> Result<Vec<arrow::array::ArrayRef>, LiteError> {
        let state = self.get_state(collection)?;
        let tuples = self.scan_raw(collection).await?;
        let refs: Vec<&[u8]> = tuples.iter().map(|t| t.as_slice()).collect();

        let mut arrays = Vec::with_capacity(col_indices.len());
        for &idx in col_indices {
            let arr = extract_column_to_arrow(&state.schema, &state.decoder, &refs, idx)
                .map_err(strict_err_to_lite)?;
            arrays.push(arr);
        }
        Ok(arrays)
    }

    /// Scan all rows in a collection and decode each to `Vec<Value>`.
    ///
    /// Returns rows in storage key order. Column order matches the schema
    /// definition order, consistent with `schema().columns`.
    pub async fn list_rows(&self, collection: &str) -> Result<Vec<Vec<Value>>, LiteError> {
        let state = self.get_state(collection)?;
        let raw_tuples = self.scan_raw(collection).await?;
        let mut rows = Vec::with_capacity(raw_tuples.len());
        for bytes in &raw_tuples {
            let values = state
                .decoder
                .extract_all(bytes)
                .map_err(strict_err_to_lite)?;
            rows.push(values);
        }
        Ok(rows)
    }

    /// Count the number of rows in a collection.
    pub async fn count(&self, collection: &str) -> Result<usize, LiteError> {
        let _state = self.get_state(collection)?;
        let prefix = format!("{collection}:");
        let entries = self
            .storage
            .scan_prefix(Namespace::Strict, prefix.as_bytes())
            .await?;
        Ok(entries.len())
    }
}

/// Extract the `__system_from_ms` value from the leading values of a bitemporal row.
///
/// In a bitemporal strict schema, `__system_from_ms` is always at user-visible
/// index 0 of the `values` slice passed to `insert` / `update`. Returns 0 if the
/// value is not an integer (should not happen for correctly-constructed rows).
fn extract_system_from_values(values: &[Value]) -> i64 {
    match values.first() {
        Some(Value::Integer(ms)) => *ms,
        _ => 0,
    }
}

/// Decode a stored tuple to values in current schema order. A tuple written
/// under an older schema version is read with that version's columns and
/// padded with NULL for the columns added since.
pub(super) fn decode_tuple(state: &CollectionState, bytes: &[u8]) -> Result<Vec<Value>, LiteError> {
    let tuple_version = state
        .decoder
        .schema_version(bytes)
        .map_err(strict_err_to_lite)?;
    if tuple_version >= state.schema.version {
        return state.decoder.extract_all(bytes).map_err(strict_err_to_lite);
    }
    let old_col_count = state
        .version_column_counts
        .get(&(tuple_version as u16))
        .copied()
        .unwrap_or(state.schema.columns.len());
    let old_schema = nodedb_types::columnar::StrictSchema {
        columns: state.schema.columns[..old_col_count].to_vec(),
        version: tuple_version,
        dropped_columns: Vec::new(),
        bitemporal: state.schema.bitemporal,
    };
    let old_decoder = nodedb_strict::TupleDecoder::new(&old_schema);
    let mut values = old_decoder.extract_all(bytes).map_err(strict_err_to_lite)?;
    values.resize(state.schema.columns.len(), Value::Null);
    Ok(values)
}
