// SPDX-License-Identifier: Apache-2.0

//! Columnar row mutations and outbound replication.

use super::segments::columnar_err_to_lite;
use super::state::ColumnarEngine;
use crate::error::LiteError;
use crate::storage::engine::StorageEngine;
use nodedb_types::columnar::ColumnarProfile;
use nodedb_types::value::Value;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

impl<S: StorageEngine> ColumnarEngine<S> {
    /// Insert a row into a columnar collection's memtable.
    ///
    /// This is a pure in-memory operation. Call [`enqueue_outbound`] from
    /// an async context after inserting to durably enqueue the row for
    /// replication to Origin.
    pub fn insert(&self, collection: &str, values: &[Value]) -> Result<(), LiteError> {
        let state_arc = self.lookup(collection)?;
        let mut s = Self::lock_state(&state_arc)?;
        s.mutation.insert(values).map_err(columnar_err_to_lite)?;
        Ok(())
    }

    /// Durably enqueue a batch of inserted rows for replication to Origin.
    ///
    /// Must be called from an async context after one or more successful
    /// [`insert`] calls. Timeseries-profile collections are routed to the
    /// `timeseries_outbound` queue; all other columnar collections use the
    /// plain `outbound` queue.
    ///
    /// Returns [`LiteError::Backpressure`] when the queue is at cap so the
    /// caller can propagate back-pressure. Other enqueue errors are logged as
    /// warnings (the local insert already succeeded) and `Ok(())` is returned.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn enqueue_outbound(
        &self,
        collection: &str,
        rows: &[Vec<Value>],
    ) -> Result<(), LiteError> {
        if rows.is_empty() {
            return Ok(());
        }

        // Read the profile and schema metadata under the lock, then drop it
        // before any await point to satisfy the no-lock-across-await rule.
        enum OutboundRoute {
            Timeseries { column_names: Vec<String> },
            Columnar { schema_bytes: Vec<u8> },
            None,
        }

        let route: OutboundRoute = {
            let state_arc = self.lookup(collection)?;
            let s = Self::lock_state(&state_arc)?;
            if matches!(s.profile, ColumnarProfile::Timeseries { .. }) {
                if self.timeseries_outbound.is_some() {
                    let column_names: Vec<String> = s
                        .mutation
                        .schema()
                        .columns
                        .iter()
                        .map(|c| c.name.clone())
                        .collect();
                    OutboundRoute::Timeseries { column_names }
                } else {
                    OutboundRoute::None
                }
            } else if self.outbound.is_some() {
                let schema_bytes = zerompk::to_msgpack_vec(s.mutation.schema()).unwrap_or_default();
                OutboundRoute::Columnar { schema_bytes }
            } else {
                OutboundRoute::None
            }
            // lock `s` is dropped here at end of block
        };

        match route {
            OutboundRoute::Timeseries { column_names } => {
                let queue = match &self.timeseries_outbound {
                    Some(q) => Arc::clone(q),
                    None => return Ok(()),
                };
                for row in rows {
                    crate::sync::reconcile_outbound_enqueue(
                        queue
                            .enqueue_row(collection, column_names.clone(), row.clone())
                            .await,
                        "timeseries insert",
                        collection,
                        "",
                    )?;
                }
            }
            OutboundRoute::Columnar { schema_bytes } => {
                let queue = match &self.outbound {
                    Some(q) => Arc::clone(q),
                    None => return Ok(()),
                };
                for row in rows {
                    crate::sync::reconcile_outbound_enqueue(
                        queue
                            .enqueue_row(collection, row.clone(), schema_bytes.clone())
                            .await,
                        "columnar insert",
                        collection,
                        "",
                    )?;
                }
            }
            OutboundRoute::None => {}
        }

        Ok(())
    }

    /// Delete a row by PK.
    pub fn delete(&self, collection: &str, pk: &Value) -> Result<bool, LiteError> {
        let state_arc = self.lookup(collection)?;
        let mut s = Self::lock_state(&state_arc)?;

        if matches!(s.profile, ColumnarProfile::Timeseries { .. }) {
            return Err(LiteError::BadRequest {
                detail: format!(
                    "DELETE not allowed on timeseries collection '{collection}' (append-only)"
                ),
            });
        }

        match s.mutation.delete(pk) {
            Ok(_) => Ok(true),
            Err(nodedb_columnar::ColumnarError::PrimaryKeyNotFound) => Ok(false),
            Err(e) => Err(columnar_err_to_lite(e)),
        }
    }

    /// Update a row: DELETE old + INSERT new.
    pub fn update(
        &self,
        collection: &str,
        old_pk: &Value,
        new_values: &[Value],
    ) -> Result<bool, LiteError> {
        let state_arc = self.lookup(collection)?;
        let mut s = Self::lock_state(&state_arc)?;

        if matches!(s.profile, ColumnarProfile::Timeseries { .. }) {
            return Err(LiteError::BadRequest {
                detail: format!(
                    "UPDATE not allowed on timeseries collection '{collection}' (append-only)"
                ),
            });
        }

        // Lite keeps no per-row surrogate sidecar for flushed segments.
        match s.mutation.update(old_pk, new_values, None) {
            Ok(_) => Ok(true),
            Err(nodedb_columnar::ColumnarError::PrimaryKeyNotFound) => Ok(false),
            Err(e) => Err(columnar_err_to_lite(e)),
        }
    }
}
