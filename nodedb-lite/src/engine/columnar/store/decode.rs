// SPDX-License-Identifier: Apache-2.0

//! Decoded column values and primary-key reconstruction.

use nodedb_columnar::mutation::MutationEngine;
use nodedb_types::value::Value;

/// Extract a single `Value` from a `DecodedColumn` at the given row index.
///
/// Returns `Value::Null` for rows whose validity bit is false.
pub(super) fn decoded_column_value(
    col: &nodedb_columnar::reader::DecodedColumn,
    row_idx: usize,
) -> Value {
    use nodedb_columnar::reader::DecodedColumn;
    match col {
        DecodedColumn::Int64 { values, valid } => {
            if *valid.get(row_idx).unwrap_or(&false) {
                Value::Integer(*values.get(row_idx).unwrap_or(&0))
            } else {
                Value::Null
            }
        }
        DecodedColumn::Float64 { values, valid } => {
            if *valid.get(row_idx).unwrap_or(&false) {
                Value::Float(*values.get(row_idx).unwrap_or(&0.0))
            } else {
                Value::Null
            }
        }
        DecodedColumn::Timestamp { values, valid } => {
            if *valid.get(row_idx).unwrap_or(&false) {
                Value::Integer(*values.get(row_idx).unwrap_or(&0))
            } else {
                Value::Null
            }
        }
        DecodedColumn::Bool { values, valid } => {
            if *valid.get(row_idx).unwrap_or(&false) {
                Value::Bool(*values.get(row_idx).unwrap_or(&false))
            } else {
                Value::Null
            }
        }
        DecodedColumn::Binary {
            data,
            offsets,
            valid,
        } => {
            if *valid.get(row_idx).unwrap_or(&false) && row_idx + 1 < offsets.len() {
                let start = offsets[row_idx] as usize;
                let end = offsets[row_idx + 1] as usize;
                if let Ok(s) = std::str::from_utf8(&data[start..end]) {
                    Value::String(s.to_string())
                } else {
                    Value::Bytes(data[start..end].to_vec())
                }
            } else {
                Value::Null
            }
        }
        DecodedColumn::DictEncoded {
            ids,
            dictionary,
            valid,
        } => {
            if *valid.get(row_idx).unwrap_or(&false) {
                let id = *ids.get(row_idx).unwrap_or(&0) as usize;
                dictionary
                    .get(id)
                    .map(|s| Value::String(s.clone()))
                    .unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
    }
}

/// Rebuild PK index entries from a decoded PK column.
pub(super) fn rebuild_pk_from_column(
    mutation: &mut MutationEngine,
    pk_col: &nodedb_columnar::reader::DecodedColumn,
    segment_id: u32,
) {
    use nodedb_columnar::pk_index::{RowLocation, encode_pk};
    use nodedb_columnar::reader::DecodedColumn;

    match pk_col {
        DecodedColumn::Int64 { values, valid } => {
            for (row_idx, (val, &is_valid)) in values.iter().zip(valid.iter()).enumerate() {
                if is_valid {
                    let pk_bytes = encode_pk(&Value::Integer(*val));
                    mutation.pk_index_mut().upsert(
                        pk_bytes,
                        RowLocation {
                            segment_id: segment_id as u64,
                            row_index: row_idx as u32,
                        },
                    );
                }
            }
        }
        DecodedColumn::Binary {
            data,
            offsets,
            valid,
        } => {
            for (row_idx, &is_valid) in valid.iter().enumerate() {
                if is_valid {
                    let start = offsets[row_idx] as usize;
                    let end = offsets[row_idx + 1] as usize;
                    let pk_bytes = data[start..end].to_vec();
                    mutation.pk_index_mut().upsert(
                        pk_bytes,
                        RowLocation {
                            segment_id: segment_id as u64,
                            row_index: row_idx as u32,
                        },
                    );
                }
            }
        }
        _ => {}
    }
}
