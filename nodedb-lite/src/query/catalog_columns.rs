// SPDX-License-Identifier: Apache-2.0

//! Column metadata the SQL catalog builds from engine schemas, persisted
//! collection configs and descriptor field hints.

use nodedb_sql::types::*;
use nodedb_types::columnar::{ColumnarSchema, FloatWidth, IntWidth, StrictSchema};

/// Build column metadata + primary key from a slice of `ColumnDef`.
///
/// Shared by `StrictSchema`/`KvConfig::schema` and `ColumnarSchema`, which
/// both carry `Vec<ColumnDef>` with identical (name, column_type, nullable,
/// default, primary_key) shape.
fn columns_from_column_defs(
    columns: &[nodedb_types::columnar::ColumnDef],
) -> (Vec<ColumnInfo>, Option<String>) {
    let cols = columns
        .iter()
        .map(|c| {
            // Resolve the declared width once here, at the catalog boundary,
            // so the write path (range validation) and the read path (OID and
            // binary payload width) cannot disagree.
            let raw_type = format!("{:?}", c.column_type);
            ColumnInfo {
                name: c.name.clone(),
                data_type: convert_column_type(&c.column_type),
                nullable: c.nullable,
                is_primary_key: c.primary_key,
                default: c.default.clone(),
                int_width: IntWidth::from_declared_type(&raw_type),
                float_width: FloatWidth::from_declared_type(&raw_type),
                raw_type: Some(raw_type),
            }
        })
        .collect();
    let pk = columns
        .iter()
        .find(|c| c.primary_key)
        .map(|c| c.name.clone());
    (cols, pk)
}

/// Build column metadata + primary key from a strict/KV schema.
pub(super) fn columns_from_strict_schema(
    schema: &StrictSchema,
) -> (Vec<ColumnInfo>, Option<String>) {
    columns_from_column_defs(&schema.columns)
}

/// Build column metadata + primary key from a live `ColumnarSchema`.
///
/// This is the schema the columnar engine actually encodes rows against —
/// the timeseries/spatial INSERT planner needs these exact columns, not the
/// descriptor's field hints.
pub(super) fn columns_from_columnar_schema(
    schema: &ColumnarSchema,
) -> (Vec<ColumnInfo>, Option<String>) {
    columns_from_column_defs(&schema.columns)
}

/// Build column metadata from `(name, type_hint)` descriptor field pairs.
pub(super) fn columns_from_fields(fields: &[(String, String)]) -> Vec<ColumnInfo> {
    fields
        .iter()
        .map(|(fname, type_hint)| {
            let ct = type_hint.parse::<nodedb_types::columnar::ColumnType>().ok();
            let data_type = ct
                .as_ref()
                .map(convert_column_type)
                .unwrap_or(SqlDataType::Bytes);
            ColumnInfo {
                name: fname.clone(),
                data_type,
                nullable: true,
                is_primary_key: false,
                default: None,
                int_width: IntWidth::from_declared_type(type_hint),
                float_width: FloatWidth::from_declared_type(type_hint),
                raw_type: Some(type_hint.clone()),
            }
        })
        .collect()
}

pub(super) fn parse_strict_config(config_json: Option<&str>) -> Option<StrictSchema> {
    config_json.and_then(|s| sonic_rs::from_str::<StrictSchema>(s).ok())
}

pub(super) fn parse_kv_config(config_json: Option<&str>) -> Option<nodedb_types::KvConfig> {
    config_json.and_then(|s| sonic_rs::from_str::<nodedb_types::KvConfig>(s).ok())
}

fn convert_column_type(ct: &nodedb_types::columnar::ColumnType) -> SqlDataType {
    use nodedb_types::columnar::ColumnType;
    match ct {
        ColumnType::Int64 => SqlDataType::Int64,
        ColumnType::Float64 => SqlDataType::Float64,
        ColumnType::String => SqlDataType::String,
        ColumnType::Bool => SqlDataType::Bool,
        ColumnType::Bytes | ColumnType::Geometry | ColumnType::Json => SqlDataType::Bytes,
        ColumnType::Timestamp | ColumnType::SystemTimestamp => SqlDataType::Timestamp,
        ColumnType::Decimal { .. } | ColumnType::Uuid | ColumnType::Ulid | ColumnType::Regex => {
            SqlDataType::String
        }
        ColumnType::Duration => SqlDataType::Int64,
        ColumnType::Array | ColumnType::Set | ColumnType::Range | ColumnType::Record => {
            SqlDataType::Bytes
        }
        ColumnType::Vector(dim) => SqlDataType::Vector(*dim as usize),
        _ => SqlDataType::Bytes,
    }
}
