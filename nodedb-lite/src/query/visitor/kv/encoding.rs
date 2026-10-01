// SPDX-License-Identifier: Apache-2.0

use crate::error::LiteError;
use crate::query::filter_convert::sql_value_to_value;
use nodedb_sql::types_expr::SqlValue;

// ── Value encoding ────────────────────────────────────────────────────────────

/// Raw bytes for a lone `value` column, by Origin's rule: a scalar encodes
/// as `nodedb_types::scalar_to_raw_bytes` writes it, and an array is
/// PostgreSQL array text.
fn sql_value_raw_bytes(v: &SqlValue) -> Vec<u8> {
    match v {
        SqlValue::Bytes(b) => b.clone(),
        SqlValue::Array(values) => pg_array_text(values).into_bytes(),
        SqlValue::Int(_)
        | SqlValue::Float(_)
        | SqlValue::Decimal(_)
        | SqlValue::String(_)
        | SqlValue::Bool(_)
        | SqlValue::Null
        | SqlValue::Timestamp(_)
        | SqlValue::Timestamptz(_) => pg_text(v).into_bytes(),
    }
}

/// The text form of one SQL value, as Origin's `sql_value_to_string` writes
/// it.
fn pg_text(v: &SqlValue) -> String {
    match v {
        SqlValue::String(s) => s.clone(),
        SqlValue::Int(i) => i.to_string(),
        SqlValue::Float(f) => f.to_string(),
        SqlValue::Decimal(d) => d.to_string(),
        SqlValue::Bool(b) => b.to_string(),
        SqlValue::Timestamp(at) | SqlValue::Timestamptz(at) => at.to_iso8601(),
        SqlValue::Bytes(b) => {
            let hex: String = b.iter().map(|byte| format!("{byte:02x}")).collect();
            format!("\\x{hex}")
        }
        SqlValue::Array(values) => pg_array_text(values),
        SqlValue::Null => String::new(),
    }
}

/// PostgreSQL array text: `{a,"two words",NULL}`.
fn pg_array_text(values: &[SqlValue]) -> String {
    let elements: Vec<String> = values
        .iter()
        .map(|value| match value {
            SqlValue::Null => "NULL".to_string(),
            SqlValue::String(s) if pg_array_string_needs_quotes(s) => {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            }
            other => pg_text(other),
        })
        .collect();
    format!("{{{}}}", elements.join(","))
}

fn pg_array_string_needs_quotes(value: &str) -> bool {
    value.is_empty()
        || value.eq_ignore_ascii_case("null")
        || value
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, ',' | '{' | '}' | '"' | '\\'))
}

/// Encode a KV insert's value columns as the stored body.
///
/// A lone `value` column stores its raw bytes. Any other column set stores
/// the map body `kv_ops::body::encode_kv_body` writes, the same body an
/// Origin row apply stores.
pub(super) fn encode_kv_value(value_cols: &[(String, SqlValue)]) -> Result<Vec<u8>, LiteError> {
    if value_cols.len() == 1 && value_cols[0].0 == "value" {
        return Ok(sql_value_raw_bytes(&value_cols[0].1));
    }
    let mut map = std::collections::HashMap::with_capacity(value_cols.len());
    for (col, sv) in value_cols {
        map.insert(col.clone(), sql_value_to_value(sv)?);
    }
    crate::query::kv_ops::body::encode_kv_body(map, nodedb_query::msgpack_scan::KvBodyShape::Map)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_lone_value_column_stores_origin_raw_bytes() {
        let cases = [
            (SqlValue::String("v1".into()), b"v1".to_vec()),
            (SqlValue::Int(7), b"7".to_vec()),
            (SqlValue::Float(1.5), b"1.5".to_vec()),
            (SqlValue::Bool(false), b"false".to_vec()),
            (SqlValue::Bytes(vec![0xff]), vec![0xff]),
            (SqlValue::Null, Vec::new()),
            (
                SqlValue::Array(vec![
                    SqlValue::String("public".into()),
                    SqlValue::String("two words".into()),
                    SqlValue::Null,
                ]),
                b"{public,\"two words\",NULL}".to_vec(),
            ),
        ];
        for (sql, expected) in cases {
            assert_eq!(super::sql_value_raw_bytes(&sql), expected, "{sql:?}");
        }
    }
}
