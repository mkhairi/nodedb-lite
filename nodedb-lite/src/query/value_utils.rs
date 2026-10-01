// SPDX-License-Identifier: Apache-2.0
//! Shared scalar-to-string conversions used by index keys and indexed lookups.

use nodedb_types::value::Value;

/// Convert a scalar `Value` into the canonical string form used as a
/// component of an index key. Non-scalar variants collapse to the empty
/// string so they can still produce a deterministic key segment.
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Uuid(s) => s.clone(),
        Value::Null => String::new(),
        _ => String::new(),
    }
}

/// Convert a scalar `LoroValue` into the canonical string form used as a
/// component of an index key. Containers and binary blobs collapse to the
/// empty string for the same reason as `value_to_string`.
pub fn loro_value_to_string(v: &loro::LoroValue) -> String {
    loro_value_to_index_key(v).unwrap_or_default()
}

/// Posting key of a document field value in a field index.
///
/// The one stringification shared by the index writers and the SQL index
/// lookup ([`sql_value_to_index_key`]), so an equality literal and a stored
/// value meet on the same key. An integer and an integral float share a key
/// (`1` and `1.0` are both `"1"`), as SQL equality treats them. Null, binary
/// and container values return `None` and are not indexed.
pub fn loro_value_to_index_key(v: &loro::LoroValue) -> Option<String> {
    match v {
        loro::LoroValue::String(s) => Some(s.to_string()),
        loro::LoroValue::I64(n) => Some(n.to_string()),
        loro::LoroValue::Double(f) => Some(f.to_string()),
        loro::LoroValue::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Posting key an SQL equality literal looks up, through
/// [`loro_value_to_index_key`].
///
/// A decimal literal (`1.5`) takes the key of the float it converts to, as
/// the scan filter converts it. `None` for a literal no stored value is
/// indexed under: null, bytes, arrays and timestamps. SQL `= NULL` is never
/// true, so a null literal must find no posting rather than the empty key.
pub fn sql_value_to_index_key(v: &nodedb_sql::types_expr::SqlValue) -> Option<String> {
    use nodedb_sql::types_expr::SqlValue;
    let scalar = match v {
        SqlValue::String(s) => loro::LoroValue::String(s.clone().into()),
        SqlValue::Int(i) => loro::LoroValue::I64(*i),
        SqlValue::Float(f) => loro::LoroValue::Double(*f),
        SqlValue::Decimal(d) => loro::LoroValue::Double(d.to_string().parse::<f64>().ok()?),
        SqlValue::Bool(b) => loro::LoroValue::Bool(*b),
        _ => return None,
    };
    loro_value_to_index_key(&scalar)
}

#[cfg(test)]
mod tests {
    use loro::LoroValue;
    use nodedb_sql::types_expr::SqlValue;

    use super::*;

    /// The SQL lookup key and the stored-value key agree for every scalar
    /// type an equality can name, a decimal literal included, and null is
    /// never indexed or looked up.
    #[test]
    fn index_key_parity_between_sql_literal_and_stored_value() {
        let pairs = [
            (SqlValue::Bool(true), LoroValue::Bool(true), "true"),
            (SqlValue::Bool(false), LoroValue::Bool(false), "false"),
            (SqlValue::Int(-7), LoroValue::I64(-7), "-7"),
            (SqlValue::Float(2.5), LoroValue::Double(2.5), "2.5"),
            (SqlValue::Float(1.0), LoroValue::I64(1), "1"),
            (SqlValue::Int(1), LoroValue::Double(1.0), "1"),
            (
                SqlValue::Decimal(rust_decimal::Decimal::new(15, 1)),
                LoroValue::Double(1.5),
                "1.5",
            ),
            (
                SqlValue::Decimal(rust_decimal::Decimal::new(20, 1)),
                LoroValue::I64(2),
                "2",
            ),
            (
                SqlValue::String("team-a".into()),
                LoroValue::String("team-a".into()),
                "team-a",
            ),
        ];
        for (sql, stored, key) in pairs {
            assert_eq!(
                sql_value_to_index_key(&sql).as_deref(),
                Some(key),
                "sql {sql:?}"
            );
            assert_eq!(
                loro_value_to_index_key(&stored).as_deref(),
                Some(key),
                "stored {stored:?}"
            );
        }

        assert_eq!(loro_value_to_index_key(&LoroValue::Null), None);
        assert_eq!(sql_value_to_index_key(&SqlValue::Null), None);
        assert_eq!(loro_value_to_string(&LoroValue::Null), "");
    }
}
