// SPDX-License-Identifier: Apache-2.0
use nodedb_types::columnar::{ColumnDef, ColumnType, StrictSchema};
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::ddl::parser::parse_strict_create_sql;

/// The error a conversion returns after rolling back: a storage-class
/// failure keeps its type, anything else is the row not fitting the target.
pub(super) fn conversion_refused(source: &str, target: &str, row: &str, e: LiteError) -> LiteError {
    match e {
        LiteError::Storage { .. } | LiteError::Corrupted { .. } | LiteError::LockPoisoned => e,
        other => LiteError::BadRequest {
            detail: format!(
                "CONVERT COLLECTION '{source}' TO {target} stopped at row '{row}' and was \
                 rolled back: {other}"
            ),
        },
    }
}

/// Parse CONVERT COLLECTION <name> TO <mode> [(<col_defs>)]
pub(super) fn parse_convert_sql(
    sql: &str,
    target_mode: &str,
) -> Result<(String, StrictSchema), LiteError> {
    let parts: Vec<&str> = sql.split_whitespace().collect();
    let source_name = parts
        .get(2)
        .ok_or(LiteError::Query("expected collection name".into()))?
        .to_lowercase();

    // Validate that the SQL TO clause matches the expected target mode.
    if let Some(to_idx) = parts.iter().position(|p| p.eq_ignore_ascii_case("TO"))
        && let Some(mode) = parts.get(to_idx + 1)
        && !mode.eq_ignore_ascii_case(target_mode)
    {
        return Err(LiteError::Query(format!(
            "expected CONVERT TO {target_mode}, got '{mode}'"
        )));
    }

    // If there are column defs in parens, parse them.
    if sql.contains('(') {
        let (_, schema) = parse_strict_create_sql(sql)?;
        Ok((source_name, schema))
    } else {
        Ok((source_name, default_convert_schema()))
    }
}

/// Target schema when a CONVERT names no columns: a text `id` key plus a
/// text `data` column.
pub(in crate::query) fn default_convert_schema() -> StrictSchema {
    StrictSchema {
        columns: vec![
            ColumnDef::required("id", ColumnType::String).with_primary_key(),
            ColumnDef::nullable("data", ColumnType::String),
        ],
        version: 1,
        dropped_columns: Vec::new(),
        bitemporal: false,
    }
}

/// Convert a Document's fields to a Vec<Value> matching the target schema.
pub(super) fn document_to_row(
    fields: &std::collections::HashMap<String, Value>,
    target_columns: &[ColumnDef],
) -> Vec<Value> {
    target_columns
        .iter()
        .map(|col| fields.get(&col.name).cloned().unwrap_or(Value::Null))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_convert_with_schema() {
        let sql = "CONVERT COLLECTION users TO strict (id BIGINT NOT NULL PRIMARY KEY, name TEXT)";
        let (name, schema) = parse_convert_sql(sql, "strict").expect("parse");
        assert_eq!(name, "users");
        assert_eq!(schema.columns.len(), 2);
    }

    #[test]
    fn parse_convert_without_schema() {
        let sql = "CONVERT COLLECTION users TO document";
        let (name, _schema) = parse_convert_sql(sql, "document").expect("parse");
        assert_eq!(name, "users");
    }

    #[test]
    fn document_to_row_maps_fields() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("name".into(), Value::String("Alice".into()));
        fields.insert("age".into(), Value::Integer(30));

        let columns = vec![
            ColumnDef::required("name", ColumnType::String),
            ColumnDef::nullable("age", ColumnType::Int64),
            ColumnDef::nullable("email", ColumnType::String),
        ];

        let row = document_to_row(&fields, &columns);
        assert_eq!(row.len(), 3);
        assert_eq!(row[0], Value::String("Alice".into()));
        assert_eq!(row[1], Value::Integer(30));
        assert_eq!(row[2], Value::Null); // email not in doc.
    }
}
