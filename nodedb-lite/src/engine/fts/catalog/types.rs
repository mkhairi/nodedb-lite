// SPDX-License-Identifier: Apache-2.0

//! Search declaration shapes and canonical validation.

use std::collections::HashSet;

use crate::error::LiteError;

/// Version 2 adds [`SearchDeclaration::per_field`]. A version-1 record reads
/// as `per_field = true`, the behaviour it was written under.
pub(crate) const DECLARATION_FORMAT: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub(crate) struct SearchDeclaration {
    pub name: String,
    pub fields: Vec<String>,
    pub analyzer: String,
    pub fuzzy: bool,
    /// `false` keeps only the whole-document index: no per-field index is
    /// built, and a search on a named field answers `TextIndexMissing`.
    pub per_field: bool,
}

/// A version-1 declaration, read only: written before `per_field` existed.
#[derive(Debug, Clone, zerompk::FromMessagePack)]
pub(crate) struct SearchDeclarationV1 {
    pub name: String,
    pub fields: Vec<String>,
    pub analyzer: String,
    pub fuzzy: bool,
}

/// A version-1 record, read only.
#[derive(Debug, Clone, zerompk::FromMessagePack)]
pub(crate) struct SearchDeclarationRecordV1 {
    pub format_version: u32,
    pub revision: u64,
    pub declaration: Option<SearchDeclarationV1>,
}

impl From<SearchDeclarationRecordV1> for SearchDeclarationRecord {
    fn from(v1: SearchDeclarationRecordV1) -> Self {
        Self {
            format_version: DECLARATION_FORMAT,
            revision: v1.revision,
            declaration: v1.declaration.map(|d| SearchDeclaration {
                name: d.name,
                fields: d.fields,
                analyzer: d.analyzer,
                fuzzy: d.fuzzy,
                per_field: true,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub(crate) struct SearchDeclarationRecord {
    pub format_version: u32,
    pub revision: u64,
    pub declaration: Option<SearchDeclaration>,
}

impl SearchDeclarationRecord {
    pub(crate) fn check(&self, collection: &str) -> Result<(), LiteError> {
        if self.format_version != DECLARATION_FORMAT || self.revision == 0 {
            return Err(LiteError::Serialization {
                detail: format!("invalid search declaration version/revision for '{collection}'"),
            });
        }
        if let Some(declaration) = &self.declaration {
            let unique: HashSet<&str> = declaration.fields.iter().map(String::as_str).collect();
            if declaration.name != format!("fts_{collection}")
                || declaration.analyzer.is_empty()
                || declaration.fields.iter().any(String::is_empty)
                || unique.len() != declaration.fields.len()
            {
                return Err(LiteError::BadRequest {
                    detail: format!(
                        "invalid search declaration for '{collection}': use its canonical name and distinct fields"
                    ),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_declarations_and_revision_tombstones_are_distinct() {
        let mut record = SearchDeclarationRecord {
            format_version: DECLARATION_FORMAT,
            revision: 1,
            declaration: Some(SearchDeclaration {
                name: "fts_docs".into(),
                fields: vec!["title".into()],
                analyzer: "standard".into(),
                fuzzy: false,
                per_field: true,
            }),
        };
        assert!(record.check("docs").is_ok());
        assert!(record.check("other").is_err());
        record
            .declaration
            .as_mut()
            .unwrap()
            .fields
            .push("title".into());
        assert!(record.check("docs").is_err());
        record.declaration = None;
        assert!(record.check("docs").is_ok());
        record.revision = 0;
        assert!(record.check("docs").is_err());
    }
}
