// SPDX-License-Identifier: Apache-2.0

//! Search declaration shapes and canonical validation.

use std::collections::HashSet;

use crate::error::LiteError;

pub(crate) const DECLARATION_FORMAT: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub(crate) struct SearchDeclaration {
    pub name: String,
    pub fields: Vec<String>,
    pub analyzer: String,
    pub fuzzy: bool,
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
