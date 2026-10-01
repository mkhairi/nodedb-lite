// SPDX-License-Identifier: Apache-2.0

//! Stable field ordering preserves whole-document phrase positions.

use crate::engine::fts::catalog::SearchDeclaration;
use nodedb_types::Value;
use std::collections::{HashMap, HashSet};

pub(super) fn selected_texts<'a>(
    fields: &'a HashMap<String, Value>,
    declaration: Option<&SearchDeclaration>,
) -> Vec<(&'a str, &'a str)> {
    let selected: Option<HashSet<&str>> = declaration
        .filter(|declaration| !declaration.fields.is_empty())
        .map(|declaration| declaration.fields.iter().map(String::as_str).collect());
    let mut texts: Vec<_> = fields
        .iter()
        .filter(|(field, _)| {
            selected
                .as_ref()
                .is_none_or(|selected| selected.contains(field.as_str()))
        })
        .filter_map(|(field, value)| match value {
            Value::String(text) => Some((field.as_str(), text.as_str())),
            _ => None,
        })
        .collect();
    texts.sort_unstable_by_key(|(field, _)| *field);
    texts
}

pub(super) fn joined_text(texts: &[(&str, &str)]) -> String {
    let mut joined = String::new();
    for (position, (_, text)) in texts.iter().enumerate() {
        if position != 0 {
            joined.push(' ');
        }
        joined.push_str(text);
    }
    joined
}
