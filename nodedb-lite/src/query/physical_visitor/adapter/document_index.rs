// SPDX-License-Identifier: Apache-2.0
//! Secondary-index `DocumentOp`s: index reads and index DDL.

use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::document_ops::index_reads;
use crate::query::document_ops::indexes::{self, CreateIndexRequest};
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::LitePhysicalFut;

/// `IndexedFetch`. The op carries the probe as text; the coerced equality the
/// read confirms candidates with matches it against the stored value
/// whatever that value's type.
pub(super) fn indexed_fetch<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    path: &str,
    value: &str,
    limit: usize,
    offset: usize,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let collection = collection.to_string();
    let path = path.to_string();
    let probe = Value::String(value.to_string());
    Ok(Box::pin(async move {
        index_reads::indexed_fetch(engine, &collection, &path, &probe, limit, offset).await
    }))
}

/// `IndexLookup`: the matching document ids.
pub(super) fn index_lookup<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    path: &str,
    value: &str,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let collection = collection.to_string();
    let path = path.to_string();
    let probe = Value::String(value.to_string());
    Ok(Box::pin(async move {
        index_reads::index_lookup(engine, &collection, &path, &probe).await
    }))
}

/// `DropIndex`: drop the index on `field`, or the index named `field`.
pub(super) fn drop_index<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    field: &str,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let collection = collection.to_string();
    let field = field.to_string();
    Ok(Box::pin(async move {
        indexes::drop_field_index(engine, &collection, &field).await
    }))
}

/// The declaration a `BackfillIndex` op carries.
pub(super) struct BackfillFlags<'o> {
    pub path: &'o str,
    pub is_array: bool,
    pub unique: bool,
    pub case_insensitive: bool,
    pub predicate: Option<&'o str>,
}

/// `BackfillIndex`: rebuild the index on the field, or declare it with the
/// op's flags and build it.
pub(super) fn backfill_index<'a, S: StorageEngine + 'a>(
    engine: &'a LiteQueryEngine<S>,
    collection: &str,
    flags: BackfillFlags<'_>,
) -> Result<LitePhysicalFut<'a>, LiteError> {
    let collection = collection.to_string();
    let field = if flags.is_array && !flags.path.ends_with("[]") {
        format!("{}[]", flags.path)
    } else {
        flags.path.to_string()
    };
    let (unique, case_insensitive) = (flags.unique, flags.case_insensitive);
    let predicate = flags.predicate.map(str::to_string);
    Ok(Box::pin(async move {
        let req = CreateIndexRequest {
            name: None,
            collection: &collection,
            field: &field,
            unique,
            case_insensitive,
            predicate: predicate.as_deref(),
            if_not_exists: false,
        };
        indexes::backfill_index(engine, req).await
    }))
}
