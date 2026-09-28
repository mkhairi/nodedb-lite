// SPDX-License-Identifier: Apache-2.0

//! Text-index upkeep for SQL writes.
//!
//! Every SQL write to a schemaless or strict collection ends here, so a
//! SQL-written row is text-searchable and a SQL update or delete leaves no
//! stale terms. The index work itself is shared with the `NodeDb` trait
//! path: `engine::fts::maintain` for schemaless documents and
//! `engine::index_integration` for strict rows.

use nodedb_types::value::Value;

use crate::engine::fts::maintain;
use crate::engine::fts::maintain::FtsOutbound;
use crate::engine::index_integration::{
    deindex_row_text, index_geohash, index_row_text, pk_row_id, row_id,
};
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

/// The queue a SQL write's text is staged on for Origin, like the CRDT delta
/// the same write produces.
fn outbound<S: StorageEngine>(engine: &LiteQueryEngine<S>) -> Option<&FtsOutbound<S>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        engine.fts_outbound.as_deref()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = engine;
        None
    }
}

/// Bring the text entries of SQL-written schemaless documents in line with
/// their current CRDT state: a document that exists is re-indexed, one that
/// does not is removed.
pub(crate) fn reindex_documents<'a, S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    doc_ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), LiteError> {
    maintain::reindex_crdt_documents(
        &engine.fts_state,
        &engine.crdt,
        outbound(engine),
        collection,
        doc_ids,
    )
}

/// Index strict rows a SQL insert wrote, given their values in schema order.
pub(crate) fn index_strict_rows<'a, S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    rows: impl IntoIterator<Item = &'a [Value]>,
) -> Result<(), LiteError> {
    let Some(schema) = engine.strict.schema(collection) else {
        return Ok(());
    };
    for values in rows {
        index_row_text(
            collection,
            &row_id(&schema.columns, values),
            &schema.columns,
            values,
            &engine.fts_state.manager,
        )?;
    }
    Ok(())
}

/// Index columnar rows a SQL write stored, given their values in schema
/// order: text columns per field and whole, and a spatial profile's geohash.
pub(crate) fn index_columnar_rows<'a, S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    rows: impl IntoIterator<Item = &'a [Value]>,
) -> Result<(), LiteError> {
    let Some(schema) = engine.columnar.schema(collection) else {
        return Ok(());
    };
    let profile = engine.columnar.profile(collection);
    for values in rows {
        let id = row_id(&schema.columns, values);
        index_row_text(
            collection,
            &id,
            &schema.columns,
            values,
            &engine.fts_state.manager,
        )?;
        index_geohash(
            collection,
            &id,
            &schema,
            profile.as_ref(),
            values,
            &engine.fts_state.manager,
        )?;
    }
    Ok(())
}

/// Remove the text entries of columnar rows a SQL delete removed.
pub(crate) fn remove_columnar_rows<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    pks: &[Value],
) -> Result<(), LiteError> {
    for pk in pks {
        deindex_row_text(collection, &pk_row_id(pk), &engine.fts_state.manager)?;
    }
    Ok(())
}

/// Bring the text entries of strict rows in line with their stored state
/// after a SQL update or delete: a row that exists is re-indexed, a key with
/// no row is removed.
pub(crate) async fn reindex_strict_rows<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    pks: &[Value],
) -> Result<(), LiteError> {
    let Some(schema) = engine.strict.schema(collection) else {
        return Ok(());
    };
    for pk in pks {
        match engine.strict.get(collection, pk).await? {
            Some(values) => index_row_text(
                collection,
                &row_id(&schema.columns, &values),
                &schema.columns,
                &values,
                &engine.fts_state.manager,
            )?,
            None => deindex_row_text(collection, &pk_row_id(pk), &engine.fts_state.manager)?,
        }
    }
    Ok(())
}
