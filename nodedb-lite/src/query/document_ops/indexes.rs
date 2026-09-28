// SPDX-License-Identifier: Apache-2.0
//! Collection registration and secondary-index DDL for the Document engine.

use std::sync::Arc;

use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;

use crate::error::LiteError;
use crate::index::rebuild::build_index;
use crate::index::{
    IndexDef, IndexEngine, IndexPredicate, canonical_field, default_index_name, field_spec,
};
use crate::nodedb::collection::CollectionMeta;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

/// Register: initialize a collection in the appropriate engine.
///
/// For strict collections (`StorageMode::Strict`), the schema is persisted to
/// the strict engine. For schemaless collections, this is a no-op — CRDT
/// collections are discovered on first write.
pub async fn register<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    storage_mode: &nodedb_physical::physical_plan::document::types::StorageMode,
) -> Result<QueryResult, LiteError> {
    use nodedb_physical::physical_plan::document::types::StorageMode;
    match storage_mode {
        StorageMode::Strict { schema } => {
            engine
                .strict
                .create_collection(collection, schema.clone())
                .await?;
        }
        StorageMode::Schemaless => {
            // Schemaless collections are auto-discovered — no registration needed.
        }
    }
    Ok(affected(0))
}

/// A `CREATE INDEX`, however it was spelled.
pub struct CreateIndexRequest<'a> {
    /// Index name; `None` derives `idx_<collection>_<field>`.
    pub name: Option<&'a str>,
    pub collection: &'a str,
    /// The field as written: `email`, `$.a.b`, or `tags[]` for an array index.
    pub field: &'a str,
    pub unique: bool,
    pub case_insensitive: bool,
    /// The `WHERE` body of a partial index.
    pub predicate: Option<&'a str>,
    /// An index that already exists makes the statement a no-op.
    pub if_not_exists: bool,
}

fn affected(n: u64) -> QueryResult {
    QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        rows_affected: n,
        command: None,
    }
}

/// The engine whose rows an index on `collection` covers.
async fn index_engine_for<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<IndexEngine, LiteError> {
    if engine.strict.schema(collection).is_some() {
        return Ok(IndexEngine::Strict);
    }
    if engine.columnar.schema(collection).is_some() {
        return Err(LiteError::Unsupported {
            detail: format!(
                "CREATE INDEX on columnar collection '{collection}': columnar collections \
                 are filtered by block statistics, not secondary indexes"
            ),
        });
    }
    let meta_key = format!("collection:{collection}");
    if let Some(bytes) = engine
        .storage
        .get(Namespace::Meta, meta_key.as_bytes())
        .await?
    {
        let meta: CollectionMeta =
            sonic_rs::from_slice(&bytes).map_err(|e| LiteError::Serialization {
                detail: format!("metadata of collection '{collection}' does not decode: {e}"),
            })?;
        return Ok(match meta.collection_type.as_str() {
            "kv" => IndexEngine::KeyValue,
            "columnar" | "timeseries" | "spatial" => {
                return Err(LiteError::Unsupported {
                    detail: format!(
                        "CREATE INDEX on {} collection '{collection}' is not supported",
                        meta.collection_type
                    ),
                });
            }
            _ => IndexEngine::Document,
        });
    }
    let known = engine
        .crdt
        .lock()
        .map_err(|_| LiteError::LockPoisoned)?
        .collection_names()
        .iter()
        .any(|n| n == collection);
    if known {
        Ok(IndexEngine::Document)
    } else {
        Err(LiteError::BadRequest {
            detail: format!("CREATE INDEX: collection '{collection}' does not exist"),
        })
    }
}

/// `CREATE INDEX`: declare the index and build its entries from the rows the
/// collection holds. A unique index over rows that already share a value is
/// refused. Answers with the number of entries built.
pub async fn create_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    req: CreateIndexRequest<'_>,
) -> Result<QueryResult, LiteError> {
    let index_engine = index_engine_for(engine, req.collection).await?;
    declare_index(engine, req, index_engine).await
}

/// Declare an index over `index_engine` rows and build it: `CREATE INDEX`
/// once the engine is known.
pub(crate) async fn declare_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    req: CreateIndexRequest<'_>,
    index_engine: IndexEngine,
) -> Result<QueryResult, LiteError> {
    let (path, is_array) = canonical_field(req.field);
    let name = req
        .name
        .map(str::to_string)
        .unwrap_or_else(|| default_index_name(req.collection, req.field));
    if let Some(existing) = engine.indexes.def_named(&name) {
        if req.if_not_exists {
            return Ok(affected(0));
        }
        return Err(LiteError::BadRequest {
            detail: format!("index '{name}' already exists on '{}'", existing.collection),
        });
    }
    let spec = field_spec(&path, is_array);
    if let Some(existing) = engine.indexes.def_on_field(req.collection, &spec) {
        if req.if_not_exists {
            return Ok(affected(0));
        }
        return Err(LiteError::BadRequest {
            detail: format!(
                "collection '{}' already has index '{}' on {spec}: \
                 drop it before indexing the field again",
                req.collection, existing.name
            ),
        });
    }
    let def = Arc::new(IndexDef {
        name,
        collection: req.collection.to_string(),
        path,
        unique: req.unique,
        case_insensitive: req.case_insensitive,
        is_array,
        predicate: req.predicate.map(IndexPredicate::parse).transpose()?,
        engine: index_engine,
    });
    Ok(affected(rebuild_index(engine, def).await?))
}

/// `DROP INDEX`: remove the index and its entries. A missing index is an
/// error unless `if_exists`.
pub async fn drop_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    name: &str,
    if_exists: bool,
) -> Result<QueryResult, LiteError> {
    match engine.indexes.drop_index(&*engine.storage, name).await? {
        Some(_) => Ok(affected(0)),
        None if if_exists => Ok(affected(0)),
        None => Err(LiteError::BadRequest {
            detail: format!("index '{name}' does not exist"),
        }),
    }
}

/// The physical `DropIndex` op: drop the index on `field` of `collection`,
/// or the index named `field`. Dropping an index that does not exist is a
/// no-op.
pub async fn drop_field_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    field: &str,
) -> Result<QueryResult, LiteError> {
    let (path, is_array) = canonical_field(field);
    let name = engine
        .indexes
        .def_on_field(collection, &field_spec(&path, is_array))
        .map(|def| def.name.clone())
        .unwrap_or_else(|| field.to_string());
    drop_index(engine, &name, true).await
}

/// Rebuild the entries of one index from its collection's rows.
pub async fn rebuild_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    def: Arc<IndexDef>,
) -> Result<u64, LiteError> {
    build_index(
        &engine.indexes,
        &*engine.storage,
        &engine.crdt,
        &engine.strict,
        def,
    )
    .await
}

/// `REINDEX`: rebuild the index named `name`, or every index on
/// `collection` when no name is given. Answers with the entries built.
pub async fn reindex<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    name: Option<&str>,
) -> Result<QueryResult, LiteError> {
    let defs: Vec<Arc<IndexDef>> = match name {
        Some(name) => {
            let def = engine
                .indexes
                .def_named(name)
                .ok_or_else(|| LiteError::BadRequest {
                    detail: format!("index '{name}' does not exist"),
                })?;
            vec![def]
        }
        None => engine
            .indexes
            .defs()
            .into_iter()
            .filter(|d| d.collection == collection)
            .collect(),
    };
    let mut built = 0;
    for def in defs {
        built += rebuild_index(engine, def).await?;
    }
    Ok(affected(built))
}

/// The physical `BackfillIndex` op: rebuild the index already declared on the
/// field, or declare one under the derived name and build it.
pub async fn backfill_index<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    req: CreateIndexRequest<'_>,
) -> Result<QueryResult, LiteError> {
    let (path, is_array) = canonical_field(req.field);
    match engine
        .indexes
        .def_on_field(req.collection, &field_spec(&path, is_array))
    {
        Some(def) => Ok(affected(rebuild_index(engine, def).await?)),
        None => create_index(engine, req).await,
    }
}
