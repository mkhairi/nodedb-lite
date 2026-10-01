// SPDX-License-Identifier: Apache-2.0
//! Persisted secondary-index specs for schemaless document collections.
//!
//! `CREATE INDEX` on a schemaless collection writes one spec record to the
//! Meta namespace under `index_spec:{collection}:{field}`. `field` is stored in the
//! planner's canonical JSON-path form (`$.scope`), the form the index-lookup
//! rewrite compares against. The SQL catalog reads the records back and lists
//! them on the collection's `CollectionInfo`. A store written before specs
//! existed holds none and opens unchanged.
//!
//! The persisted state is always `Building`. Readiness is derived when a
//! statement is planned ([`load_planner_index_specs`]): a spec is `Ready`
//! iff the CRDT engine holds its built postings and the lookup answers it
//! exactly as a full scan would.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use nodedb_sql::types::{EngineType, IndexSpec, IndexState, SqlCatalog};
use nodedb_types::Namespace;

use crate::engine::crdt::CrdtEngine;
use crate::error::LiteError;
use crate::query::catalog::LiteCatalog;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::{StorageEngine, WriteOp};

use super::indexes::bare_path;

/// Meta-namespace key prefix for index spec records.
const META_INDEX_SPEC_PREFIX: &str = "index_spec:";

/// Build state of a persisted index. Only `Ready` lets the planner rewrite an
/// equality into an index lookup. Every record is written `Building`; the
/// planner sees `Ready` only through [`load_planner_index_specs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum PersistedIndexState {
    Building,
    Ready,
}

/// One persisted index spec, encoded as JSON like `CollectionMeta`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PersistedIndexSpec {
    collection: String,
    name: String,
    /// Canonical JSON-path form, for example `$.scope`.
    field: String,
    unique: bool,
    case_insensitive: bool,
    /// Partial-index predicate as SQL text. The planner rejects partial
    /// indexes today, so this is always `None`.
    predicate: Option<String>,
    state: PersistedIndexState,
}

impl PersistedIndexSpec {
    fn to_planner(&self) -> IndexSpec {
        IndexSpec {
            name: self.name.clone(),
            field: self.field.clone(),
            unique: self.unique,
            case_insensitive: self.case_insensitive,
            // A persisted Ready is never trusted: readiness is derived by
            // `load_planner_index_specs`.
            state: IndexState::Building,
            predicate: self.predicate.clone(),
        }
    }
}

/// Canonical index field form. Must match the private `canonical_index_field`
/// in `nodedb-sql` (`engine_rules/index_lookup.rs`): a path that starts with
/// `$` is kept, a bare name `f` becomes `$.f`. The planner compares an index
/// spec's field against that form, so a drift here disables index lookups.
pub(crate) fn canonical_index_field(field: &str) -> String {
    if field.starts_with('$') {
        field.to_string()
    } else {
        format!("$.{field}")
    }
}

fn spec_key(collection: &str, canonical_field: &str) -> String {
    format!("{META_INDEX_SPEC_PREFIX}{collection}:{canonical_field}")
}

/// Whether `canonical_field` names one top-level key of a document, the only
/// form the in-memory postings index (`$.scope`, not `$.a.b` or `$[0]`).
fn is_top_level_field(canonical_field: &str) -> bool {
    let bare = bare_path(canonical_field);
    !bare.is_empty() && !bare.contains(['.', '['])
}

/// Whether an index lookup on `canonical_field` returns the rows a full scan
/// returns for the same equality.
///
/// - A nested path indexes nothing.
/// - `id` names the row id in the scan, not the document's `id` key.
/// - A case-insensitive index receives the literal lowercased, so the scan's
///   case-sensitive equality cannot be checked on the fetched rows.
fn lookup_matches_scan(canonical_field: &str, case_insensitive: bool) -> bool {
    !case_insensitive && is_top_level_field(canonical_field) && bare_path(canonical_field) != "id"
}

/// What a `CREATE INDEX` writes, from [`prepare_index_spec`].
pub(crate) enum IndexSpecPlan {
    /// The target is not a schemaless document collection. No spec is
    /// written, and the index entries are backfilled.
    NotSchemaless,
    /// `IF NOT EXISTS` found the spec already written. Nothing is written.
    Exists,
    /// A new spec to write with [`write_index_spec`].
    New(PendingIndexSpec),
}

/// An index spec record ready to write, produced by [`prepare_index_spec`].
pub(crate) struct PendingIndexSpec {
    key: String,
    bytes: Vec<u8>,
    collection: String,
    /// Canonical JSON-path form.
    field: String,
    case_insensitive: bool,
}

/// Build the spec record for a `CREATE INDEX` when the target is a schemaless
/// document collection. Other engines get no spec record
/// ([`IndexSpecPlan::NotSchemaless`]).
///
/// Reads the key first and writes nothing, so the caller can refuse a
/// duplicate before any side effect. An existing spec is an error unless
/// `if_not_exists`, which returns [`IndexSpecPlan::Exists`] and keeps the
/// existing record. A nested path is refused: the postings index top-level
/// keys only.
///
/// The spec is stored as `Building`. Readiness is derived at planning time.
pub(crate) async fn prepare_index_spec<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    index_name: Option<&str>,
    collection: &str,
    field: &str,
    unique: bool,
    case_insensitive: bool,
    if_not_exists: bool,
) -> Result<IndexSpecPlan, LiteError> {
    if !is_schemaless_document(engine, collection).await? {
        return Ok(IndexSpecPlan::NotSchemaless);
    }
    let canonical = canonical_index_field(field);
    let key = spec_key(collection, &canonical);
    if let Some(existing) = engine.storage.get(Namespace::Meta, key.as_bytes()).await? {
        if if_not_exists {
            return Ok(IndexSpecPlan::Exists);
        }
        let existing: PersistedIndexSpec =
            sonic_rs::from_slice(&existing).map_err(|e| LiteError::Serialization {
                detail: format!("decode index spec: {e}"),
            })?;
        return Err(LiteError::Query(format!(
            "index on {collection}({field}) already exists as {}",
            existing.name
        )));
    }
    if !is_top_level_field(&canonical) {
        return Err(LiteError::BadRequest {
            detail: format!(
                "CREATE INDEX on {collection}({field}): a schemaless collection indexes \
                 top-level fields only; index a top-level field instead of a nested path"
            ),
        });
    }
    let default_name = || {
        let bare = canonical.strip_prefix("$.").unwrap_or(&canonical);
        format!("idx_{collection}_{bare}")
    };
    let spec = PersistedIndexSpec {
        collection: collection.to_string(),
        name: index_name.map_or_else(default_name, str::to_string),
        field: canonical.clone(),
        unique,
        case_insensitive,
        predicate: None,
        state: PersistedIndexState::Building,
    };
    let bytes = sonic_rs::to_vec(&spec).map_err(|e| LiteError::Serialization {
        detail: format!("encode index spec: {e}"),
    })?;
    Ok(IndexSpecPlan::New(PendingIndexSpec {
        key,
        bytes,
        collection: collection.to_string(),
        field: canonical,
        case_insensitive,
    }))
}

/// Write a spec record built by [`prepare_index_spec`], then build the
/// index's in-memory postings from the collection's current documents.
pub(crate) async fn write_index_spec<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    pending: PendingIndexSpec,
) -> Result<(), LiteError> {
    engine
        .storage
        .put(Namespace::Meta, pending.key.as_bytes(), &pending.bytes)
        .await?;
    engine
        .crdt
        .lock()
        .map_err(|_| LiteError::LockPoisoned)?
        .register_field_index(
            &pending.collection,
            &pending.field,
            pending.case_insensitive,
        );
    Ok(())
}

/// Resolve `collection` through the SQL catalog. A collection no engine knows
/// yet counts as schemaless: Lite registers a schemaless collection on its
/// first write.
async fn is_schemaless_document<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
) -> Result<bool, LiteError> {
    let metas =
        crate::nodedb::collection::ddl::load_persisted_collection_metas(engine.storage.as_ref())
            .await
            .map_err(|e| LiteError::Storage {
                detail: e.to_string(),
            })?;
    let catalog = LiteCatalog::new(
        Arc::clone(&engine.crdt),
        Arc::clone(&engine.strict),
        Arc::clone(&engine.columnar),
        metas,
    );
    let info = catalog
        .get_collection(nodedb_types::DatabaseId::DEFAULT, collection)
        .map_err(|e| LiteError::Query(e.to_string()))?;
    Ok(info.is_none_or(|i| i.engine == EngineType::DocumentSchemaless))
}

/// The spec records a `DROP INDEX` names, from [`index_spec_drop_ops`].
pub(crate) struct IndexSpecDrop {
    /// Delete ops for the spec records.
    pub(crate) ops: Vec<WriteOp>,
    /// `(collection, canonical field)` of each spec, for removing its
    /// in-memory index once the delete is durable.
    pub(crate) indexes: Vec<(String, String)>,
}

/// Delete ops for the spec records a `DROP INDEX` names.
///
/// With a collection, a spec matches by index name or by field. Without one,
/// a spec in any collection matches by index name only.
pub(crate) async fn index_spec_drop_ops<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    name_or_field: &str,
) -> Result<IndexSpecDrop, LiteError> {
    let prefix = if collection.is_empty() {
        META_INDEX_SPEC_PREFIX.to_string()
    } else {
        format!("{META_INDEX_SPEC_PREFIX}{collection}:")
    };
    let pairs = engine
        .storage
        .scan_prefix(Namespace::Meta, prefix.as_bytes())
        .await?;
    let canonical = canonical_index_field(name_or_field);
    let mut ops = Vec::new();
    let mut indexes = Vec::new();
    for (key, value) in pairs {
        // The Meta namespace holds other key families under the same prefix
        // bytes. A record that does not decode as a spec is not one.
        let Ok(spec) = sonic_rs::from_slice::<PersistedIndexSpec>(&value) else {
            continue;
        };
        let matches = if collection.is_empty() {
            spec.name == name_or_field
        } else {
            spec.collection == collection && (spec.name == name_or_field || spec.field == canonical)
        };
        if matches {
            ops.push(WriteOp::Delete {
                ns: Namespace::Meta,
                key,
            });
            indexes.push((spec.collection, spec.field));
        }
    }
    Ok(IndexSpecDrop { ops, indexes })
}

/// Delete ops for every spec record of `collection`, for `DROP COLLECTION`.
pub(crate) async fn index_spec_drop_all_ops<S: StorageEngine>(
    storage: &S,
    collection: &str,
) -> Result<Vec<WriteOp>, LiteError> {
    let prefix = format!("{META_INDEX_SPEC_PREFIX}{collection}:");
    let pairs = storage
        .scan_prefix(Namespace::Meta, prefix.as_bytes())
        .await?;
    Ok(pairs
        .into_iter()
        .map(|(key, _)| WriteOp::Delete {
            ns: Namespace::Meta,
            key,
        })
        .collect())
}

/// Load every persisted index spec, grouped by collection name, in the
/// planner's `IndexSpec` form.
pub(crate) async fn load_index_specs<S: StorageEngine>(
    storage: &S,
) -> Result<HashMap<String, Vec<IndexSpec>>, LiteError> {
    let pairs = storage
        .scan_prefix(Namespace::Meta, META_INDEX_SPEC_PREFIX.as_bytes())
        .await?;
    let mut map: HashMap<String, Vec<IndexSpec>> = HashMap::new();
    for (key, value) in &pairs {
        // Skip records that are not specs, as the collection-meta loader does.
        match sonic_rs::from_slice::<PersistedIndexSpec>(value) {
            Ok(spec) => map
                .entry(spec.collection.clone())
                .or_default()
                .push(spec.to_planner()),
            Err(e) => tracing::warn!(
                key = %String::from_utf8_lossy(key),
                error = %e,
                "index spec decode failed; skipping record"
            ),
        }
    }
    Ok(map)
}

/// Load every persisted index spec with the state the planner plans against.
///
/// A spec is `Ready` iff `crdt` holds its built postings and the lookup
/// returns the rows a full scan returns ([`lookup_matches_scan`]). Every
/// other spec is `Building`, whatever its record says, so the planner keeps
/// the full scan. The
/// postings are registered on `CREATE INDEX` and rebuilt at open, so a spec
/// turns `Ready` at either point with no write.
pub(crate) async fn load_planner_index_specs<S: StorageEngine>(
    storage: &S,
    crdt: &Mutex<CrdtEngine>,
) -> Result<HashMap<String, Vec<IndexSpec>>, LiteError> {
    let mut specs = load_index_specs(storage).await?;
    let crdt = crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
    for (collection, collection_specs) in specs.iter_mut() {
        for spec in collection_specs.iter_mut() {
            let ready = lookup_matches_scan(&spec.field, spec.case_insensitive)
                && crdt.has_field_index(collection, &spec.field);
            spec.state = if ready {
                IndexState::Ready
            } else {
                IndexState::Building
            };
        }
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PagedbStorageMem;

    #[test]
    fn canonical_index_field_forms() {
        assert_eq!(canonical_index_field("scope"), "$.scope");
        assert_eq!(canonical_index_field("$.a.b"), "$.a.b");
        assert_eq!(canonical_index_field("a.b"), "$.a.b");
    }

    #[test]
    fn lookup_matches_scan_for_top_level_case_sensitive_fields_only() {
        assert!(lookup_matches_scan("$.scope", false));
        assert!(!lookup_matches_scan("$.scope", true));
        assert!(!lookup_matches_scan("$.a.b", false));
        assert!(!lookup_matches_scan("$[0]", false));
        assert!(!lookup_matches_scan("$.id", false));
        assert!(!lookup_matches_scan("$", false));
    }

    /// A record persisted as `Ready` is demoted when the derivation says not
    /// ready: a nested field, or a top-level field with no postings.
    #[tokio::test]
    async fn persisted_ready_loads_as_building_without_postings() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        for field in ["$.a.b", "$.scope"] {
            let spec = PersistedIndexSpec {
                collection: "notes".into(),
                name: format!("idx_{field}"),
                field: field.into(),
                unique: false,
                case_insensitive: false,
                predicate: None,
                state: PersistedIndexState::Ready,
            };
            storage
                .put(
                    Namespace::Meta,
                    spec_key("notes", field).as_bytes(),
                    &sonic_rs::to_vec(&spec).unwrap(),
                )
                .await
                .unwrap();
        }
        let loaded = load_index_specs(&storage).await.unwrap();
        assert!(
            loaded["notes"]
                .iter()
                .all(|s| matches!(s.state, IndexState::Building)),
            "the loader never surfaces a persisted Ready"
        );
        let crdt = Mutex::new(CrdtEngine::new(1).unwrap());
        let planned = load_planner_index_specs(&storage, &crdt).await.unwrap();
        assert!(
            planned["notes"]
                .iter()
                .all(|s| matches!(s.state, IndexState::Building)),
            "no postings or nested path: planner keeps the full scan"
        );
    }

    #[tokio::test]
    async fn loader_skips_non_spec_record_under_prefix() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        storage
            .put(
                Namespace::Meta,
                format!("{META_INDEX_SPEC_PREFIX}notes:junk").as_bytes(),
                b"not json",
            )
            .await
            .unwrap();
        let specs = load_index_specs(&storage).await.unwrap();
        assert!(specs.is_empty());
    }
}
