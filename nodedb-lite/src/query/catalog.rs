//! SqlCatalog implementation for Lite.
//!
//! Resolves collection metadata for query planning. Collections that carry
//! persisted metadata (created via DDL or materialized from an inbound sync
//! schema announcement) are surfaced from that metadata so the planner sees the
//! REAL engine, bitemporal flag, and column schema. Collections without
//! persisted metadata (implicit CRDT-only collections) fall through to
//! engine-based detection for backward compatibility.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use nodedb_sql::types::*;
use nodedb_types::collection::CollectionType;
use nodedb_types::columnar::{
    ColumnarProfile, ColumnarSchema, DocumentMode, FloatWidth, IntWidth, StrictSchema,
};
use nodedb_types::sync::wire::CollectionDescriptor;

use crate::engine::columnar::ColumnarEngine;
use crate::engine::crdt::CrdtEngine;
use crate::engine::strict::StrictEngine;
use crate::nodedb::collection::CollectionMeta;
use crate::storage::engine::StorageEngine;

/// Catalog adapter for Lite that resolves collections from local engines.
pub struct LiteCatalog<S: StorageEngine> {
    crdt: Arc<Mutex<CrdtEngine>>,
    strict: Arc<StrictEngine<S>>,
    columnar: Arc<ColumnarEngine<S>>,
    /// Snapshot of persisted collection metadata, keyed by name. Loaded by the
    /// query engine (async) before planning, so `get_collection` can surface
    /// the real engine/bitemporal/columns without touching async storage.
    metas: HashMap<String, CollectionMeta>,
    /// Names of the arrays the array engine holds, snapshotted before
    /// planning. The planner refuses table-shaped statements such as
    /// `TRUNCATE` on an array by name.
    array_names: Vec<String>,
    /// Persisted index specs, keyed by collection name. Loaded by the query
    /// engine (async) before planning, like `metas`.
    index_specs: HashMap<String, Vec<IndexSpec>>,
}

impl<S: StorageEngine> LiteCatalog<S> {
    pub fn new(
        crdt: Arc<Mutex<CrdtEngine>>,
        strict: Arc<StrictEngine<S>>,
        columnar: Arc<ColumnarEngine<S>>,
        metas: HashMap<String, CollectionMeta>,
    ) -> Self {
        Self {
            crdt,
            strict,
            columnar,
            metas,
            array_names: Vec::new(),
            index_specs: HashMap::new(),
        }
    }

    /// Register the array names the planner can see.
    pub fn with_arrays(mut self, array_names: Vec<String>) -> Self {
        self.array_names = array_names;
        self
    }

    /// Register the persisted index specs the planner can see.
    pub fn with_index_specs(mut self, index_specs: HashMap<String, Vec<IndexSpec>>) -> Self {
        self.index_specs = index_specs;
        self
    }

    /// Index specs listed on a collection's `CollectionInfo`. Specs persist
    /// only for schemaless document collections, so every other engine
    /// lists none.
    fn indexes_for(&self, name: &str, engine: EngineType) -> Vec<IndexSpec> {
        if engine != EngineType::DocumentSchemaless {
            return Vec::new();
        }
        self.index_specs.get(name).cloned().unwrap_or_default()
    }

    /// Build a `CollectionInfo` from persisted metadata.
    fn info_from_meta(&self, name: &str, meta: &CollectionMeta) -> CollectionInfo {
        // Prefer the full serialized descriptor when present (synced collections).
        if let Some(dj) = &meta.descriptor_json
            && let Ok(desc) = sonic_rs::from_str::<CollectionDescriptor>(dj)
        {
            return self.info_from_descriptor(name, &desc);
        }
        // Fallback: derive from the flat meta fields (DDL-created collections).
        self.info_from_meta_strings(name, meta)
    }

    /// Resolve columnar-family columns from the live columnar engine schema
    /// when the collection has been materialized (the ingest planner must
    /// encode against the engine's exact schema); otherwise fall back to the
    /// `(name, type_hint)` field hints carried in the descriptor/meta.
    fn columnar_columns_or_fields(
        &self,
        name: &str,
        fields: &[(String, String)],
    ) -> Vec<ColumnInfo> {
        match self.columnar.schema(name) {
            Some(schema) => columns_from_columnar_schema(&schema).0,
            None => columns_from_fields(fields),
        }
    }

    /// Build from a full descriptor. Exhaustive over `CollectionType`.
    fn info_from_descriptor(&self, name: &str, desc: &CollectionDescriptor) -> CollectionInfo {
        let (engine, columns, primary_key) = match &desc.collection_type {
            CollectionType::Document(DocumentMode::Schemaless) => (
                EngineType::DocumentSchemaless,
                columns_from_fields(&desc.fields),
                None,
            ),
            CollectionType::Document(DocumentMode::Strict(schema)) => {
                // Prefer the freshest in-memory schema (post-ALTER) if present.
                let schema = self.strict.schema(name).unwrap_or_else(|| schema.clone());
                let (cols, pk) = columns_from_strict_schema(&schema);
                (EngineType::DocumentStrict, cols, pk)
            }
            // All columnar-family profiles (plain / timeseries / spatial) surface
            // to the SQL planner as `Columnar`: Lite routes SQL INSERTs for these
            // through the columnar DML path, and the columnar engine applies the
            // timeseries/spatial profile internally. The timeseries-specific
            // ingest path is reserved for the ILP/metric API, not SQL rows, so
            // reporting `Timeseries`/`Spatial` here would misroute SQL inserts.
            // Columns come from the live columnar engine's schema when
            // materialized, else the descriptor's field hints (lazy sync case).
            CollectionType::Columnar(
                ColumnarProfile::Plain
                | ColumnarProfile::Timeseries { .. }
                | ColumnarProfile::Spatial { .. },
            ) => {
                let cols = self.columnar_columns_or_fields(name, &desc.fields);
                (EngineType::Columnar, cols, None)
            }
            CollectionType::KeyValue(cfg) => {
                let (cols, pk) = columns_from_strict_schema(&cfg.schema);
                (EngineType::KeyValue, cols, pk)
            }
        };
        CollectionInfo {
            name: name.into(),
            engine,
            columns,
            primary_key,
            has_auto_tier: false,
            indexes: self.indexes_for(name, engine),
            bitemporal: desc.bitemporal,
            primary: desc.primary,
            vector_primary: desc.vector_primary.clone(),
            partition_strategy: desc.partition_strategy.clone(),
            open_schema: CollectionInfo::open_schema_for(engine),
        }
    }

    /// Build from the flat meta strings when no descriptor is stored.
    fn info_from_meta_strings(&self, name: &str, meta: &CollectionMeta) -> CollectionInfo {
        let (engine, columns, primary_key) = match meta.collection_type.as_str() {
            "document_strict" => {
                let schema = self
                    .strict
                    .schema(name)
                    .or_else(|| parse_strict_config(meta.config_json.as_deref()));
                match schema {
                    Some(s) => {
                        let (cols, pk) = columns_from_strict_schema(&s);
                        (EngineType::DocumentStrict, cols, pk)
                    }
                    None => (
                        EngineType::DocumentStrict,
                        columns_from_fields(&meta.fields),
                        None,
                    ),
                }
            }
            // All columnar-family profiles surface to the SQL planner as
            // `Columnar` (profile applied engine-side; see `info_from_descriptor`).
            "columnar" | "timeseries" | "spatial" => (
                EngineType::Columnar,
                self.columnar_columns_or_fields(name, &meta.fields),
                None,
            ),
            "kv" => match parse_kv_config(meta.config_json.as_deref()) {
                Some(cfg) => {
                    let (cols, pk) = columns_from_strict_schema(&cfg.schema);
                    (EngineType::KeyValue, cols, pk)
                }
                None => (
                    EngineType::KeyValue,
                    columns_from_fields(&meta.fields),
                    None,
                ),
            },
            // "document", "document_schemaless", or anything else → schemaless.
            _ => (
                EngineType::DocumentSchemaless,
                columns_from_fields(&meta.fields),
                None,
            ),
        };
        CollectionInfo {
            name: name.into(),
            engine,
            columns,
            primary_key,
            has_auto_tier: false,
            indexes: self.indexes_for(name, engine),
            bitemporal: meta.bitemporal,
            primary: nodedb_types::PrimaryEngine::Document,
            vector_primary: None,
            partition_strategy: nodedb_types::PartitionStrategy::default(),
            open_schema: CollectionInfo::open_schema_for(engine),
        }
    }
}

impl<S: StorageEngine> SqlCatalog for LiteCatalog<S> {
    fn array_exists(&self, name: &str) -> bool {
        self.array_names.iter().any(|n| n == name)
    }

    fn get_collection(
        &self,
        _database_id: nodedb_types::id::DatabaseId,
        name: &str,
    ) -> Result<Option<CollectionInfo>, nodedb_sql::catalog::SqlCatalogError> {
        // Persisted metadata (DDL or synced) is authoritative.
        if let Some(meta) = self.metas.get(name) {
            return Ok(Some(self.info_from_meta(name, meta)));
        }

        // ── Backward-compat fallback: engine-based detection for collections
        // without persisted metadata (e.g. implicit CRDT-only collections). ──

        // Strict collections: surface the real schema, including bitemporal.
        if let Some(schema) = self.strict.schema(name) {
            let (columns, pk) = columns_from_strict_schema(&schema);
            return Ok(Some(CollectionInfo {
                name: name.into(),
                engine: EngineType::DocumentStrict,
                columns,
                primary_key: pk,
                has_auto_tier: false,
                indexes: Vec::new(),
                bitemporal: schema.bitemporal,
                primary: nodedb_types::PrimaryEngine::Document,
                vector_primary: None,
                partition_strategy: nodedb_types::PartitionStrategy::default(),
                open_schema: CollectionInfo::open_schema_for(EngineType::DocumentStrict),
            }));
        }

        // Columnar-family collections surface to the SQL planner as `Columnar`
        // regardless of profile (timeseries/spatial): Lite routes SQL INSERTs
        // through the columnar DML path and the columnar engine applies the
        // profile internally. Columns come from the live engine schema; the
        // bitemporal flag is surfaced from the engine.
        if let Some(schema) = self.columnar.schema(name) {
            let (columns, primary_key) = columns_from_columnar_schema(&schema);
            return Ok(Some(CollectionInfo {
                name: name.into(),
                engine: EngineType::Columnar,
                columns,
                primary_key,
                has_auto_tier: false,
                indexes: Vec::new(),
                bitemporal: self.columnar.is_bitemporal(name),
                primary: nodedb_types::PrimaryEngine::Document,
                vector_primary: None,
                partition_strategy: nodedb_types::PartitionStrategy::default(),
                open_schema: CollectionInfo::open_schema_for(EngineType::Columnar),
            }));
        }

        // CRDT (schemaless) collections: dynamic schema, synthesize an id key.
        if let Ok(crdt) = self.crdt.lock()
            && crdt.collection_names().iter().any(|n| n == name)
        {
            return Ok(Some(CollectionInfo {
                name: name.into(),
                engine: EngineType::DocumentSchemaless,
                columns: vec![ColumnInfo {
                    name: "id".into(),
                    data_type: SqlDataType::String,
                    nullable: false,
                    is_primary_key: true,
                    default: None,
                    raw_type: None,
                    // A synthesized text key is neither an integer nor a float.
                    int_width: None,
                    float_width: None,
                }],
                primary_key: Some("id".into()),
                has_auto_tier: false,
                indexes: self.indexes_for(name, EngineType::DocumentSchemaless),
                bitemporal: false,
                primary: nodedb_types::PrimaryEngine::Document,
                vector_primary: None,
                partition_strategy: nodedb_types::PartitionStrategy::default(),
                open_schema: CollectionInfo::open_schema_for(EngineType::DocumentSchemaless),
            }));
        }

        Ok(None)
    }
}

/// Build column metadata + primary key from a slice of `ColumnDef`.
///
/// Shared by `StrictSchema`/`KvConfig::schema` and `ColumnarSchema`, which
/// both carry `Vec<ColumnDef>` with identical (name, column_type, nullable,
/// default, primary_key) shape.
fn columns_from_column_defs(
    columns: &[nodedb_types::columnar::ColumnDef],
) -> (Vec<ColumnInfo>, Option<String>) {
    let cols = columns
        .iter()
        .map(|c| {
            // Resolve the declared width once here, at the catalog boundary,
            // so the write path (range validation) and the read path (OID and
            // binary payload width) cannot disagree.
            let raw_type = format!("{:?}", c.column_type);
            ColumnInfo {
                name: c.name.clone(),
                data_type: convert_column_type(&c.column_type),
                nullable: c.nullable,
                is_primary_key: c.primary_key,
                default: c.default.clone(),
                int_width: IntWidth::from_declared_type(&raw_type),
                float_width: FloatWidth::from_declared_type(&raw_type),
                raw_type: Some(raw_type),
            }
        })
        .collect();
    let pk = columns
        .iter()
        .find(|c| c.primary_key)
        .map(|c| c.name.clone());
    (cols, pk)
}

/// Build column metadata + primary key from a strict/KV schema.
fn columns_from_strict_schema(schema: &StrictSchema) -> (Vec<ColumnInfo>, Option<String>) {
    columns_from_column_defs(&schema.columns)
}

/// Build column metadata + primary key from a live `ColumnarSchema`.
///
/// This is the schema the columnar engine actually encodes rows against —
/// the timeseries/spatial INSERT planner needs these exact columns, not the
/// descriptor's field hints.
fn columns_from_columnar_schema(schema: &ColumnarSchema) -> (Vec<ColumnInfo>, Option<String>) {
    columns_from_column_defs(&schema.columns)
}

/// Build column metadata from `(name, type_hint)` descriptor field pairs.
fn columns_from_fields(fields: &[(String, String)]) -> Vec<ColumnInfo> {
    fields
        .iter()
        .map(|(fname, type_hint)| {
            let ct = type_hint.parse::<nodedb_types::columnar::ColumnType>().ok();
            let data_type = ct
                .as_ref()
                .map(convert_column_type)
                .unwrap_or(SqlDataType::Bytes);
            ColumnInfo {
                name: fname.clone(),
                data_type,
                nullable: true,
                is_primary_key: false,
                default: None,
                int_width: IntWidth::from_declared_type(type_hint),
                float_width: FloatWidth::from_declared_type(type_hint),
                raw_type: Some(type_hint.clone()),
            }
        })
        .collect()
}

fn parse_strict_config(config_json: Option<&str>) -> Option<StrictSchema> {
    config_json.and_then(|s| sonic_rs::from_str::<StrictSchema>(s).ok())
}

fn parse_kv_config(config_json: Option<&str>) -> Option<nodedb_types::KvConfig> {
    config_json.and_then(|s| sonic_rs::from_str::<nodedb_types::KvConfig>(s).ok())
}

fn convert_column_type(ct: &nodedb_types::columnar::ColumnType) -> SqlDataType {
    use nodedb_types::columnar::ColumnType;
    match ct {
        ColumnType::Int64 => SqlDataType::Int64,
        ColumnType::Float64 => SqlDataType::Float64,
        ColumnType::String => SqlDataType::String,
        ColumnType::Bool => SqlDataType::Bool,
        ColumnType::Bytes | ColumnType::Geometry | ColumnType::Json => SqlDataType::Bytes,
        ColumnType::Timestamp | ColumnType::SystemTimestamp => SqlDataType::Timestamp,
        ColumnType::Decimal { .. } | ColumnType::Uuid | ColumnType::Ulid | ColumnType::Regex => {
            SqlDataType::String
        }
        ColumnType::Duration => SqlDataType::Int64,
        ColumnType::Array | ColumnType::Set | ColumnType::Range | ColumnType::Record => {
            SqlDataType::Bytes
        }
        ColumnType::Vector(dim) => SqlDataType::Vector(*dim as usize),
        _ => SqlDataType::Bytes,
    }
}

#[cfg(test)]
mod tests {
    use nodedb_client::NodeDb;
    use nodedb_types::collection::CollectionType;
    use nodedb_types::collection_config::{PartitionStrategy, PrimaryEngine};
    use nodedb_types::columnar::{ColumnDef, ColumnType, StrictSchema};
    use nodedb_types::document::Document;
    use nodedb_types::id::DatabaseId;
    use nodedb_types::sync::wire::CollectionDescriptor;
    use nodedb_types::value::Value;

    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::{Encryption, PagedbStorageDefault};
    use crate::{NodeDbLite, PagedbStorageMem};

    async fn make_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        NodeDbLite::open(storage).await.unwrap()
    }

    fn descriptor(name: &str, ct: CollectionType, bitemporal: bool) -> CollectionDescriptor {
        CollectionDescriptor {
            tenant_id: 1,
            database_id: DatabaseId::new(1),
            name: name.into(),
            collection_type: ct,
            bitemporal,
            crdt: false,
            fields: vec![("v".into(), "BIGINT".into())],
            primary: PrimaryEngine::Document,
            vector_primary: None,
            partition_strategy: PartitionStrategy::default(),
            declared_primary_key: None,
            descriptor_version: 1,
        }
    }

    async fn catalog_for<S: StorageEngine>(db: &NodeDbLite<S>) -> LiteCatalog<S> {
        let metas =
            crate::nodedb::collection::ddl::load_persisted_collection_metas(db.storage.as_ref())
                .await
                .unwrap();
        let index_specs =
            crate::query::document_ops::index_spec::load_index_specs(db.storage.as_ref())
                .await
                .unwrap();
        LiteCatalog::new(
            Arc::clone(&db.crdt),
            Arc::clone(&db.strict),
            Arc::clone(&db.columnar),
            metas,
        )
        .with_index_specs(index_specs)
    }

    /// Write one schemaless document so the CRDT engine knows `collection`.
    async fn put_scoped_doc<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) {
        let mut doc = Document::new("d1");
        doc.set("scope", Value::String("team-a".into()));
        db.document_put(collection, doc).await.unwrap();
    }

    /// Check that `info` lists exactly the index `CREATE INDEX idx_notes_scope
    /// ON notes (scope)` creates, in the planner's canonical field form.
    fn assert_lists_scope_index(info: &CollectionInfo) {
        assert_eq!(info.engine, EngineType::DocumentSchemaless);
        assert_eq!(info.indexes.len(), 1, "one index spec listed");
        let idx = &info.indexes[0];
        assert_eq!(idx.name, "idx_notes_scope");
        assert_eq!(idx.field, "$.scope", "field in canonical JSON-path form");
        assert!(!idx.unique);
        assert!(!idx.case_insensitive);
        assert!(idx.predicate.is_none());
        // Postings are not maintained on writes yet, so the planner must not
        // rewrite queries to an index lookup.
        assert_eq!(idx.state, IndexState::Building);
    }

    /// `CREATE INDEX` on a schemaless collection lists the index spec on the
    /// collection's `CollectionInfo`. Before specs were persisted the catalog
    /// always listed no indexes, so this assertion failed.
    #[tokio::test]
    async fn catalog_lists_index_after_create_index() {
        let db = make_db().await;
        put_scoped_doc(&db, "notes").await;
        db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
            .await
            .unwrap();

        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "notes")
            .unwrap()
            .expect("collection surfaced");
        assert_lists_scope_index(&info);

        db.execute_sql("DROP INDEX idx_notes_scope ON notes", &[])
            .await
            .unwrap();
        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "notes")
            .unwrap()
            .expect("collection surfaced");
        assert!(info.indexes.is_empty(), "DROP INDEX removes the spec");
    }

    /// A duplicate `CREATE INDEX` on the same field errors before any write,
    /// and `IF NOT EXISTS` keeps the existing spec untouched.
    #[tokio::test]
    async fn duplicate_create_index_is_refused() {
        let db = make_db().await;
        put_scoped_doc(&db, "notes").await;
        db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
            .await
            .unwrap();

        let err = db
            .execute_sql("CREATE INDEX other_name ON notes (scope)", &[])
            .await
            .expect_err("duplicate refused");
        assert!(
            err.to_string()
                .contains("already exists as idx_notes_scope")
        );

        db.execute_sql(
            "CREATE INDEX IF NOT EXISTS other_name ON notes (scope)",
            &[],
        )
        .await
        .unwrap();
        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "notes")
            .unwrap()
            .expect("collection surfaced");
        assert_lists_scope_index(&info);
    }

    /// `DROP COLLECTION` removes the collection's index specs.
    #[tokio::test]
    async fn drop_collection_removes_index_specs() {
        let db = make_db().await;
        db.execute_sql("CREATE COLLECTION notes", &[])
            .await
            .unwrap();
        put_scoped_doc(&db, "notes").await;
        db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
            .await
            .unwrap();
        assert!(
            crate::query::document_ops::index_spec::load_index_specs(db.storage.as_ref())
                .await
                .unwrap()
                .contains_key("notes")
        );

        db.execute_sql("DROP COLLECTION notes", &[]).await.unwrap();

        let specs = crate::query::document_ops::index_spec::load_index_specs(db.storage.as_ref())
            .await
            .unwrap();
        assert!(
            !specs.contains_key("notes"),
            "specs removed with collection"
        );
        let catalog = catalog_for(&db).await;
        let listed = catalog
            .get_collection(DatabaseId::new(1), "notes")
            .unwrap()
            .map(|i| i.indexes)
            .unwrap_or_default();
        assert!(listed.is_empty());
    }

    /// `NodeDbLite::drop_collection` removes the collection's index specs.
    #[tokio::test]
    async fn api_drop_collection_removes_index_specs() {
        let db = make_db().await;
        put_scoped_doc(&db, "notes").await;
        db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
            .await
            .unwrap();

        db.drop_collection("notes").await.unwrap();

        let specs = crate::query::document_ops::index_spec::load_index_specs(db.storage.as_ref())
            .await
            .unwrap();
        assert!(!specs.contains_key("notes"));
    }

    /// A strict collection persists no index spec: specs exist only for
    /// schemaless document collections.
    #[tokio::test]
    async fn strict_collection_persists_no_index_spec() {
        let db = make_db().await;
        db.execute_sql(
            "CREATE COLLECTION s (id INT PRIMARY KEY, scope TEXT) WITH (engine='document_strict')",
            &[],
        )
        .await
        .unwrap();
        // The statement may succeed or be rejected for a strict collection.
        // Either way no spec record is written.
        let _outcome = db
            .execute_sql("CREATE INDEX idx_s_scope ON s (scope)", &[])
            .await;

        let specs = crate::query::document_ops::index_spec::load_index_specs(db.storage.as_ref())
            .await
            .unwrap();
        assert!(!specs.contains_key("s"));
        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "s")
            .unwrap()
            .expect("collection surfaced");
        assert_eq!(info.engine, EngineType::DocumentStrict);
        assert!(info.indexes.is_empty());
    }

    /// The index spec is durable: a reopened database lists it again.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn index_spec_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index_spec.db");

        {
            let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
                .await
                .unwrap();
            let db = NodeDbLite::open(storage).await.unwrap();
            // The DDL persists collection meta, so the reopened catalog
            // resolves `notes` without depending on CRDT state replay.
            db.execute_sql("CREATE COLLECTION notes", &[])
                .await
                .unwrap();
            put_scoped_doc(&db, "notes").await;
            db.execute_sql("CREATE INDEX idx_notes_scope ON notes (scope)", &[])
                .await
                .unwrap();
            db.flush().await.unwrap();
        }

        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .unwrap();
        let db = NodeDbLite::open(storage).await.unwrap();
        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "notes")
            .unwrap()
            .expect("collection surfaced after reopen");
        assert_lists_scope_index(&info);
    }

    /// lite#3 repro: a synced strict collection must surface its REAL engine,
    /// bitemporal flag, and columns from persisted metadata — not the old
    /// hardcoded `bitemporal: false` / fake schema. This assertion FAILS
    /// against the pre-fix catalog because that path always returned
    /// `bitemporal: false`.
    #[tokio::test]
    async fn synced_strict_surfaces_real_engine_and_bitemporal() {
        let db = make_db().await;
        let schema = StrictSchema::new(vec![
            ColumnDef::required("id", ColumnType::Int64).with_primary_key(),
            ColumnDef::nullable("name", ColumnType::String),
        ])
        .unwrap();
        let desc = descriptor("s", CollectionType::strict(schema), true);
        db.register_collection_from_descriptor(&desc).await.unwrap();

        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "s")
            .unwrap()
            .expect("collection surfaced");

        assert_eq!(info.engine, EngineType::DocumentStrict);
        assert!(
            info.bitemporal,
            "lite#3: bitemporal must be REAL, not hardcoded false"
        );
        assert_eq!(info.primary_key.as_deref(), Some("id"));
        assert_eq!(info.columns.len(), 2);
        assert_eq!(info.columns[0].name, "id");
    }

    /// A synced timeseries collection surfaces to the SQL planner as `Columnar`
    /// (Lite plans all columnar-family via the columnar path; the profile lives
    /// engine-side) and correctly carries the bitemporal flag from the meta —
    /// even though it is created lazily engine-side (no in-memory schema yet).
    #[tokio::test]
    async fn synced_timeseries_surfaces_engine_from_meta() {
        let db = make_db().await;
        let desc = descriptor("t", CollectionType::timeseries("ts", "1h"), true);
        db.register_collection_from_descriptor(&desc).await.unwrap();

        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "t")
            .unwrap()
            .expect("collection surfaced");

        assert_eq!(info.engine, EngineType::Columnar);
        assert!(info.bitemporal);
    }

    /// Regression repro: a locally-created timeseries collection (via
    /// `columnar.create_collection`, the path used by `query/ddl/timeseries.rs`)
    /// has NO persisted `CollectionMeta` — it hits the fallback branch. The
    /// fallback must surface the `Columnar` engine (Lite plans columnar-family
    /// SQL via the columnar path) WITH the engine's real columns, not empty
    /// columns (empty columns broke row resolution / SELECT).
    #[tokio::test]
    async fn fallback_timeseries_surfaces_real_columns() {
        let db = make_db().await;
        let schema = nodedb_types::columnar::ColumnarSchema {
            columns: vec![
                ColumnDef::required("time", ColumnType::Timestamp),
                ColumnDef::nullable("value", ColumnType::Float64),
            ],
            version: 1,
        };
        db.columnar
            .create_collection(
                "metrics",
                schema,
                nodedb_types::columnar::ColumnarProfile::Timeseries {
                    time_key: "time".into(),
                    interval: "1h".into(),
                },
                false,
            )
            .await
            .unwrap();

        // No persisted meta for this collection — confirms the fallback path.
        let metas =
            crate::nodedb::collection::ddl::load_persisted_collection_metas(db.storage.as_ref())
                .await
                .unwrap();
        assert!(!metas.contains_key("metrics"));

        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "metrics")
            .unwrap()
            .expect("collection surfaced");

        assert_eq!(info.engine, EngineType::Columnar);
        assert_eq!(
            info.columns.len(),
            2,
            "timeseries fallback must surface real engine columns, not empty"
        );
        assert_eq!(info.columns[0].name, "time");
        assert_eq!(info.columns[1].name, "value");
    }

    /// A synced KV collection surfaces the KeyValue engine with real columns.
    #[tokio::test]
    async fn synced_kv_surfaces_engine_and_columns() {
        let db = make_db().await;
        let schema = StrictSchema::new(vec![
            ColumnDef::required("k", ColumnType::String).with_primary_key(),
            ColumnDef::nullable("val", ColumnType::Bytes),
        ])
        .unwrap();
        let desc = descriptor("kvc", CollectionType::kv(schema), false);
        db.register_collection_from_descriptor(&desc).await.unwrap();

        let catalog = catalog_for(&db).await;
        let info = catalog
            .get_collection(DatabaseId::new(1), "kvc")
            .unwrap()
            .expect("collection surfaced");

        assert_eq!(info.engine, EngineType::KeyValue);
        assert_eq!(info.primary_key.as_deref(), Some("k"));
        assert_eq!(info.columns.len(), 2);
    }
}
