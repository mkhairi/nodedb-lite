// SPDX-License-Identifier: Apache-2.0

//! Vector delete on a collection declared vector-primary: the row exists
//! for its vector, so it goes with the last vector even when it carries
//! payload fields.

use nodedb_client::NodeDb;
use nodedb_types::collection::CollectionType;
use nodedb_types::collection_config::{PartitionStrategy, PrimaryEngine};
use nodedb_types::document::Document;
use nodedb_types::id::DatabaseId;
use nodedb_types::sync::wire::CollectionDescriptor;
use nodedb_types::value::Value;

use crate::PagedbStorageMem;
use crate::engine::vector::row::is_vector_primary;
use crate::nodedb::NodeDbLite;
use crate::nodedb::collection::CollectionMeta;
use crate::storage::engine::StorageEngine;

async fn open_db() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory()
        .await
        .expect("in-memory storage");
    NodeDbLite::open(storage).await.expect("open")
}

/// Persist `name`'s collection meta with a vector-primary descriptor, as an
/// inbound schema announcement does.
async fn declare_vector_primary(db: &NodeDbLite<PagedbStorageMem>, name: &str) {
    let descriptor = CollectionDescriptor {
        tenant_id: 1,
        database_id: DatabaseId::DEFAULT,
        name: name.into(),
        collection_type: CollectionType::document(),
        bitemporal: false,
        crdt: false,
        fields: Vec::new(),
        primary: PrimaryEngine::Vector,
        vector_primary: None,
        partition_strategy: PartitionStrategy::default(),
        declared_primary_key: None,
        descriptor_version: 1,
    };
    let meta = CollectionMeta {
        name: name.into(),
        collection_type: "document".into(),
        created_at_ms: 0,
        fields: Vec::new(),
        config_json: None,
        descriptor_json: Some(sonic_rs::to_string(&descriptor).expect("descriptor json")),
        bitemporal: false,
        crdt: false,
    };
    db.storage
        .put(
            nodedb_types::Namespace::Meta,
            format!("collection:{name}").as_bytes(),
            &sonic_rs::to_vec(&meta).expect("meta json"),
        )
        .await
        .expect("put meta");
}

#[tokio::test]
async fn vector_delete_on_vector_collection_removes_row() {
    let db = open_db().await;
    declare_vector_primary(&db, "vp").await;
    assert!(
        is_vector_primary(&*db.storage, "vp")
            .await
            .expect("read meta")
    );

    let mut payload = Document::new("v1");
    payload.set("title", Value::String("hello".into()));
    db.vector_insert("vp", "v1", &[1.0, 0.0], Some(payload))
        .await
        .expect("vector_insert");
    db.vector_delete("vp", "v1").await.expect("vector_delete");

    assert!(
        db.document_get("vp", "v1")
            .await
            .expect("document_get")
            .is_none(),
        "a vector-primary row goes with its vector"
    );
}

#[tokio::test]
async fn undeclared_collection_is_not_vector_primary() {
    let db = open_db().await;
    assert!(
        !is_vector_primary(&*db.storage, "plain")
            .await
            .expect("read meta")
    );
}
