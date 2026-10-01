// SPDX-License-Identifier: Apache-2.0

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nodedb_client::NodeDb;
use nodedb_lite::{CorruptionPolicy, Encryption, LiteConfig, NodeDbLite, PagedbStorageDefault};
use nodedb_types::graph::Direction;
use nodedb_types::text_search::TextSearchParams;
use nodedb_types::{Document, NodeId, Value};

type Database = NodeDbLite<PagedbStorageDefault>;

fn config() -> LiteConfig {
    LiteConfig {
        auto_flush_ms: 0,
        auto_compact_ms: 0,
        sync_enabled: false,
        ..LiteConfig::default()
    }
}

fn passphrase(secret: &str) -> Encryption {
    Encryption::Passphrase {
        passphrase: secret.into(),
        m_cost: 8,
        t_cost: 1,
        p_cost: 1,
    }
}

fn salt_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".salt");
    PathBuf::from(name)
}

async fn open(path: &Path, encryption: Encryption) -> Arc<Database> {
    Database::open_at_path_with_config(path, encryption, config())
        .await
        .unwrap()
}

async fn pending_state(db: &Database) {
    let mut document = Document::new("d1");
    document.set("title", Value::String("portable snapshot".into()));
    db.document_put("docs", document).await.unwrap();
    db.kv_put("settings", "pending", b"value").await.unwrap();
    db.graph_insert_edge(
        "links",
        &NodeId::try_new("a").unwrap(),
        &NodeId::try_new("b").unwrap(),
        "LINK",
        None,
    )
    .await
    .unwrap();
    db.vector_insert("vectors", "v1", &[1.0, 0.0], None)
        .await
        .unwrap();
}

async fn check_pending_state(db: &Database) {
    assert_eq!(
        db.document_get("docs", "d1")
            .await
            .unwrap()
            .unwrap()
            .get_str("title"),
        Some("portable snapshot")
    );
    assert_eq!(
        db.kv_get("settings", "pending").await.unwrap(),
        Some(b"value".to_vec())
    );
    let graph = db
        .graph_traverse(
            "links",
            &NodeId::try_new("a").unwrap(),
            1,
            Direction::Out,
            None,
        )
        .await
        .unwrap();
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.edges[0].to.as_str(), "b");
    let vectors = db
        .vector_search("vectors", &[1.0, 0.0], 1, None, None)
        .await
        .unwrap();
    assert_eq!(vectors[0].id, "v1");
    let text = db
        .text_search(
            "docs",
            "title",
            "portable",
            1,
            TextSearchParams::default(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(text[0].id, "d1");
}

#[tokio::test]
async fn snapshot_flushes_pending_engines_and_restores_writable_replicas() {
    for encryption in [
        Encryption::Plaintext,
        Encryption::RawKey([0x42; 32]),
        passphrase("correct"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let snapshot = dir.path().join("snapshot");
        let destination = dir.path().join("restored");
        let db = open(&source, encryption.clone()).await;
        pending_state(&db).await;
        let peer = db.peer_id();
        db.snapshot_to(&snapshot).await.unwrap();
        let restored =
            Database::restore_from(&snapshot, &destination, encryption.clone(), config())
                .await
                .unwrap();
        assert_eq!(restored.peer_id(), peer);
        check_pending_state(&restored).await;
        if matches!(encryption, Encryption::Passphrase { .. }) {
            let source_salt = std::fs::read(salt_path(&source)).unwrap();
            let destination_salt = std::fs::read(salt_path(&destination)).unwrap();
            assert_eq!(destination_salt.len(), 16);
            assert_ne!(source_salt, destination_salt);
        }
        restored
            .kv_put("settings", "after", b"writable")
            .await
            .unwrap();
        restored
            .document_put("docs", Document::new("after"))
            .await
            .unwrap();
        restored.flush().await.unwrap();
        restored.shutdown().await;
        drop(restored);
        let reopened = open(&destination, encryption).await;
        assert_eq!(reopened.peer_id(), peer);
        check_pending_state(&reopened).await;
        assert_eq!(
            reopened.kv_get("settings", "after").await.unwrap(),
            Some(b"writable".to_vec())
        );
        assert!(
            reopened
                .document_get("docs", "after")
                .await
                .unwrap()
                .is_some()
        );
        reopened.shutdown().await;
        db.shutdown().await;
    }
}

#[tokio::test]
async fn restore_rejects_wrong_credentials_and_kdf_parameters_without_artifacts() {
    for (encryption, wrong) in [
        (Encryption::RawKey([1; 32]), Encryption::RawKey([2; 32])),
        (passphrase("correct"), passphrase("incorrect")),
        (
            passphrase("correct"),
            Encryption::Passphrase {
                passphrase: "correct".into(),
                m_cost: 16,
                t_cost: 1,
                p_cost: 1,
            },
        ),
        (Encryption::Plaintext, Encryption::RawKey([0; 32])),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let snapshot = dir.path().join("snapshot");
        let destination = dir.path().join("restored");
        let db = open(&source, encryption).await;
        db.kv_put("settings", "keep", b"source").await.unwrap();
        db.snapshot_to(&snapshot).await.unwrap();
        assert!(
            Database::restore_from(&snapshot, &destination, wrong, config())
                .await
                .is_err()
        );
        assert!(!destination.exists());
        assert!(!salt_path(&destination).exists());
        assert_eq!(
            db.kv_get("settings", "keep").await.unwrap(),
            Some(b"source".to_vec())
        );
        db.shutdown().await;
    }
}

#[tokio::test]
async fn malformed_metadata_and_corrupt_pages_never_trigger_discard_policy() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("source"), Encryption::Plaintext).await;
    let snapshot = dir.path().join("snapshot");
    let destination = dir.path().join("restored");
    db.snapshot_to(&snapshot).await.unwrap();
    let metadata = snapshot.join("lite-snapshot.msgpack");
    let original = std::fs::read(&metadata).unwrap();
    let discard = LiteConfig {
        corruption_policy: CorruptionPolicy::DiscardStoreAndRecreate,
        ..config()
    };
    for bytes in [vec![0xff], vec![0; 1025]] {
        std::fs::write(&metadata, bytes).unwrap();
        assert!(
            Database::restore_from(
                &snapshot,
                &destination,
                Encryption::Plaintext,
                discard.clone()
            )
            .await
            .is_err()
        );
        assert!(!destination.exists());
    }
    std::fs::remove_file(&metadata).unwrap();
    assert!(
        Database::restore_from(
            &snapshot,
            &destination,
            Encryption::Plaintext,
            discard.clone()
        )
        .await
        .is_err()
    );
    std::fs::write(&metadata, original).unwrap();
    std::fs::write(snapshot.join("main.db"), b"invalid").unwrap();
    assert!(
        Database::restore_from(&snapshot, &destination, Encryption::Plaintext, discard)
            .await
            .is_err()
    );
    assert!(!destination.exists());
    assert!(snapshot.join("main.db").exists());
    db.shutdown().await;
}

#[tokio::test]
async fn existing_export_restore_and_salt_destinations_remain_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("source"), Encryption::Plaintext).await;
    let snapshot = dir.path().join("snapshot");
    let existing = dir.path().join("existing");
    std::fs::create_dir(&existing).unwrap();
    std::fs::write(existing.join("keep"), b"keep").unwrap();
    assert!(db.snapshot_to(&existing).await.is_err());
    assert_eq!(std::fs::read(existing.join("keep")).unwrap(), b"keep");
    db.snapshot_to(&snapshot).await.unwrap();
    assert!(
        Database::restore_from(&snapshot, &existing, Encryption::Plaintext, config())
            .await
            .is_err()
    );
    assert_eq!(std::fs::read(existing.join("keep")).unwrap(), b"keep");
    let destination = dir.path().join("restored");
    std::fs::write(salt_path(&destination), b"keep").unwrap();
    assert!(
        Database::restore_from(&snapshot, &destination, Encryption::Plaintext, config())
            .await
            .is_err()
    );
    assert_eq!(std::fs::read(salt_path(&destination)).unwrap(), b"keep");
    assert!(!destination.exists());
    db.shutdown().await;
}

#[cfg(unix)]
#[tokio::test]
async fn dangling_destination_symlinks_remain_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("source"), Encryption::Plaintext).await;
    let snapshot = dir.path().join("snapshot");
    db.snapshot_to(&snapshot).await.unwrap();
    let destination = dir.path().join("restored");
    std::os::unix::fs::symlink(dir.path().join("absent"), &destination).unwrap();
    assert!(
        Database::restore_from(&snapshot, &destination, Encryption::Plaintext, config())
            .await
            .is_err()
    );
    assert!(
        std::fs::symlink_metadata(&destination)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    std::fs::remove_file(&destination).unwrap();
    let salt = salt_path(&destination);
    std::os::unix::fs::symlink(dir.path().join("absent"), &salt).unwrap();
    assert!(
        Database::restore_from(&snapshot, &destination, Encryption::Plaintext, config())
            .await
            .is_err()
    );
    assert!(
        std::fs::symlink_metadata(&salt)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!destination.exists());
    db.shutdown().await;
}

#[tokio::test]
async fn failed_lite_reopen_preserves_the_writable_fork_and_fresh_salt() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("source"), passphrase("correct")).await;
    db.kv_put("settings", "keep", b"value").await.unwrap();
    let snapshot = dir.path().join("snapshot");
    let destination = dir.path().join("restored");
    db.snapshot_to(&snapshot).await.unwrap();
    let invalid = LiteConfig {
        hnsw_percent: 101,
        ..config()
    };
    assert!(
        Database::restore_from(&snapshot, &destination, passphrase("correct"), invalid)
            .await
            .is_err()
    );
    assert!(destination.join("main.db").exists());
    assert_eq!(std::fs::read(salt_path(&destination)).unwrap().len(), 16);
    let restored = open(&destination, passphrase("correct")).await;
    assert_eq!(
        restored.kv_get("settings", "keep").await.unwrap(),
        Some(b"value".to_vec())
    );
    restored.shutdown().await;
    db.shutdown().await;
}

#[tokio::test]
async fn concurrent_source_writes_leave_export_readable_and_source_writable() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("source"), Encryption::Plaintext).await;
    db.kv_put("settings", "before", b"baseline").await.unwrap();
    let writer_db = Arc::clone(&db);
    let writer = tokio::spawn(async move {
        for value in 0..8 {
            writer_db
                .kv_put("settings", "during", value.to_string().as_bytes())
                .await
                .unwrap();
            writer_db.flush().await.unwrap();
            tokio::task::yield_now().await;
        }
    });
    let snapshot = dir.path().join("snapshot");
    db.snapshot_to(&snapshot).await.unwrap();
    writer.await.unwrap();
    let restored = Database::restore_from(
        &snapshot,
        dir.path().join("restored"),
        Encryption::Plaintext,
        config(),
    )
    .await
    .unwrap();
    assert_eq!(
        restored.kv_get("settings", "before").await.unwrap(),
        Some(b"baseline".to_vec())
    );
    assert_eq!(
        db.kv_get("settings", "during").await.unwrap(),
        Some(b"7".to_vec())
    );
    restored
        .kv_put("settings", "after", b"restored")
        .await
        .unwrap();
    restored.flush().await.unwrap();
    restored.shutdown().await;
    db.shutdown().await;
}
