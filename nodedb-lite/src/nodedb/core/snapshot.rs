// SPDX-License-Identifier: Apache-2.0

//! Native snapshot export and writable restore APIs.

use std::path::Path;
use std::sync::Arc;

use nodedb_types::error::{NodeDbError, NodeDbResult};

use crate::config::LiteConfig;
use crate::storage::encryption::Encryption;
use crate::storage::pagedb_storage::PagedbStorageDefault;

use super::types::NodeDbLite;

impl NodeDbLite<PagedbStorageDefault> {
    /// Flush pending state and export a native snapshot to a new directory.
    ///
    /// Quiesce concurrent writes for one application-level point in time.
    /// PageDB pins its published state and blocks storage writes during export.
    /// This API is unavailable on WASM.
    pub async fn snapshot_to(&self, path: impl AsRef<Path>) -> NodeDbResult<()> {
        self.flush().await?;
        self.storage
            .export_snapshot(path.as_ref())
            .await
            .map_err(NodeDbError::from)
    }

    /// Restore a native snapshot into a new writable destination and retain Lite replica identity.
    ///
    /// Supply the source encryption mode and KDF costs. Passphrase restores create a fresh destination salt.
    /// PageDB forks its storage identity through rekeying. Restore always uses fail-closed corruption handling.
    /// This API is unavailable on WASM.
    pub async fn restore_from(
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
        encryption: Encryption,
        mut config: LiteConfig,
    ) -> NodeDbResult<Arc<Self>> {
        config.corruption_policy = crate::storage::corruption::CorruptionPolicy::FailClosed;
        let destination = destination.as_ref();
        let storage =
            PagedbStorageDefault::restore_snapshot(source.as_ref(), destination, encryption)
                .await
                .map_err(NodeDbError::from)?;
        Self::open_with_config(storage, config)
            .await
            .map_err(|error| {
                NodeDbError::storage(format!(
                    "snapshot writable fork remains at {} but Lite reopen returned: {error}",
                    destination.display()
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodedb_client::NodeDb;

    #[tokio::test]
    async fn incomplete_vector_state_stops_snapshot_before_destination_creation() {
        let dir = tempfile::tempdir().unwrap();
        let db = NodeDbLite::open_at_path_with_config(
            dir.path().join("source"),
            Encryption::Plaintext,
            LiteConfig {
                auto_flush_ms: 0,
                sync_enabled: false,
                ..LiteConfig::default()
            },
        )
        .await
        .unwrap();
        db.vector_insert("vectors", "v1", &[1.0, 0.0], None)
            .await
            .unwrap();
        crate::engine::vector::durable::remove(&*db.storage, "vectors", "v1")
            .await
            .unwrap();
        let destination = dir.path().join("snapshot");
        assert!(db.snapshot_to(&destination).await.is_err());
        assert!(!destination.exists());
    }
}
