// SPDX-License-Identifier: Apache-2.0

//! Publish PageDB snapshot files and required Lite encryption metadata.

use std::io::Write;
use std::path::Path;

use crate::error::LiteError;
use crate::storage::pagedb_storage::PagedbStorageDefault;

use super::metadata::METADATA_FILE;
use super::ownership::{OwnedDirectory, io_error, sync_parent};

impl PagedbStorageDefault {
    pub(crate) async fn export_snapshot(&self, destination: &Path) -> Result<(), LiteError> {
        let descriptor = self
            .snapshot_descriptor
            .as_ref()
            .ok_or_else(|| LiteError::Storage {
                detail: "native storage lacks snapshot encryption metadata".into(),
            })?;
        let mut ownership = OwnedDirectory::claim(destination, false)?;
        let result = async {
            self.db
                .snapshot_to(destination)
                .await
                .map_err(LiteError::from)?;
            let bytes = descriptor.encode()?;
            let metadata = destination.join(METADATA_FILE);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&metadata)
                .map_err(|error| io_error(&metadata, error))?;
            file.write_all(&bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| io_error(&metadata, error))?;
            sync_parent(&metadata)?;
            sync_parent(destination)?;
            Ok(())
        }
        .await;
        if let Err(primary) = result {
            return Err(ownership.cleanup_error(primary));
        }
        ownership.preserve();
        Ok(())
    }
}
