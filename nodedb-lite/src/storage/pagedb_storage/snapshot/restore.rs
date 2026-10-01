// SPDX-License-Identifier: Apache-2.0

//! Authenticate a read-only snapshot and fork a fresh writable PageDB identity.

use std::path::Path;

use pagedb::Db;
use pagedb::vfs::tokio_backend::TokioVfs;
use zeroize::Zeroizing;

use crate::error::LiteError;
use crate::storage::encryption::Encryption;
use crate::storage::pagedb_storage::PagedbStorageDefault;
use crate::storage::pagedb_storage::types::lite_open_options;

use super::metadata::SnapshotDescriptor;
use super::ownership::OwnedDirectory;

impl PagedbStorageDefault {
    pub(crate) async fn restore_snapshot(
        source: &Path,
        destination: &Path,
        encryption: Encryption,
    ) -> Result<Self, LiteError> {
        let descriptor = SnapshotDescriptor::read(source)?;
        let source_key = Zeroizing::new(descriptor.source_key(&encryption)?);
        let mut ownership = OwnedDirectory::claim(destination, true)?;
        let result = async {
            let restored =
                Db::<TokioVfs>::restore_from(source, destination, lite_open_options(), *source_key)
                    .await
                    .map_err(LiteError::from)?;
            let destination_key = Zeroizing::new(match &encryption {
                Encryption::Passphrase {
                    passphrase,
                    m_cost,
                    t_cost,
                    p_cost,
                } => {
                    let salt = ownership.create_salt()?;
                    crate::storage::encryption::derive_key(
                        passphrase, &salt, *m_cost, *t_cost, *p_cost,
                    )?
                }
                Encryption::Plaintext => [0; 32],
                Encryption::RawKey(key) => *key,
            });
            let writer = restored
                .rekey_into_writer(*destination_key)
                .await
                .map_err(LiteError::from)?;
            ownership.preserve();
            drop(writer);
            Self::open(destination, encryption)
                .await
                .map_err(|error| LiteError::Storage {
                    detail: format!(
                        "snapshot writable fork remains at {} but storage reopen returned: {error}",
                        destination.display()
                    ),
                })
        }
        .await;
        result.map_err(|primary| ownership.cleanup_error(primary))
    }
}
