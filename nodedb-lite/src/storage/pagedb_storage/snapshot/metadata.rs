// SPDX-License-Identifier: Apache-2.0

//! Versioned nonsecret encryption metadata for portable snapshots.

use std::io::Read;
use std::path::Path;

use crate::error::LiteError;
use crate::storage::encryption::{Encryption, derive_key, load_or_create_salt};

pub(super) const METADATA_FILE: &str = "lite-snapshot.msgpack";
const MAX_METADATA_BYTES: u64 = 1024;

#[derive(Clone, Copy, PartialEq, Eq, zerompk::ToMessagePack, zerompk::FromMessagePack)]
#[msgpack(c_enum)]
pub(super) enum EncryptionMode {
    Plaintext,
    RawKey,
    Passphrase,
}

#[derive(Clone, zerompk::ToMessagePack, zerompk::FromMessagePack)]
pub(crate) struct SnapshotDescriptor {
    version: u32,
    mode: EncryptionMode,
    salt: [u8; 16],
    m_cost: u32,
    t_cost: u32,
    p_cost: u32,
}

impl SnapshotDescriptor {
    pub(crate) fn from_native(encryption: &Encryption, path: &Path) -> Result<Self, LiteError> {
        let salt = match encryption {
            Encryption::Passphrase { .. } => load_or_create_salt(path)?,
            _ => [0; 16],
        };
        Ok(Self::with_salt(encryption, salt))
    }

    pub(super) fn with_salt(encryption: &Encryption, salt: [u8; 16]) -> Self {
        let (mode, m_cost, t_cost, p_cost) = match encryption {
            Encryption::Plaintext => (EncryptionMode::Plaintext, 0, 0, 0),
            Encryption::RawKey(_) => (EncryptionMode::RawKey, 0, 0, 0),
            Encryption::Passphrase {
                m_cost,
                t_cost,
                p_cost,
                ..
            } => (EncryptionMode::Passphrase, *m_cost, *t_cost, *p_cost),
        };
        Self {
            version: 1,
            mode,
            salt,
            m_cost,
            t_cost,
            p_cost,
        }
    }

    pub(super) fn encode(&self) -> Result<Vec<u8>, LiteError> {
        zerompk::to_msgpack_vec(self).map_err(|error| LiteError::Serialization {
            detail: format!("snapshot metadata encode: {error}"),
        })
    }

    pub(super) fn read(source: &Path) -> Result<Self, LiteError> {
        let path = source.join(METADATA_FILE);
        let file = std::fs::File::open(&path).map_err(|error| metadata_error(&path, error))?;
        let mut bytes = Vec::new();
        file.take(MAX_METADATA_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| metadata_error(&path, error))?;
        if bytes.len() as u64 > MAX_METADATA_BYTES {
            return Err(metadata_error(&path, "metadata exceeds 1024 bytes"));
        }
        let descriptor: Self =
            zerompk::from_msgpack(&bytes).map_err(|error| metadata_error(&path, error))?;
        if descriptor.version != 1 {
            return Err(metadata_error(
                &path,
                format!("unsupported version {}", descriptor.version),
            ));
        }
        if descriptor.encode()? != bytes {
            return Err(metadata_error(&path, "malformed metadata shape"));
        }
        if descriptor.mode != EncryptionMode::Passphrase
            && (descriptor.salt != [0; 16]
                || descriptor.m_cost != 0
                || descriptor.t_cost != 0
                || descriptor.p_cost != 0)
        {
            return Err(metadata_error(
                &path,
                "unexpected salt or KDF costs for encryption mode",
            ));
        }
        Ok(descriptor)
    }

    pub(super) fn source_key(&self, encryption: &Encryption) -> Result<[u8; 32], LiteError> {
        match (self.mode, encryption) {
            (EncryptionMode::Plaintext, Encryption::Plaintext) => Ok([0; 32]),
            (EncryptionMode::RawKey, Encryption::RawKey(key)) => Ok(*key),
            (
                EncryptionMode::Passphrase,
                Encryption::Passphrase {
                    passphrase,
                    m_cost,
                    t_cost,
                    p_cost,
                },
            ) => {
                if (*m_cost, *t_cost, *p_cost) != (self.m_cost, self.t_cost, self.p_cost) {
                    return Err(LiteError::Encryption {
                        detail: "snapshot KDF parameters differ from supplied encryption config"
                            .into(),
                    });
                }
                derive_key(passphrase, &self.salt, *m_cost, *t_cost, *p_cost)
            }
            _ => Err(LiteError::Encryption {
                detail: "snapshot encryption mode differs from supplied encryption config".into(),
            }),
        }
    }
}

fn metadata_error(path: &Path, error: impl std::fmt::Display) -> LiteError {
    LiteError::Corrupted {
        detail: format!("snapshot metadata {}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_rejects_versions_shapes_and_supplied_cost_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(METADATA_FILE);
        let mut descriptor = SnapshotDescriptor::with_salt(&Encryption::Plaintext, [0; 16]);
        descriptor.version = 2;
        std::fs::write(&path, descriptor.encode().unwrap()).unwrap();
        assert!(SnapshotDescriptor::read(dir.path()).is_err());
        std::fs::write(&path, [0xff]).unwrap();
        assert!(SnapshotDescriptor::read(dir.path()).is_err());
        std::fs::write(&path, vec![0; 1025]).unwrap();
        assert!(SnapshotDescriptor::read(dir.path()).is_err());
        let encryption = Encryption::Passphrase {
            passphrase: "secret".into(),
            m_cost: 8,
            t_cost: 1,
            p_cost: 1,
        };
        let descriptor = SnapshotDescriptor::with_salt(&encryption, [1; 16]);
        let other = Encryption::Passphrase {
            passphrase: "secret".into(),
            m_cost: 16,
            t_cost: 1,
            p_cost: 1,
        };
        assert!(descriptor.source_key(&other).is_err());
        assert!(descriptor.source_key(&Encryption::Plaintext).is_err());
    }
}
