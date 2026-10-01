// SPDX-License-Identifier: Apache-2.0

//! Exclusive destination ownership and cleanup before snapshot publication.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::LiteError;
use crate::storage::encryption::salt_sidecar_path;

pub(super) struct OwnedDirectory {
    path: PathBuf,
    identity: std::fs::Metadata,
    salt_identity: Option<std::fs::Metadata>,
    keep: bool,
}

impl OwnedDirectory {
    pub(super) fn claim(path: &Path, reject_salt: bool) -> Result<Self, LiteError> {
        if reject_salt {
            let salt = salt_sidecar_path(path);
            match std::fs::symlink_metadata(&salt) {
                Ok(_) => return Err(io_error(&salt, "salt destination already exists")),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error(&salt, error)),
            }
        }
        std::fs::create_dir(path).map_err(|error| io_error(path, error))?;
        let identity = std::fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
        Ok(Self {
            path: path.to_path_buf(),
            identity,
            salt_identity: None,
            keep: false,
        })
    }

    pub(super) fn create_salt(&mut self) -> Result<[u8; 16], LiteError> {
        let salt_path = salt_sidecar_path(&self.path);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&salt_path)
            .map_err(|error| io_error(&salt_path, error))?;
        self.salt_identity = Some(
            file.metadata()
                .map_err(|error| io_error(&salt_path, error))?,
        );
        let mut salt = [0; 16];
        getrandom::fill(&mut salt).map_err(|error| LiteError::Encryption {
            detail: format!("snapshot destination salt generation: {error}"),
        })?;
        file.write_all(&salt)
            .and_then(|()| file.sync_all())
            .map_err(|error| io_error(&salt_path, error))?;
        sync_parent(&salt_path)?;
        Ok(salt)
    }

    pub(super) fn preserve(&mut self) {
        self.keep = true;
    }

    fn cleanup(&mut self) -> Result<(), LiteError> {
        if self.keep {
            return Ok(());
        }
        let mut errors = Vec::new();
        match std::fs::symlink_metadata(&self.path) {
            Ok(current)
                if current.is_dir()
                    && !current.file_type().is_symlink()
                    && same_identity(&current, &self.identity) =>
            {
                if let Err(error) = std::fs::remove_dir_all(&self.path) {
                    errors.push(format!("{}: {error}", self.path.display()));
                }
            }
            Ok(_) => errors.push(format!(
                "{}: destination ownership changed",
                self.path.display()
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => errors.push(format!("{}: {error}", self.path.display())),
        }
        if let Some(identity) = &self.salt_identity {
            let salt = salt_sidecar_path(&self.path);
            match std::fs::symlink_metadata(&salt) {
                Ok(current) if same_identity(&current, identity) => {
                    if let Err(error) = std::fs::remove_file(&salt) {
                        errors.push(format!("{}: {error}", salt.display()));
                    }
                }
                Ok(_) => errors.push(format!("{}: salt ownership changed", salt.display())),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => errors.push(format!("{}: {error}", salt.display())),
            }
        }
        if errors.is_empty() {
            self.keep = true;
            Ok(())
        } else {
            Err(io_error(&self.path, errors.join(", ")))
        }
    }

    pub(super) fn cleanup_error(&mut self, primary: LiteError) -> LiteError {
        let Err(cleanup) = self.cleanup() else {
            return primary;
        };
        let context = format!(". Cleanup also returned: {cleanup}");
        match primary {
            LiteError::Storage { detail } => LiteError::Storage {
                detail: detail + &context,
            },
            LiteError::Corrupted { detail } => LiteError::Corrupted {
                detail: detail + &context,
            },
            LiteError::Encryption { detail } => LiteError::Encryption {
                detail: detail + &context,
            },
            LiteError::Serialization { detail } => LiteError::Serialization {
                detail: detail + &context,
            },
            other => {
                tracing::error!(%cleanup, "snapshot cleanup error after primary error");
                other
            }
        }
    }
}

impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            tracing::error!(path = %self.path.display(), %error, "snapshot destination cleanup error");
        }
    }
}

fn same_identity(current: &std::fs::Metadata, original: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        current.dev() == original.dev() && current.ino() == original.ino()
    }
    #[cfg(not(unix))]
    {
        match (current.created(), original.created()) {
            (Ok(current), Ok(original)) => current == original,
            _ => false,
        }
    }
}

pub(super) fn sync_parent(path: &Path) -> Result<(), LiteError> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|error| io_error(parent, error))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

pub(super) fn io_error(path: &Path, error: impl std::fmt::Display) -> LiteError {
    LiteError::Storage {
        detail: format!("snapshot path {}: {error}", path.display()),
    }
}
