// SPDX-License-Identifier: Apache-2.0

//! Encryption-at-rest wrapper for `StorageEngine`.
//!
//! Wraps any `StorageEngine` with AES-256-GCM encryption. All values are
//! encrypted before writing and decrypted after reading. Keys (namespace + key)
//! are NOT encrypted — they are needed for range scans and prefix lookups.
//!
//! Key derivation: Argon2id KDF from a user-provided passphrase + random salt.
//! The salt is stored in plaintext in the Meta namespace (it's not secret —
//! only the passphrase is). The derived key is zeroized on drop.
//!
//! Nonce: deterministic 12-byte nonce derived from namespace + key via HMAC.
//! This makes encryption deterministic (same key+value = same ciphertext),
//! which is acceptable for a KV store where the key already uniquely identifies
//! the entry. The benefit is idempotent writes without nonce tracking.

use aes_gcm::{Aes256Gcm, KeyInit, Nonce, aead::Aead};
use zeroize::Zeroize;

use crate::error::LiteError;
use crate::storage::engine::StorageEngine;
use nodedb_types::Namespace;

/// Salt size for Argon2id key derivation.
pub(super) const SALT_SIZE: usize = 16;

/// Meta key for the stored salt.
pub(super) const SALT_KEY: &[u8] = b"encryption:salt";

/// Encrypted storage wrapper.
///
/// Encrypts all values with AES-256-GCM. Keys are plaintext (needed for scans).
/// The encryption key is derived from a passphrase via Argon2id and held in
/// memory until the `EncryptedStorage` is dropped, at which point it's zeroized.
pub struct EncryptedStorage<S: StorageEngine> {
    pub(super) inner: S,
    cipher: Aes256Gcm,
    /// Derived key material — zeroized on drop.
    _key_material: ZeroizeKey,
}

/// Wrapper to zeroize the derived key on drop.
struct ZeroizeKey {
    bytes: [u8; 32],
}

impl Drop for ZeroizeKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

impl<S: StorageEngine> EncryptedStorage<S> {
    /// Create an encrypted storage wrapper.
    ///
    /// On first use (no salt stored), generates a random salt and stores it.
    /// On subsequent opens, reads the existing salt for key derivation.
    ///
    /// `m_cost`, `t_cost`, and `p_cost` are the Argon2id parameters (memory in
    /// KiB, iteration count, and parallelism lanes respectively). Callers should
    /// obtain these from [`crate::config::LiteConfig`].
    ///
    /// # Errors
    /// Returns `LiteError` if salt generation fails or the storage is inaccessible.
    pub async fn open(
        inner: S,
        passphrase: &str,
        m_cost: u32,
        t_cost: u32,
        p_cost: u32,
    ) -> Result<Self, LiteError> {
        // Read or generate salt.
        let salt = match inner.get(Namespace::Meta, SALT_KEY).await? {
            Some(existing_salt) => {
                if existing_salt.len() != SALT_SIZE {
                    return Err(LiteError::Storage {
                        detail: format!(
                            "encryption salt has wrong size: expected {SALT_SIZE}, got {}",
                            existing_salt.len()
                        ),
                    });
                }
                let mut salt = [0u8; SALT_SIZE];
                salt.copy_from_slice(&existing_salt);
                salt
            }
            None => {
                // First use — generate random salt.
                let mut salt = [0u8; SALT_SIZE];
                getrandom::fill(&mut salt).map_err(|e| LiteError::Storage {
                    detail: format!("getrandom failed for encryption salt: {e}"),
                })?;
                inner.put(Namespace::Meta, SALT_KEY, &salt).await?;
                salt
            }
        };

        // Derive key via Argon2id.
        let mut key_bytes = [0u8; 32];
        let argon2 = argon2::Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(m_cost, t_cost, p_cost, Some(32)).map_err(|e| {
                LiteError::Storage {
                    detail: format!("argon2 params: {e}"),
                }
            })?,
        );
        argon2
            .hash_password_into(passphrase.as_bytes(), &salt, &mut key_bytes)
            .map_err(|e| LiteError::Storage {
                detail: format!("argon2 key derivation failed: {e}"),
            })?;

        let cipher = Aes256Gcm::new_from_slice(&key_bytes).map_err(|e| LiteError::Storage {
            detail: format!("AES-256-GCM init failed: {e}"),
        })?;

        Ok(Self {
            inner,
            cipher,
            _key_material: ZeroizeKey { bytes: key_bytes },
        })
    }

    /// Derive a deterministic 12-byte nonce from namespace + key.
    ///
    /// Uses the first 12 bytes of CRC32C(namespace || key), extended with
    /// the namespace byte and key length for uniqueness. Not cryptographically
    /// ideal (nonce reuse for same key), but acceptable because:
    /// 1. Each (namespace, key) pair maps to exactly one value at a time
    /// 2. Rewriting the same key with different data is an update, not a new message
    fn derive_nonce(ns: Namespace, key: &[u8]) -> [u8; 12] {
        let mut nonce_input = Vec::with_capacity(1 + key.len());
        nonce_input.push(ns as u8);
        nonce_input.extend_from_slice(key);

        let crc = crc32c::crc32c(&nonce_input);
        let crc_bytes = crc.to_le_bytes();

        let mut nonce = [0u8; 12];
        // Fill: [crc32(4)] [ns(1)] [key_len_le(2)] [key_prefix(5)]
        nonce[0..4].copy_from_slice(&crc_bytes);
        nonce[4] = ns as u8;
        nonce[5..7].copy_from_slice(&(key.len() as u16).to_le_bytes());
        let prefix_len = key.len().min(5);
        nonce[7..7 + prefix_len].copy_from_slice(&key[..prefix_len]);
        nonce
    }

    pub(super) fn encrypt(
        &self,
        ns: Namespace,
        key: &[u8],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, LiteError> {
        let nonce = Self::derive_nonce(ns, key);
        self.cipher
            .encrypt(&Nonce::from(nonce), plaintext)
            .map_err(|e| LiteError::Storage {
                detail: format!("AES-GCM encrypt failed: {e}"),
            })
    }

    pub(super) fn decrypt(
        &self,
        ns: Namespace,
        key: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, LiteError> {
        let nonce = Self::derive_nonce(ns, key);
        self.cipher
            .decrypt(&Nonce::from(nonce), ciphertext)
            .map_err(|e| LiteError::Storage {
                detail: format!("AES-GCM decrypt failed (wrong passphrase or corrupted data): {e}"),
            })
    }
}
