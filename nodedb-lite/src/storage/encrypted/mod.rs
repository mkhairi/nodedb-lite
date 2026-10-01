// SPDX-License-Identifier: Apache-2.0

//! Encryption-at-rest storage wrapper.

mod crypto;
mod engine;

pub use crypto::EncryptedStorage;
