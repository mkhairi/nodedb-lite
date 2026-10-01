// SPDX-License-Identifier: Apache-2.0

//! Encryption-at-rest storage wrapper.

mod crypto;
mod engine;
mod prefix_scan;

pub use crypto::EncryptedStorage;
