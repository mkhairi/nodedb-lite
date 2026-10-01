// SPDX-License-Identifier: Apache-2.0

//! `StorageEngine` implementations, one per target family.

mod rows;
mod scans;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;
