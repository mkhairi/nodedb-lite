// SPDX-License-Identifier: Apache-2.0

//! Persistence phases for `NodeDbLite::flush`.

mod execute;
mod indexes;
#[cfg(not(target_arch = "wasm32"))]
mod segments;
