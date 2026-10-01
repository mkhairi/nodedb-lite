// SPDX-License-Identifier: Apache-2.0

//! FTS outbound ownership and durable transport wiring.

#[cfg(test)]
use state::injected;
mod spill;
mod staging;
mod state;
mod transport;
mod types;

pub use state::FtsOutbound;
pub use types::{PendingFtsDelete, PendingFtsIndex};
