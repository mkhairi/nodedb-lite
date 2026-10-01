// SPDX-License-Identifier: Apache-2.0

//! Columnar engine for Lite: manages per-collection memtables, segments,
//! delete bitmaps, and PK indexes against the StorageEngine.
//!
//! Segments are stored in the `Columnar` namespace as:
//! - `{collection}:seg:{segment_id}` — segment bytes
//! - `{collection}:del:{segment_id}` — delete bitmap bytes
//! - `{collection}:meta` — segment metadata (list of segment IDs + row counts)
//!
//! Schemas are stored in the `Meta` namespace as `columnar_schema:{collection}`.
//!
//! All public methods take `&self`. The collection map lives behind an
//! `RwLock`; each collection's mutable state lives behind an inner
//! `std::sync::Mutex` that is only ever held briefly and never across `.await`.

mod compact;
mod decode;
mod flush;
mod lifecycle;
mod mutate;
mod read;
mod schema;
mod segments;
mod state;

pub(super) use segments::remove_segment_bytes;
pub use state::ColumnarEngine;
pub(super) use state::SegmentMeta;
