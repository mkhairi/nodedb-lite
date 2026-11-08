// SPDX-License-Identifier: Apache-2.0

//! The `PagedbStorage` handle, its VFS aliases, and the shared `OpenOptions`.

use std::sync::Arc;

use pagedb::Db;
use pagedb::options::{OpenOptions, RetainPolicy};
use pagedb::vfs::Vfs;
use pagedb::vfs::memory::MemVfs;

#[cfg(not(target_arch = "wasm32"))]
use pagedb::vfs::DefaultVfs;

/// `PagedbStorage` backed by the native platform VFS (io_uring on Linux, etc.).
#[cfg(not(target_arch = "wasm32"))]
pub type PagedbStorageDefault = PagedbStorage<DefaultVfs>;

/// `PagedbStorage` backed by an in-memory VFS (tests / ephemeral use).
pub type PagedbStorageMem = PagedbStorage<MemVfs>;

/// `PagedbStorage` backed by the browser OPFS VFS (persistent, wasm32 only).
///
/// Constructed via [`PagedbStorage::open_opfs`](PagedbStorage).
#[cfg(target_arch = "wasm32")]
pub type PagedbStorageOpfs = PagedbStorage<pagedb::vfs::opfs::OpfsVfs>;

/// Build the `OpenOptions` used for all `PagedbStorage` instances.
///
/// `RetainPolicy::Disabled` is selected because Lite does not need
/// point-in-time reads; skipping commit-history tracking shaves latency
/// from every `WriteTxn::commit`.
///
/// Native builds raise the buffer pool from pagedb's 1,024 pages (4 MiB)
/// to [`NATIVE_BUFFER_POOL_PAGES`]. The browser build keeps the default.
pub(crate) fn lite_open_options() -> OpenOptions {
    let options = OpenOptions::default().with_commit_history_retain(RetainPolicy::Disabled);
    #[cfg(not(target_arch = "wasm32"))]
    let options = options.with_buffer_pool_pages(NATIVE_BUFFER_POOL_PAGES);
    options
}

/// Buffer pool size for native stores: 16,384 pages, 64 MiB at 4 KiB.
///
/// Measured on a 1.5 GB, 90k-object store with a read-only query mix.
/// The read hit rate rose from 91.8% at 1,024 pages to 98.9%. Warm query
/// p50 fell from 55 ms to 35 ms, for 63 MB more RSS. 65,536 pages gave the
/// same latency for 250 MB more. The pool also sets the dirty-page count
/// that triggers a spill, and write latency did not change at this size.
#[cfg(not(target_arch = "wasm32"))]
const NATIVE_BUFFER_POOL_PAGES: usize = 16_384;

/// pagedb-backed KV storage.
///
/// The inner `Db<V>` lives behind `Arc` for cheap cloning across async methods.
/// No outer `Mutex` is needed: `Db::begin_write` already acquires an internal
/// async mutex (single-writer serialization is enforced by pagedb itself).
pub struct PagedbStorage<V: Vfs + Clone> {
    pub(crate) db: Arc<Db<V>>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) snapshot_descriptor: Option<super::snapshot::metadata::SnapshotDescriptor>,
}

// Clone shares the database handle and copies only nonsecret snapshot metadata.
impl<V: Vfs + Clone> Clone for PagedbStorage<V> {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
            #[cfg(not(target_arch = "wasm32"))]
            snapshot_descriptor: self.snapshot_descriptor.clone(),
        }
    }
}
