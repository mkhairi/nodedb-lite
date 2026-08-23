// SPDX-License-Identifier: Apache-2.0

//! The opaque database handle and the temp directory backing `:memory:`.

use std::sync::Arc;

use nodedb_lite::{NodeDbLite, PagedbStorageDefault};

/// Minimal RAII temp-directory wrapper used for the `:memory:` path.
///
/// Deleted on drop. No external crate dependency required.
pub(crate) struct OwnedTempDir(pub(crate) std::path::PathBuf);

impl Drop for OwnedTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl OwnedTempDir {
    /// Create a unique temporary directory under `std::env::temp_dir()`.
    pub(crate) fn new() -> Option<Self> {
        let mut path = std::env::temp_dir();
        // Use process-id + a monotonic counter for uniqueness.
        let pid = std::process::id();
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        path.push(format!("nodedb-lite-ffi-{pid}-{n}"));
        if std::fs::create_dir_all(&path).is_ok() {
            Some(Self(path))
        } else {
            None
        }
    }
}

/// Opaque handle to a NodeDB-Lite database.
///
/// Created by `nodedb_open`, freed by `nodedb_close`.
///
/// The `Arc<NodeDbLite>` is what the constructor returned, so the background
/// flush and compaction tasks live exactly as long as this handle does.
///
/// `_tmpdir` is `Some` when the database was opened with the `:memory:` path.
/// The directory is deleted when the handle is dropped.
///
/// `sync_task` holds the background sync loop (if `nodedb_start_sync` was
/// called). It is aborted and joined before the runtime is dropped — see
/// `Drop` — so closing a handle with active sync cannot tear the runtime down
/// under a mid-poll sync task (SIGSEGV, #11). Mutex because handles are shared
/// behind `Arc` (registry lookups) while start/stop mutate the slot.
pub struct NodeDbHandle {
    pub(crate) db: Arc<NodeDbLite<PagedbStorageDefault>>,
    pub(crate) rt: tokio::runtime::Runtime,
    pub(crate) _tmpdir: Option<OwnedTempDir>,
    pub(crate) sync_task: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// How long `Drop` waits for the sync task to wind down after abort.
pub(crate) const SYNC_STOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Drop for NodeDbHandle {
    fn drop(&mut self) {
        // Stop sync deterministically BEFORE the runtime is dropped: the sync
        // task may be mid-poll (connect/handshake/push) on a worker thread,
        // and tearing the runtime down under it crashed the process (#11).
        let task = self.sync_task.lock().unwrap().take();
        if let Some(task) = task {
            task.abort();
            // Bounded wait: abort cancels the task at its next await point.
            // A connect attempt can take a moment, so cap the wait — the
            // runtime drop after the deadline is still safe because the
            // runtime's own shutdown also cancels remaining tasks; the abort
            // makes cancellation immediate instead of racing teardown.
            let deadline = tokio::time::Instant::now() + SYNC_STOP_TIMEOUT;
            let mut task = task;
            self.rt.block_on(async move {
                tokio::select! {
                    _ = &mut task => {}
                    _ = tokio::time::sleep_until(deadline) => {}
                }
            });
        }
    }
}
