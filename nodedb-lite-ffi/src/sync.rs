// SPDX-License-Identifier: Apache-2.0

//! CRDT sync to an Origin server.

use std::os::raw::c_char;
use std::sync::Arc;

use nodedb_lite::sync::{SyncClient, SyncConfig, SyncDelegate};

use crate::handle::NodeDbHandle;
use crate::status::{NODEDB_ERR_FAILED, NODEDB_ERR_NULL, NODEDB_ERR_UTF8, NODEDB_OK};
use crate::util::{ffi_guard, handle_ref, ptr_to_str};

/// Start background CRDT sync to an Origin server.
///
/// Connects via WebSocket to the given URL, authenticates with the JWT token,
/// and continuously pushes pending deltas / receives shape updates.
/// Runs forever in the background with auto-reconnect.
///
/// Returns `NODEDB_OK` on successful launch (sync runs asynchronously),
/// `NODEDB_ERR_FAILED` if a sync task is already running on this handle.
///
/// The background task is owned by the handle: `nodedb_close` stops it
/// deterministically before tearing down the runtime, and `nodedb_stop_sync`
/// stops it on demand (e.g. reconnect with a new token).
///
/// # Safety
/// `url` and `jwt_token` must be valid null-terminated UTF-8 strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_start_sync(
    handle: *mut NodeDbHandle,
    url: *const c_char,
    jwt_token: *const c_char,
) -> i32 {
    ffi_guard(NODEDB_ERR_FAILED, || {
        let Some(h) = handle_ref(handle) else {
            return NODEDB_ERR_NULL;
        };
        let Some(url_str) = ptr_to_str(url) else {
            return NODEDB_ERR_UTF8;
        };
        let Some(jwt_str) = ptr_to_str(jwt_token) else {
            return NODEDB_ERR_UTF8;
        };

        let config = SyncConfig {
            url: url_str.to_string(),
            jwt_token: jwt_str.to_string(),
            client_version: format!("nodedb-lite-ffi/{}", env!("CARGO_PKG_VERSION")),
            min_backoff: std::time::Duration::from_secs(1),
            max_backoff: std::time::Duration::from_secs(60),
            ping_interval: std::time::Duration::from_secs(30),
            max_batch_size: 100,
            token_provider: None,
            token_lifetime_secs: 0,
        };

        // Guard the slot: a second start must not silently detach the running
        // loop (dropping its JoinHandle would orphan it and make it
        // unstoppable). Reject instead.
        let mut slot = h.sync_task.lock().unwrap();
        if slot.is_some() {
            return NODEDB_ERR_FAILED;
        }

        // Spawn the sync loop ourselves (rather than via `start_sync`) so the
        // JoinHandle is retained in the handle — `nodedb_close` needs it to
        // stop the task deterministically before the runtime is dropped (#11).
        let client = Arc::new(SyncClient::new(config));
        let delegate: Arc<dyn SyncDelegate> = Arc::clone(&h.db) as _;
        let client_task = Arc::clone(&client);
        let task = h.rt.spawn(async move {
            nodedb_lite::sync::run_sync_loop(client_task, delegate).await;
        });
        *slot = Some(task);

        NODEDB_OK
    })
}

/// Stop background sync started by `nodedb_start_sync`.
///
/// Aborts the sync task and waits (bounded) for it to wind down. The database
/// handle stays open and usable; sync can be restarted with a new call.
///
/// Returns `NODEDB_OK` if a sync task was running and was stopped,
/// `NODEDB_ERR_FAILED` if no sync was active, `NODEDB_ERR_NULL` for a NULL
/// handle.
///
/// # Safety
/// `handle` must be a valid handle returned by `nodedb_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_stop_sync(handle: *mut NodeDbHandle) -> i32 {
    ffi_guard(NODEDB_ERR_FAILED, || {
        let Some(h) = handle_ref(handle) else {
            return NODEDB_ERR_NULL;
        };
        let Some(task) = h.sync_task.lock().unwrap().take() else {
            return NODEDB_ERR_FAILED;
        };
        task.abort();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut task = task;
        h.rt.block_on(async move {
            tokio::select! {
                _ = &mut task => {}
                _ = tokio::time::sleep_until(deadline) => {}
            }
        });
        NODEDB_OK
    })
}
