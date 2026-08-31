// SPDX-License-Identifier: Apache-2.0

//! `NodeDbLite::start_auto_flush` — durable background flush task.

use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::storage::engine::StorageEngine;

use super::types::NodeDbLite;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Start a background task that calls the global `flush()` every
    /// `interval_ms` milliseconds, bounding the data-loss window uniformly
    /// across all engines (KV buffer, vector id-map, CRDT deltas, CSR graph,
    /// spatial, FTS).
    ///
    /// # Durability contract
    ///
    /// `await`-ing a write operation (e.g. `kv_put`, `vector_insert`) returning
    /// `Ok` does NOT guarantee on-disk durability. Durability is bounded by
    /// `interval_ms`. For guaranteed durability, call `flush()` explicitly after
    /// writes.
    ///
    /// # Usage
    ///
    /// The `open*` constructors already start this task from
    /// [`LiteConfig::auto_flush_ms`](crate::config::LiteConfig::auto_flush_ms),
    /// so calling it is only needed to change the interval afterwards:
    ///
    /// ```ignore
    /// let db = NodeDbLite::open(storage).await?;
    /// db.start_auto_flush(5_000); // slow the flusher down to five seconds
    /// ```
    ///
    /// Each call replaces the auto-flush task already running rather than
    /// adding a second one. Two tasks of this kind keep the database alive
    /// between them — each holds a strong handle across its flush, and their
    /// hold windows overlap — so the store could never be dropped and every
    /// later reopen failed with "already open".
    ///
    /// Opening with `auto_flush_ms: 0` is still the way to take full manual
    /// control of when flushes happen.
    ///
    /// # Task lifecycle
    ///
    /// The task is registered with the database's task registry, so
    /// [`NodeDbLite::shutdown`](crate::NodeDbLite::shutdown) stops it before
    /// the host drops its async runtime. It also holds a `Weak` reference, so
    /// a database dropped without a shutdown still ends the loop at its next
    /// tick rather than keeping the database alive.
    ///
    /// # Disabling
    ///
    /// Pass `interval_ms = 0` to skip spawning entirely (auto-flush disabled).
    pub fn start_auto_flush(self: &Arc<Self>, interval_ms: u64) {
        if interval_ms == 0 {
            return;
        }

        // Replace any auto-flush task already running, rather than adding a
        // second one. Each task upgrades its `Weak` to a strong handle and
        // holds it across the flush; one task's hold window has gaps, and the
        // database drops in a gap. Two tasks on the same period interleave, so
        // the strong count never reaches zero, neither `Weak` upgrade ever
        // fails, and neither task ever takes its exit branch. The store is then
        // never dropped, its file lock is held for the life of the process, and
        // every reopen fails with "already open".
        //
        // Calling this to change the interval is documented usage, and
        // `open_with_config` has already started one from
        // `LiteConfig::auto_flush_ms`, so reaching the two-task state took
        // nothing more than following the docs.
        self.tasks.stop_nowait(crate::tasks::TaskKind::AutoFlush);

        let weak: Weak<Self> = Arc::downgrade(self);
        let period = Duration::from_millis(interval_ms);

        let (stop_tx, mut stop) = crate::tasks::TaskRegistry::signal();
        let handle = crate::runtime::spawn(async move {
            let mut ticker = crate::runtime::interval(period);
            // A flush that outlasts its own period must not be followed by the
            // ticks it missed, back to back. Each pass takes the `crdt` lock,
            // so a burst of them is a stretch of wall time in which no CRDT
            // read can be scheduled — the shape a large store degenerated into
            // once a single flush took longer than `interval_ms`.
            ticker.delay_missed_ticks();
            // Consume the first tick so the initial period elapses before the
            // first flush (matches Tokio's immediate-first-tick semantics on
            // native; on WASM the first tick already waits one period).
            ticker.tick().await;

            loop {
                if !stop.tick_or_stop(&mut ticker).await {
                    break;
                }

                let db = match weak.upgrade() {
                    Some(db) => db,
                    None => break,
                };

                if let Err(e) = db.flush().await {
                    tracing::warn!(error = %e, "auto-flush failed");
                }

                // Drop the strong Arc before the next tick so the loop does
                // not keep the database alive between ticks.
                drop(db);
            }
        });
        self.tasks
            .track(crate::tasks::TaskKind::AutoFlush, stop_tx, handle);
    }
}
