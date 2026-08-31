// SPDX-License-Identifier: Apache-2.0

//! Lifecycle for the database's background tasks.
//!
//! Auto-flush, auto-compact and the sync loop all outlive the call that starts
//! them. Left detached, nothing can stop them: the host tears its async runtime
//! down while a task is still polling, and on a native runtime that lands the
//! process somewhere it cannot recover from.
//!
//! Every such task is registered here with a stop signal and a handle. Shutdown
//! is cooperative first: the signal is set, each task leaves its loop at a point
//! it chose, and only a task that ignores the signal past
//! [`TASK_STOP_TIMEOUT`] is aborted. Cancelling a database task at an arbitrary
//! await point can drop it mid-write, so abort is the backstop, never the
//! mechanism.

use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::watch;

use crate::runtime::{Interval, TaskHandle};

/// How long shutdown waits for a task to leave its loop before aborting it.
pub const TASK_STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Which background task a registration belongs to.
///
/// Lets [`TaskRegistry::stop`] cancel one kind — stopping sync without closing
/// the database — while shutdown stops every kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    AutoFlush,
    AutoCompact,
    Sync,
}

/// The stop side of a task's signal, held by the task itself.
pub struct StopSignal {
    rx: watch::Receiver<bool>,
}

impl StopSignal {
    /// Whether shutdown has been requested.
    pub fn is_stopping(&self) -> bool {
        *self.rx.borrow()
    }

    /// Resolve once shutdown is requested.
    ///
    /// Resolves immediately when the signal is already set, and also when the
    /// sender is gone — a dropped registry means the database is going away,
    /// which is the same answer.
    pub async fn stopped(&mut self) {
        while !*self.rx.borrow_and_update() {
            if self.rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Wait for the next tick of `ticker`, unless shutdown comes first.
    ///
    /// Returns `false` when the task must leave its loop. A periodic task
    /// polls on a long period — minutes, for compaction — so waiting for the
    /// next tick to notice the signal would stall shutdown for that long.
    pub async fn tick_or_stop(&mut self, ticker: &mut Interval) -> bool {
        tokio::select! {
            _ = ticker.tick() => !self.is_stopping(),
            _ = self.stopped() => false,
        }
    }
}

/// One registered task: its stop signal and its handle.
struct Registered {
    kind: TaskKind,
    stop: watch::Sender<bool>,
    handle: TaskHandle,
}

/// The database's set of running background tasks.
#[derive(Default)]
pub struct TaskRegistry {
    tasks: Mutex<Vec<Registered>>,
}

impl TaskRegistry {
    /// Create the stop signal for a task that is about to be spawned.
    ///
    /// The sender is handed back to [`TaskRegistry::track`] with the task's
    /// handle once the spawn succeeds. Separating the two keeps the registry
    /// free of the `Send` bound that differs between native and WASM spawns.
    pub fn signal() -> (watch::Sender<bool>, StopSignal) {
        let (tx, rx) = watch::channel(false);
        (tx, StopSignal { rx })
    }

    /// Record a spawned task so shutdown can stop it.
    pub fn track(&self, kind: TaskKind, stop: watch::Sender<bool>, handle: TaskHandle) {
        self.lock().push(Registered { kind, stop, handle });
    }

    /// Whether a task of `kind` is registered.
    pub fn has(&self, kind: TaskKind) -> bool {
        self.lock().iter().any(|task| task.kind == kind)
    }

    /// Signal every task of `kind` to stop, without waiting for it.
    ///
    /// For a caller that is replacing a periodic task from a synchronous
    /// context and cannot await the old one. The outgoing task leaves its loop
    /// at its next stop check and is dropped from the registry now, so it can
    /// no longer hold the database alive past its current iteration — which is
    /// what two overlapping tasks of the same kind do to each other.
    ///
    /// Returns `true` when at least one task was signalled.
    pub fn stop_nowait(&self, kind: TaskKind) -> bool {
        let taken: Vec<Registered> = {
            let mut tasks = self.lock();
            let (matching, rest) = tasks.drain(..).partition(|task| task.kind == kind);
            *tasks = rest;
            matching
        };
        for task in &taken {
            let _ = task.stop.send(true);
        }
        !taken.is_empty()
    }

    /// Stop every task of `kind` and wait for it to wind down.
    ///
    /// Returns `true` when at least one task was running.
    pub async fn stop(&self, kind: TaskKind) -> bool {
        let taken: Vec<Registered> = {
            let mut tasks = self.lock();
            let (matching, rest) = tasks.drain(..).partition(|task| task.kind == kind);
            *tasks = rest;
            matching
        };
        let stopped = !taken.is_empty();
        Self::wind_down(taken).await;
        stopped
    }

    /// Stop every registered task and wait for them to wind down.
    ///
    /// Idempotent: a second call has nothing left to stop.
    pub async fn shutdown(&self) {
        let taken: Vec<Registered> = self.lock().drain(..).collect();
        Self::wind_down(taken).await;
    }

    /// Signal each task, then join it, aborting whatever outlasts the timeout.
    async fn wind_down(tasks: Vec<Registered>) {
        // Signal every task before joining any of them, so their wind-downs
        // overlap instead of costing the timeout each in turn.
        for task in &tasks {
            let _ = task.stop.send(true);
        }
        for mut task in tasks {
            if !task.handle.join_within(TASK_STOP_TIMEOUT).await {
                tracing::warn!(
                    kind = ?task.kind,
                    timeout_ms = TASK_STOP_TIMEOUT.as_millis(),
                    "background task ignored its stop signal; aborting"
                );
                task.handle.abort();
            }
        }
    }

    /// Take the lock, recovering from a poisoned one.
    ///
    /// A panic in a task's registration must not make the database
    /// unclosable — the whole point of the registry is that shutdown works.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Registered>> {
        self.tasks.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stop_signal_resolves_when_set() {
        let (tx, mut signal) = TaskRegistry::signal();
        assert!(!signal.is_stopping());
        tx.send(true).unwrap();
        signal.stopped().await;
        assert!(signal.is_stopping());
    }

    #[tokio::test]
    async fn stop_signal_resolves_when_sender_dropped() {
        let (tx, mut signal) = TaskRegistry::signal();
        drop(tx);
        // A dropped registry means the database is going away.
        signal.stopped().await;
    }

    #[tokio::test]
    async fn shutdown_stops_a_registered_task() {
        let registry = TaskRegistry::default();
        let (tx, mut signal) = TaskRegistry::signal();
        let handle = crate::runtime::spawn(async move {
            signal.stopped().await;
        });
        registry.track(TaskKind::AutoFlush, tx, handle);
        assert!(registry.has(TaskKind::AutoFlush));

        registry.shutdown().await;
        assert!(!registry.has(TaskKind::AutoFlush));
    }

    #[tokio::test]
    async fn stop_takes_only_the_named_kind() {
        let registry = TaskRegistry::default();
        for kind in [TaskKind::AutoFlush, TaskKind::Sync] {
            let (tx, mut signal) = TaskRegistry::signal();
            let handle = crate::runtime::spawn(async move {
                signal.stopped().await;
            });
            registry.track(kind, tx, handle);
        }

        assert!(registry.stop(TaskKind::Sync).await);
        assert!(!registry.has(TaskKind::Sync));
        assert!(registry.has(TaskKind::AutoFlush));

        // Nothing of that kind is left to stop.
        assert!(!registry.stop(TaskKind::Sync).await);

        registry.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_aborts_a_task_that_ignores_the_signal() {
        let registry = TaskRegistry::default();
        let (tx, _signal) = TaskRegistry::signal();
        let handle = crate::runtime::spawn(async move {
            // Never observes the signal; shutdown must abort it.
            std::future::pending::<()>().await;
        });
        registry.track(TaskKind::Sync, tx, handle);

        // Would hang without the abort backstop; the timeout bounds it.
        tokio::time::timeout(TASK_STOP_TIMEOUT * 2, registry.shutdown())
            .await
            .expect("shutdown must not outlast the stop timeout");
    }
}
