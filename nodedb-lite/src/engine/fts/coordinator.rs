// SPDX-License-Identifier: Apache-2.0

//! Source-write admission and sticky text-checkpoint trust.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::OwnedMutexGuard;

use super::state::FtsState;
use crate::error::LiteError;

/// Exclusive admission shared by admitted helpers and checkpoint writers.
pub(crate) struct TextMutationPermit {
    _permit: OwnedMutexGuard<()>,
}

/// A source mutation marks checkpoint trust false unless it finishes successfully.
pub(crate) struct TextMutationGuard {
    state: Arc<FtsState>,
    permit: TextMutationPermit,
    armed: bool,
}

impl FtsState {
    pub(crate) async fn admit_mutation(self: &Arc<Self>) -> TextMutationGuard {
        self.mutation_guard(self.admit_exclusive().await)
    }

    /// Transfer owned admission into a source-write guard inside durable work.
    pub(crate) fn mutation_guard(
        self: &Arc<Self>,
        permit: TextMutationPermit,
    ) -> TextMutationGuard {
        TextMutationGuard {
            state: self.clone(),
            permit,
            armed: true,
        }
    }

    pub(crate) fn try_admit_mutation(
        self: &Arc<Self>,
        operation: &str,
    ) -> Result<TextMutationGuard, LiteError> {
        let permit = self.mutation_gate.clone().try_lock_owned().map_err(|_| LiteError::Backpressure {
            detail: format!("text mutation admission busy for '{operation}': retry after the current mutation"),
        })?;
        Ok(TextMutationGuard {
            state: self.clone(),
            permit: TextMutationPermit { _permit: permit },
            armed: true,
        })
    }

    pub(crate) async fn admit_exclusive(self: &Arc<Self>) -> TextMutationPermit {
        TextMutationPermit {
            _permit: self.mutation_gate.clone().lock_owned().await,
        }
    }

    /// Persist collection deletion policy before its source metadata changes.
    pub(crate) async fn persist_collection_tombstone<S: crate::storage::engine::StorageEngine>(
        &self,
        storage: &S,
        collection: &str,
        _permit: &TextMutationPermit,
    ) -> Result<(), LiteError> {
        let (record, replacement) = {
            let manager = self.manager.lock().map_err(|_| LiteError::LockPoisoned)?;
            let record = manager.next_declaration_record(collection, &None)?;
            let replacement = manager.begin_replacement(collection, record.clone())?;
            (record, replacement)
        };
        super::checkpoint::persist_checkpoint_incomplete(storage).await?;
        super::catalog::persist_declaration_tombstone(storage, collection, record.revision).await?;
        self.manager
            .lock()
            .map_err(|_| LiteError::LockPoisoned)?
            .publish_replacement(replacement);
        Ok(())
    }

    pub(crate) fn checkpoint_trusted(&self) -> bool {
        self.checkpoint_trusted.load(Ordering::Acquire)
    }

    /// Only successful full startup recovery restores trust.
    pub(crate) fn mark_checkpoint_trusted(&self) {
        self.checkpoint_trusted.store(true, Ordering::Release);
    }
}

impl TextMutationGuard {
    pub(crate) fn permit(&self) -> &TextMutationPermit {
        &self.permit
    }

    pub(crate) fn finish<T, E>(mut self, result: Result<T, E>) -> Result<T, E> {
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for TextMutationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.state
                .checkpoint_trusted
                .store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FtsState;
    use crate::error::LiteError;
    use std::sync::Arc;

    fn state() -> Arc<FtsState> {
        let state = Arc::new(FtsState::new(crate::engine::fts::manager::test_governor()));
        state.mark_checkpoint_trusted();
        state
    }

    #[tokio::test]
    async fn mutation_error_and_cancelled_guard_keep_distrust_sticky() {
        let state = state();
        let guard = state.admit_mutation().await;
        assert!(guard.finish::<(), _>(Err("operation error")).is_err());
        assert!(!state.checkpoint_trusted());
        state
            .admit_mutation()
            .await
            .finish::<(), ()>(Ok(()))
            .unwrap();
        assert!(!state.checkpoint_trusted());
        state.mark_checkpoint_trusted();
        drop(state.admit_mutation().await);
        assert!(!state.checkpoint_trusted());
    }

    #[tokio::test]
    async fn synchronous_busy_admission_preserves_trust() {
        let state = state();
        let permit = state.admit_exclusive().await;
        assert!(matches!(
            state.try_admit_mutation("bulk update"),
            Err(LiteError::Backpressure { .. })
        ));
        assert!(state.checkpoint_trusted());
        drop(permit);
        state
            .try_admit_mutation("bulk update")
            .unwrap()
            .finish::<(), ()>(Ok(()))
            .unwrap();
        assert!(state.checkpoint_trusted());
    }

    #[tokio::test]
    async fn asynchronous_mutation_waits_for_exclusive_admission() {
        let state = state();
        let permit = state.admit_exclusive().await;
        let mut admission = Box::pin(state.admit_mutation());
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut admission)
                .await
                .is_err()
        );
        assert!(state.checkpoint_trusted());
        drop(permit);
        let guard = admission.await;
        let _borrowed = guard.permit();
        guard.finish::<(), ()>(Ok(())).unwrap();
        assert!(state.checkpoint_trusted());
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn cancelled_task_marks_distrust_before_releasing_admission() {
        let state = state();
        let owned = state.clone();
        let (entered, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let guard = owned.admit_mutation().await;
            entered.send(()).unwrap();
            std::future::pending::<()>().await;
            guard.finish::<(), ()>(Ok(())).unwrap();
        });
        ready.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(!state.checkpoint_trusted());
        state
            .try_admit_mutation("after cancellation")
            .unwrap()
            .finish::<(), ()>(Ok(()))
            .unwrap();
    }
}
