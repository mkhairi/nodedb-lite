// SPDX-License-Identifier: Apache-2.0

//! Deterministic storage interruptions for outbound ownership tests.

use crate::{
    error::LiteError,
    nodedb::lock_ext::LockExt,
    storage::{
        engine::{KvPair, StorageEngine, WriteOp},
        pagedb_storage::PagedbStorageMem,
    },
};
use nodedb_types::Namespace;
use std::sync::Mutex;
use tokio::sync::Notify;

#[derive(Clone, Copy)]
pub(super) enum Interruption {
    ErrorBefore,
    ErrorAfter,
    PauseBefore,
    PauseAfter,
    BackpressureAfter,
}

pub(super) struct PutRule {
    pub namespace: Namespace,
    pub skip: usize,
    pub interruption: Interruption,
}

pub(super) struct InterruptedStorage {
    pub inner: PagedbStorageMem,
    pub rule: Mutex<Option<PutRule>>,
    pub scan_error: Mutex<Option<Namespace>>,
    pub get_backpressure: Mutex<Option<Namespace>>,
    pub entered: Notify,
    pub resume: Notify,
}

impl InterruptedStorage {
    pub async fn new() -> Self {
        Self {
            inner: PagedbStorageMem::open_in_memory().await.unwrap(),
            rule: Mutex::new(None),
            scan_error: Mutex::new(None),
            get_backpressure: Mutex::new(None),
            entered: Notify::new(),
            resume: Notify::new(),
        }
    }

    fn interruption(&self, namespace: Namespace) -> Option<Interruption> {
        let mut rule = self.rule.lock_or_recover();
        if let Some(current) = rule.as_mut()
            && current.namespace == namespace
        {
            if current.skip == 0 {
                return rule.take().map(|rule| rule.interruption);
            }
            current.skip -= 1;
        }
        None
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl StorageEngine for InterruptedStorage {
    async fn get(&self, ns: Namespace, key: &[u8]) -> Result<Option<Vec<u8>>, LiteError> {
        let excluded = {
            let mut target = self.get_backpressure.lock_or_recover();
            if *target == Some(ns) {
                target.take();
                true
            } else {
                false
            }
        };
        if excluded {
            return Err(LiteError::Backpressure {
                detail: "injected read backpressure".into(),
            });
        }
        self.inner.get(ns, key).await
    }

    async fn put(&self, ns: Namespace, key: &[u8], value: &[u8]) -> Result<(), LiteError> {
        let interruption = self.interruption(ns);
        match interruption {
            Some(Interruption::ErrorBefore) => {
                return Err(LiteError::Storage {
                    detail: "injected pre-commit storage error".into(),
                });
            }
            Some(Interruption::PauseBefore) => {
                self.entered.notify_one();
                self.resume.notified().await;
            }
            _ => {}
        }
        self.inner.put(ns, key, value).await?;
        match interruption {
            Some(Interruption::ErrorAfter) => Err(LiteError::Storage {
                detail: "injected post-commit storage error".into(),
            }),
            Some(Interruption::BackpressureAfter) => Err(LiteError::Backpressure {
                detail: "injected post-commit backpressure".into(),
            }),
            Some(Interruption::PauseAfter) => {
                self.entered.notify_one();
                self.resume.notified().await;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    async fn delete(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
        self.inner.delete(ns, key).await
    }
    async fn scan_prefix(&self, ns: Namespace, prefix: &[u8]) -> Result<Vec<KvPair>, LiteError> {
        self.inner.scan_prefix(ns, prefix).await
    }
    async fn batch_write(&self, ops: &[WriteOp]) -> Result<(), LiteError> {
        self.inner.batch_write(ops).await
    }
    async fn count(&self, ns: Namespace) -> Result<u64, LiteError> {
        self.inner.count(ns).await
    }
    async fn scan_range(
        &self,
        ns: Namespace,
        start: &[u8],
        limit: usize,
    ) -> Result<Vec<KvPair>, LiteError> {
        let excluded = {
            let mut target = self.scan_error.lock_or_recover();
            if *target == Some(ns) {
                target.take();
                true
            } else {
                false
            }
        };
        if excluded {
            return Err(LiteError::Storage {
                detail: "injected drain storage error".into(),
            });
        }
        self.inner.scan_range(ns, start, limit).await
    }

    async fn scan_range_bounded(
        &self,
        ns: Namespace,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<KvPair>, LiteError> {
        self.inner.scan_range_bounded(ns, start, end, limit).await
    }
}
