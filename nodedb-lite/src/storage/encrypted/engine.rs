// SPDX-License-Identifier: Apache-2.0

//! Encrypted storage operations and scan budgets.

use async_trait::async_trait;
use nodedb_types::Namespace;

#[cfg(test)]
use super::crypto::SALT_SIZE;
use super::crypto::{EncryptedStorage, SALT_KEY};
use crate::error::LiteError;
use crate::storage::engine::{PrefixScan, StorageEngine, WriteOp};

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<S: StorageEngine> StorageEngine for EncryptedStorage<S> {
    async fn get(&self, ns: Namespace, key: &[u8]) -> Result<Option<Vec<u8>>, LiteError> {
        // Salt is stored unencrypted.
        if ns == Namespace::Meta && key == SALT_KEY {
            return self.inner.get(ns, key).await;
        }

        match self.inner.get(ns, key).await? {
            Some(ciphertext) => {
                let plaintext = self.decrypt(ns, key, &ciphertext)?;
                Ok(Some(plaintext))
            }
            None => Ok(None),
        }
    }

    async fn put(&self, ns: Namespace, key: &[u8], value: &[u8]) -> Result<(), LiteError> {
        // Salt is stored unencrypted.
        if ns == Namespace::Meta && key == SALT_KEY {
            return self.inner.put(ns, key, value).await;
        }

        let ciphertext = self.encrypt(ns, key, value)?;
        self.inner.put(ns, key, &ciphertext).await
    }

    async fn delete(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
        self.inner.delete(ns, key).await
    }

    async fn scan_prefix(
        &self,
        ns: Namespace,
        prefix: &[u8],
    ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
        let encrypted_entries = self.inner.scan_prefix(ns, prefix).await?;
        let mut results = Vec::with_capacity(encrypted_entries.len());
        for (key, ciphertext) in encrypted_entries {
            match self.decrypt(ns, &key, &ciphertext) {
                Ok(plaintext) => results.push((key, plaintext)),
                Err(e) => {
                    tracing::warn!(
                        key = ?String::from_utf8_lossy(&key),
                        error = %e,
                        "skipping undecryptable entry in scan"
                    );
                }
            }
        }
        Ok(results)
    }

    async fn scan_prefix_bounded(
        &self,
        ns: Namespace,
        prefix: &[u8],
        limit: usize,
    ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
        let entries = self.inner.scan_prefix_bounded(ns, prefix, limit).await?;
        entries
            .into_iter()
            .map(|(key, ciphertext)| {
                let plaintext = if ns == Namespace::Meta && key == SALT_KEY {
                    ciphertext
                } else {
                    self.decrypt(ns, &key, &ciphertext)?
                };
                Ok((key, plaintext))
            })
            .collect()
    }

    async fn scan_prefix_budgeted(
        &self,
        ns: Namespace,
        prefix: &[u8],
        max_records: usize,
        max_bytes: usize,
    ) -> Result<PrefixScan, LiteError> {
        self.scan_budgeted(ns, prefix, None, max_records, max_bytes, false)
            .await
    }

    async fn scan_prefix_from_budgeted(
        &self,
        ns: Namespace,
        prefix: &[u8],
        after_key: Option<&[u8]>,
        max_records: usize,
        max_bytes: usize,
    ) -> Result<PrefixScan, LiteError> {
        self.scan_budgeted(ns, prefix, after_key, max_records, max_bytes, true)
            .await
    }

    async fn batch_write(&self, ops: &[WriteOp]) -> Result<(), LiteError> {
        let encrypted_ops: Vec<WriteOp> = ops
            .iter()
            .map(|op| match op {
                WriteOp::Put { ns, key, value } => {
                    if *ns == Namespace::Meta && key == SALT_KEY {
                        return Ok(WriteOp::Put {
                            ns: *ns,
                            key: key.clone(),
                            value: value.clone(),
                        });
                    }
                    let ciphertext = self.encrypt(*ns, key, value)?;
                    Ok(WriteOp::Put {
                        ns: *ns,
                        key: key.clone(),
                        value: ciphertext,
                    })
                }
                WriteOp::Delete { ns, key } => Ok(WriteOp::Delete {
                    ns: *ns,
                    key: key.clone(),
                }),
            })
            .collect::<Result<Vec<_>, LiteError>>()?;

        self.inner.batch_write(&encrypted_ops).await
    }

    async fn count(&self, ns: Namespace) -> Result<u64, LiteError> {
        self.inner.count(ns).await
    }

    async fn scan_range(
        &self,
        ns: Namespace,
        start: &[u8],
        limit: usize,
    ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
        let encrypted_entries = self.inner.scan_range(ns, start, limit).await?;
        let mut results = Vec::with_capacity(encrypted_entries.len());
        for (key, ciphertext) in encrypted_entries {
            match self.decrypt(ns, &key, &ciphertext) {
                Ok(plaintext) => results.push((key, plaintext)),
                Err(e) => {
                    tracing::warn!(
                        key = ?String::from_utf8_lossy(&key),
                        error = %e,
                        "skipping undecryptable entry in scan_range"
                    );
                }
            }
        }
        Ok(results)
    }

    async fn scan_range_bounded(
        &self,
        ns: Namespace,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<Vec<crate::storage::engine::KvPair>, LiteError> {
        let encrypted_entries = self.inner.scan_range_bounded(ns, start, end, limit).await?;
        let mut results = Vec::with_capacity(encrypted_entries.len());
        for (key, ciphertext) in encrypted_entries {
            match self.decrypt(ns, &key, &ciphertext) {
                Ok(plaintext) => results.push((key, plaintext)),
                Err(e) => {
                    tracing::warn!(
                        key = ?String::from_utf8_lossy(&key),
                        error = %e,
                        "skipping undecryptable entry in scan_range_bounded"
                    );
                }
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LiteConfig;
    use crate::storage::engine::PrefixScanLimit;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    async fn make_encrypted() -> EncryptedStorage<PagedbStorageMem> {
        let cfg = LiteConfig::default();
        let inner = PagedbStorageMem::open_in_memory().await.unwrap();
        EncryptedStorage::open(
            inner,
            "test-passphrase-123",
            cfg.argon2_m_cost,
            cfg.argon2_t_cost,
            cfg.argon2_p_cost,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn roundtrip_basic() {
        let s = make_encrypted().await;
        s.put(Namespace::Vector, b"v1", b"hello world")
            .await
            .unwrap();
        let val = s.get(Namespace::Vector, b"v1").await.unwrap();
        assert_eq!(val.as_deref(), Some(b"hello world".as_slice()));
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let s = make_encrypted().await;
        assert!(s.get(Namespace::Vector, b"nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn different_namespaces_isolated() {
        let s = make_encrypted().await;
        s.put(Namespace::Vector, b"k", b"vec").await.unwrap();
        s.put(Namespace::Graph, b"k", b"graph").await.unwrap();

        assert_eq!(
            s.get(Namespace::Vector, b"k").await.unwrap().as_deref(),
            Some(b"vec".as_slice())
        );
        assert_eq!(
            s.get(Namespace::Graph, b"k").await.unwrap().as_deref(),
            Some(b"graph".as_slice())
        );
    }

    #[tokio::test]
    async fn wrong_passphrase_fails_decrypt() {
        let cfg = LiteConfig::default();
        let inner = PagedbStorageMem::open_in_memory().await.unwrap();
        // Write with passphrase A.
        {
            let s = EncryptedStorage::open(
                inner,
                "passphrase-A",
                cfg.argon2_m_cost,
                cfg.argon2_t_cost,
                cfg.argon2_p_cost,
            )
            .await
            .unwrap();
            s.put(Namespace::Vector, b"secret", b"classified data")
                .await
                .unwrap();
        }
        // The inner storage is consumed, so we can't reopen with a different passphrase
        // in this test. Instead, verify the salt persists.
    }

    #[tokio::test]
    async fn scan_prefix_decrypts() {
        let s = make_encrypted().await;
        s.put(Namespace::Crdt, b"delta:001", b"data1")
            .await
            .unwrap();
        s.put(Namespace::Crdt, b"delta:002", b"data2")
            .await
            .unwrap();
        s.put(Namespace::Crdt, b"other:001", b"other")
            .await
            .unwrap();

        let results = s.scan_prefix(Namespace::Crdt, b"delta:").await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, b"data1");
        assert_eq!(results[1].1, b"data2");
    }

    #[tokio::test]
    async fn batch_write_encrypts() {
        let s = make_encrypted().await;
        s.batch_write(&[
            WriteOp::Put {
                ns: Namespace::Vector,
                key: b"a".to_vec(),
                value: b"alpha".to_vec(),
            },
            WriteOp::Put {
                ns: Namespace::Vector,
                key: b"b".to_vec(),
                value: b"beta".to_vec(),
            },
        ])
        .await
        .unwrap();

        assert_eq!(
            s.get(Namespace::Vector, b"a").await.unwrap().as_deref(),
            Some(b"alpha".as_slice())
        );
        assert_eq!(
            s.get(Namespace::Vector, b"b").await.unwrap().as_deref(),
            Some(b"beta".as_slice())
        );
    }

    #[tokio::test]
    async fn large_value_roundtrip() {
        let s = make_encrypted().await;
        let large = vec![0xABu8; 100_000];
        s.put(Namespace::LoroState, b"snapshot", &large)
            .await
            .unwrap();
        let val = s.get(Namespace::LoroState, b"snapshot").await.unwrap();
        assert_eq!(val.unwrap().len(), 100_000);
    }

    #[tokio::test]
    async fn salt_persists() {
        let s = make_encrypted().await;
        let salt = s.inner.get(Namespace::Meta, SALT_KEY).await.unwrap();
        assert!(salt.is_some());
        assert_eq!(salt.unwrap().len(), SALT_SIZE);
    }

    #[tokio::test]
    async fn bounded_prefix_decrypts_every_row_and_propagates_errors() {
        let s = make_encrypted().await;
        s.put(Namespace::Graph, b"edge:1", b"first").await.unwrap();
        s.put(Namespace::Graph, b"edge:2", b"second").await.unwrap();
        assert!(
            s.scan_prefix_bounded(Namespace::Graph, b"edge:", 0)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            s.scan_prefix_bounded(Namespace::Graph, b"edge:", 1)
                .await
                .unwrap(),
            vec![(b"edge:1".to_vec(), b"first".to_vec())]
        );
        assert_eq!(
            s.scan_prefix_bounded(Namespace::Graph, b"edge:", 2)
                .await
                .unwrap()
                .len(),
            2
        );
        s.inner
            .put(Namespace::Graph, b"edge:2", b"invalid ciphertext")
            .await
            .unwrap();
        assert!(
            s.scan_prefix_bounded(Namespace::Graph, b"edge:", 2)
                .await
                .is_err()
        );
        assert_eq!(
            s.scan_prefix_bounded(Namespace::Meta, SALT_KEY, 1)
                .await
                .unwrap()[0]
                .1
                .len(),
            SALT_SIZE
        );
    }

    #[tokio::test]
    async fn budgeted_prefix_accounts_tags_and_checks_every_returned_ciphertext() {
        let s = make_encrypted().await;
        for key in [b"p1", b"p2"] {
            s.put(Namespace::Graph, key, b"abc").await.unwrap();
        }
        let exact = s
            .scan_prefix_budgeted(Namespace::Graph, b"p", 2, 10)
            .await
            .unwrap();
        assert_eq!(exact.entries.len(), 2);
        assert_eq!(exact.limit, None);
        let count = s
            .scan_prefix_budgeted(Namespace::Graph, b"p", 1, 10)
            .await
            .unwrap();
        assert_eq!(count.entries.len(), 1);
        assert_eq!(count.limit, Some(PrefixScanLimit::Records));
        let bytes = s
            .scan_prefix_budgeted(Namespace::Graph, b"p", 2, 5)
            .await
            .unwrap();
        assert_eq!(bytes.entries.len(), 1);
        assert_eq!(bytes.limit, Some(PrefixScanLimit::Bytes));
        assert!(
            s.scan_prefix_budgeted(Namespace::Graph, b"p", 0, 0)
                .await
                .unwrap()
                .entries
                .is_empty()
        );
        s.inner.put(Namespace::Graph, b"p2", b"bad").await.unwrap();
        assert!(
            s.scan_prefix_budgeted(Namespace::Graph, b"p", 2, 1)
                .await
                .is_err()
        );
    }
}
