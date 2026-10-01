// SPDX-License-Identifier: Apache-2.0

//! Encrypted prefix continuation without skipping excluded plaintext rows.

use nodedb_types::Namespace;

use super::crypto::{EncryptedStorage, SALT_KEY};
use crate::error::LiteError;
use crate::storage::engine::{
    PrefixScan, PrefixScanLimit, StorageEngine, check_prefix_cursor, prefix_scan_budget_error,
};

impl<S: StorageEngine> EncryptedStorage<S> {
    pub(super) async fn scan_budgeted(
        &self,
        ns: Namespace,
        prefix: &[u8],
        after_key: Option<&[u8]>,
        max_records: usize,
        max_bytes: usize,
        continuation: bool,
    ) -> Result<PrefixScan, LiteError> {
        if max_records == 0 {
            return Ok(PrefixScan::default());
        }
        check_prefix_cursor(prefix, after_key)?;
        // AES-GCM adds a fixed 16-byte authentication tag to each value.
        let encrypted_budget = max_bytes.saturating_add(max_records.saturating_mul(16));
        let scan = if continuation {
            self.inner
                .scan_prefix_from_budgeted(ns, prefix, after_key, max_records, encrypted_budget)
                .await
                .map_err(|error| match error {
                    LiteError::Backpressure { detail } => LiteError::Backpressure {
                        detail: format!(
                            "{detail}. Plaintext scan prefix {prefix:?}, cursor {after_key:?}, byte budget {max_bytes}"
                        ),
                    },
                    error => error,
                })?
        } else {
            self.inner
                .scan_prefix_budgeted(ns, prefix, max_records, encrypted_budget)
                .await?
        };
        let mut result = PrefixScan {
            entries: Vec::with_capacity(scan.entries.len()),
            limit: scan.limit,
        };
        let mut bytes = 0usize;
        for (key, ciphertext) in scan.entries {
            let plaintext = if ns == Namespace::Meta && key == SALT_KEY {
                ciphertext
            } else {
                self.decrypt(ns, &key, &ciphertext)?
            };
            let next_bytes = key
                .len()
                .checked_add(plaintext.len())
                .and_then(|n| bytes.checked_add(n));
            let Some(next_bytes) = next_bytes.filter(|&n| n <= max_bytes) else {
                result.limit = Some(PrefixScanLimit::Bytes);
                if continuation {
                    break;
                }
                // Existing non-continuation scans authenticate every returned ciphertext.
                bytes = usize::MAX;
                continue;
            };
            bytes = next_bytes;
            result.entries.push((key, plaintext));
        }
        if continuation && result.entries.is_empty() && result.limit == Some(PrefixScanLimit::Bytes)
        {
            return Err(prefix_scan_budget_error(prefix, after_key, max_bytes));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LiteConfig;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    #[tokio::test]
    async fn encrypted_continuation_retains_exact_boundaries_and_excluded_rows() {
        let config = LiteConfig::default();
        let inner = PagedbStorageMem::open_in_memory().await.unwrap();
        let storage = EncryptedStorage::open(
            inner,
            "test-passphrase-123",
            config.argon2_m_cost,
            config.argon2_t_cost,
            config.argon2_p_cost,
        )
        .await
        .unwrap();
        for (key, value) in [
            (b"p1".as_slice(), b"abc".as_slice()),
            (b"p2", b"0123456789"),
            (b"p3", b"x"),
        ] {
            storage.put(Namespace::Graph, key, value).await.unwrap();
        }
        let batch = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", None, 3, 5)
            .await
            .unwrap();
        assert_eq!(batch.entries, vec![(b"p1".to_vec(), b"abc".to_vec())]);
        assert_eq!(batch.limit, Some(PrefixScanLimit::Bytes));
        assert!(matches!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"p1"), 3, 5)
                .await,
            Err(LiteError::Backpressure { .. })
        ));
        let resumed = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"p1"), 3, 15)
            .await
            .unwrap();
        assert_eq!(
            resumed.entries,
            vec![
                (b"p2".to_vec(), b"0123456789".to_vec()),
                (b"p3".to_vec(), b"x".to_vec())
            ]
        );
        assert_eq!(resumed.limit, None);
        let counted = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", None, 1, 100)
            .await
            .unwrap();
        assert_eq!(counted.entries[0].0, b"p1");
        assert_eq!(counted.limit, Some(PrefixScanLimit::Records));
        assert!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", None, 1, 4)
                .await
                .is_err()
        );
        assert!(matches!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"q"), 1, 100)
                .await,
            Err(LiteError::BadRequest { .. })
        ));
        let salt = storage
            .scan_prefix_from_budgeted(Namespace::Meta, SALT_KEY, None, 1, 100)
            .await
            .unwrap();
        assert_eq!(salt.entries[0].1.len(), super::super::crypto::SALT_SIZE);
        storage
            .inner
            .put(Namespace::Graph, b"p3", b"invalid ciphertext")
            .await
            .unwrap();
        let stopped = storage
            .scan_prefix_from_budgeted(Namespace::Graph, b"p", None, 10, 5)
            .await
            .unwrap();
        assert_eq!(stopped.entries, vec![(b"p1".to_vec(), b"abc".to_vec())]);
        assert_eq!(stopped.limit, Some(PrefixScanLimit::Bytes));
        assert!(
            storage
                .scan_prefix_from_budgeted(Namespace::Graph, b"p", Some(b"p2"), 1, 100)
                .await
                .is_err()
        );
    }
}
