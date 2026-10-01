// SPDX-License-Identifier: Apache-2.0

use nodedb_lite::{PagedbStorageMem, storage::engine::StorageEngine};
use nodedb_types::Namespace;

#[tokio::test]
async fn prefix_limits_preserve_order_and_namespace_isolation() {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    for key in [b"edge:3", b"edge:1", b"edge:2", b"other:"] {
        storage.put(Namespace::Graph, key, key).await.unwrap();
    }
    storage
        .put(Namespace::Vector, b"edge:0", b"outside")
        .await
        .unwrap();
    assert!(
        storage
            .scan_prefix_bounded(Namespace::Graph, b"edge:", 0)
            .await
            .unwrap()
            .is_empty()
    );
    let one = storage
        .scan_prefix_bounded(Namespace::Graph, b"edge:", 1)
        .await
        .unwrap();
    assert_eq!(one, vec![(b"edge:1".to_vec(), b"edge:1".to_vec())]);
    let exact = storage
        .scan_prefix_bounded(Namespace::Graph, b"edge:", 3)
        .await
        .unwrap();
    assert_eq!(exact.len(), 3);
    assert_eq!(exact[2].0, b"edge:3");
    assert_eq!(
        storage
            .scan_prefix_bounded(Namespace::Graph, b"edge:", 4)
            .await
            .unwrap(),
        exact
    );
    assert!(
        storage
            .scan_prefix_bounded(Namespace::Graph, b"missing:", 10)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn budgeted_prefix_reports_exact_and_cumulative_limits() {
    use nodedb_lite::storage::engine::PrefixScanLimit;
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    for key in [b"p1", b"p2"] {
        storage.put(Namespace::Graph, key, b"abc").await.unwrap();
    }
    storage
        .put(Namespace::Vector, b"p0", b"outside")
        .await
        .unwrap();
    let exact = storage
        .scan_prefix_budgeted(Namespace::Graph, b"p", 2, 10)
        .await
        .unwrap();
    assert_eq!(exact.entries.len(), 2);
    assert_eq!(exact.limit, None);
    let count = storage
        .scan_prefix_budgeted(Namespace::Graph, b"p", 1, 10)
        .await
        .unwrap();
    assert_eq!(count.entries.len(), 1);
    assert_eq!(count.limit, Some(PrefixScanLimit::Records));
    let bytes = storage
        .scan_prefix_budgeted(Namespace::Graph, b"p", 2, 5)
        .await
        .unwrap();
    assert_eq!(bytes.entries, vec![(b"p1".to_vec(), b"abc".to_vec())]);
    assert_eq!(bytes.limit, Some(PrefixScanLimit::Bytes));
    let small = storage
        .scan_prefix_budgeted(Namespace::Graph, b"p", 2, 4)
        .await
        .unwrap();
    assert!(small.entries.is_empty());
    assert_eq!(small.limit, Some(PrefixScanLimit::Bytes));
    let zero = storage
        .scan_prefix_budgeted(Namespace::Graph, b"p", 0, 10)
        .await
        .unwrap();
    assert!(zero.entries.is_empty());
    assert_eq!(zero.limit, None);
    storage
        .put(Namespace::Graph, b"large", &[0; 4096])
        .await
        .unwrap();
    let large = storage
        .scan_prefix_budgeted(Namespace::Graph, b"large", 1, 64)
        .await
        .unwrap();
    assert!(large.entries.is_empty());
    assert_eq!(large.limit, Some(PrefixScanLimit::Bytes));
}
