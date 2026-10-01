// SPDX-License-Identifier: Apache-2.0

use nodedb_lite::{NodeDbLite, PagedbStorageMem};
use std::sync::Arc;

pub(super) async fn open_test_db() -> Arc<NodeDbLite<PagedbStorageMem>> {
    let storage = PagedbStorageMem::open_in_memory().await.unwrap();
    NodeDbLite::open(storage).await.unwrap()
}
