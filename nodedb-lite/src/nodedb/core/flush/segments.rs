// SPDX-License-Identifier: Apache-2.0

//! Persist vector payloads and graph adjacency segments.

use super::indexes::IndexJobs;
use crate::{engine::vector::segment::segment_payload, storage::engine::StorageEngine};
use nodedb_types::error::{NodeDbError, NodeDbResult};

pub(super) async fn write_segments<S: StorageEngine>(
    storage: &S,
    jobs: IndexJobs,
) -> NodeDbResult<()> {
    if let Some(ext) = storage.as_vector_segment_ext() {
        for (name, sources) in jobs.vectors {
            if let Some((dim, vectors, surrogates)) = segment_payload(storage, &name, sources)
                .await
                .map_err(NodeDbError::from)?
            {
                ext.write_vector_segment(&name, dim, &vectors, &surrogates)
                    .await
                    .map_err(NodeDbError::from)?;
            }
        }
    }
    if let Some(ext) = storage.as_graph_segment_ext() {
        for (name, checkpoint) in jobs.csr {
            ext.write_graph_segment(&name, &checkpoint)
                .await
                .map_err(NodeDbError::from)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PagedbStorageMem,
        config::LiteConfig,
        engine::vector::pagedb_backing::PagedbBacking,
        error::LiteError,
        nodedb::NodeDbLite,
        storage::{
            engine::{KvPair, WriteOp},
            graph_segment_ext::GraphSegmentExt,
            vector_segment_ext::VectorSegmentExt,
        },
    };
    use async_trait::async_trait;
    use nodedb_client::NodeDb;
    use nodedb_types::{Namespace, id::NodeId};
    use std::sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    };

    const VECTOR_WRITE: u8 = 1;
    const GRAPH_WRITE: u8 = 2;
    const VECTOR_READ: u8 = 3;
    const FTS_WRITE: u8 = 4;
    const SPATIAL_WRITE: u8 = 5;

    struct InjectedStorage {
        inner: PagedbStorageMem,
        mode: Arc<AtomicU8>,
    }

    fn injected(operation: &str) -> LiteError {
        LiteError::Storage {
            detail: format!("injected {operation} error"),
        }
    }

    #[async_trait]
    impl StorageEngine for InjectedStorage {
        async fn get(&self, ns: Namespace, key: &[u8]) -> Result<Option<Vec<u8>>, LiteError> {
            self.inner.get(ns, key).await
        }
        async fn put(&self, ns: Namespace, key: &[u8], value: &[u8]) -> Result<(), LiteError> {
            match (self.mode.load(Ordering::Relaxed), ns) {
                (FTS_WRITE, Namespace::FtsDeletePending) => {
                    return Err(injected("fts outbound write"));
                }
                (SPATIAL_WRITE, Namespace::SpatialDeletePending) => {
                    return Err(injected("spatial outbound write"));
                }
                _ => {}
            }
            self.inner.put(ns, key, value).await
        }
        async fn delete(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
            self.inner.delete(ns, key).await
        }
        async fn scan_prefix(
            &self,
            ns: Namespace,
            prefix: &[u8],
        ) -> Result<Vec<KvPair>, LiteError> {
            if self.mode.load(Ordering::Relaxed) == VECTOR_READ
                && ns == Namespace::Vector
                && prefix.starts_with(b"vr:")
            {
                return Err(injected("durable vector read"));
            }
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
        fn as_vector_segment_ext(&self) -> Option<&dyn VectorSegmentExt> {
            Some(self)
        }
        fn as_graph_segment_ext(&self) -> Option<&dyn GraphSegmentExt> {
            Some(self)
        }
    }

    #[async_trait]
    impl VectorSegmentExt for InjectedStorage {
        async fn write_vector_segment(
            &self,
            name: &str,
            dim: usize,
            vectors: &[Vec<f32>],
            ids: &[u64],
        ) -> Result<(), LiteError> {
            if self.mode.load(Ordering::Relaxed) == VECTOR_WRITE {
                return Err(injected("vector segment write"));
            }
            self.inner
                .write_vector_segment(name, dim, vectors, ids)
                .await
        }
        async fn open_vector_segment(
            &self,
            name: &str,
        ) -> Result<Option<PagedbBacking>, LiteError> {
            self.inner.open_vector_segment(name).await
        }
        async fn delete_vector_segment(&self, name: &str) -> Result<(), LiteError> {
            self.inner.delete_vector_segment(name).await
        }
    }

    #[async_trait]
    impl GraphSegmentExt for InjectedStorage {
        async fn write_graph_segment(&self, name: &str, bytes: &[u8]) -> Result<(), LiteError> {
            if self.mode.load(Ordering::Relaxed) == GRAPH_WRITE {
                return Err(injected("graph segment write"));
            }
            self.inner.write_graph_segment(name, bytes).await
        }
        async fn open_graph_segment(&self, name: &str) -> Result<Option<Box<[u8]>>, LiteError> {
            self.inner.open_graph_segment(name).await
        }
        async fn delete_graph_segment(&self, name: &str) -> Result<(), LiteError> {
            self.inner.delete_graph_segment(name).await
        }
    }

    async fn open() -> (Arc<NodeDbLite<InjectedStorage>>, Arc<AtomicU8>) {
        let mode = Arc::new(AtomicU8::new(0));
        let storage = InjectedStorage {
            inner: PagedbStorageMem::open_in_memory().await.unwrap(),
            mode: Arc::clone(&mode),
        };
        let db = NodeDbLite::open_with_config(
            storage,
            LiteConfig {
                auto_flush_ms: 0,
                auto_compact_ms: 0,
                ..LiteConfig::default()
            },
        )
        .await
        .unwrap();
        (db, mode)
    }

    #[tokio::test]
    async fn flush_reports_segment_and_durable_read_errors_and_retries() {
        for (mode_value, expected) in [
            (VECTOR_WRITE, "vector segment write"),
            (GRAPH_WRITE, "graph segment write"),
            (VECTOR_READ, "durable vector read"),
        ] {
            let (db, mode) = open().await;
            db.vector_insert("vectors", "a", &[1.0, 2.0], None)
                .await
                .unwrap();
            db.graph_insert_edge(
                "graph",
                &NodeId::try_new("a").unwrap(),
                &NodeId::try_new("b").unwrap(),
                "LINK",
                None,
            )
            .await
            .unwrap();
            mode.store(mode_value, Ordering::Relaxed);
            assert!(db.flush().await.unwrap_err().to_string().contains(expected));
            mode.store(0, Ordering::Relaxed);
            db.flush().await.unwrap();
            assert!(
                db.storage
                    .inner
                    .open_vector_segment("vectors")
                    .await
                    .unwrap()
                    .is_some()
            );
            assert!(
                db.storage
                    .inner
                    .open_graph_segment("graph")
                    .await
                    .unwrap()
                    .is_some()
            );
        }
    }

    #[tokio::test]
    async fn flush_reports_missing_bound_rows() {
        let (db, _) = open().await;
        db.vector_insert("vectors", "a", &[1.0, 2.0], None)
            .await
            .unwrap();
        crate::engine::vector::durable::remove(&db.storage.inner, "vectors", "a")
            .await
            .unwrap();
        let error = db.flush().await.unwrap_err();
        assert!(error.to_string().contains("no durable row"));
    }

    #[tokio::test]
    async fn flush_reports_outbound_staging_errors() {
        for (mode_value, expected) in [
            (FTS_WRITE, "fts outbound write"),
            (SPATIAL_WRITE, "spatial outbound write"),
        ] {
            let (db, mode) = open().await;
            if mode_value == FTS_WRITE {
                db.fts_outbound.as_ref().unwrap().stage_delete("docs", "a");
            } else {
                db.spatial_outbound
                    .as_ref()
                    .unwrap()
                    .stage_delete("places", "location", "a");
            }
            mode.store(mode_value, Ordering::Relaxed);
            assert!(db.flush().await.unwrap_err().to_string().contains(expected));
        }
    }
}
