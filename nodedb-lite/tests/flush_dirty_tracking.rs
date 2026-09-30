// SPDX-License-Identifier: Apache-2.0

//! Flush writes a derived HNSW, CSR, sparse, spatial, or full-text artifact
//! only when it changed.
//!
//! `flush()` runs on a timer. Rewriting every collection's graph checkpoint,
//! the vector id-map, every vector segment, every CSR adjacency checkpoint,
//! every sparse index, every spatial R-tree and doc-map, and every full-text
//! index on each tick costs their full size whether or not anything changed:
//! an idle store with a large vector collection rewrote hundreds of megabytes
//! per tick.
//!
//! These assert on per-artifact write counters and on the keys a probing
//! storage wrapper saw, not on elapsed time, so they fail for the reason they
//! name on any machine.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nodedb_client::NodeDb;
use nodedb_lite::engine::vector::pagedb_backing::PagedbBacking;
use nodedb_lite::error::LiteError;
use nodedb_lite::nodedb::{FTS_SURROGATES_KEY, FlushArtifact, ID_MAP_KEY, spatial_rtree_key};
use nodedb_lite::storage::array_segment_ext::ArraySegmentExt;
use nodedb_lite::storage::columnar_segment_ext::ColumnarSegmentExt;
use nodedb_lite::storage::engine::{CompactionOutcome, KvPair};
use nodedb_lite::storage::fts_segment_ext::FtsSegmentExt;
use nodedb_lite::storage::graph_segment_ext::GraphSegmentExt;
use nodedb_lite::storage::spatial_segment_ext::SpatialSegmentExt;
use nodedb_lite::storage::vector_segment_ext::VectorSegmentExt;
use nodedb_lite::{
    Encryption, LiteConfig, NodeDbLite, PagedbStorageDefault, StorageEngine, WriteOp,
};
use nodedb_types::document::Document;
use nodedb_types::geometry::Geometry;
use nodedb_types::id::NodeId;
use nodedb_types::text_search::TextSearchParams;
use nodedb_types::value::Value;
use nodedb_types::{BoundingBox, Namespace};
use tokio::sync::Notify;

const DIM: usize = 8;
const ALPHA: &str = "alpha";
const BETA: &str = "beta";
const GRAPH_A: &str = "graph_a";
const GRAPH_B: &str = "graph_b";

/// Upper bound on waiting for a flush to reach a gated write. Reaching it
/// means the flush never planned the write the test is racing.
const GATE_TIMEOUT: Duration = Duration::from_secs(30);

/// A pseudo-random vector per seed. The default metric is cosine, so evenly
/// spaced vectors would point in nearly one direction and tie; these do not.
fn vector(seed: usize) -> Vec<f32> {
    let mut state = (seed as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..DIM)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 1_000) as f32 / 1_000.0 - 0.5
        })
        .collect()
}

/// Ids that sort in insertion order, so a vector segment built from the
/// durable rows lines up with the graph's node order.
fn doc_id(i: usize) -> String {
    format!("d{i:03}")
}

fn manual_flush_config() -> LiteConfig {
    LiteConfig {
        auto_flush_ms: 0,
        ..LiteConfig::default()
    }
}

/// Open with auto-flush disabled so every flush in the test is one we asked for.
async fn open_manual_flush(path: &std::path::Path) -> Arc<NodeDbLite<PagedbStorageDefault>> {
    let storage = PagedbStorageDefault::open(path, Encryption::Plaintext)
        .await
        .expect("open storage");
    NodeDbLite::open_with_config(storage, manual_flush_config())
        .await
        .expect("open db")
}

/// Open over a [`ProbeStorage`], returning the probe that controls it.
async fn open_probed(path: &std::path::Path) -> (Arc<NodeDbLite<ProbeStorage>>, Arc<Probe>) {
    open_probed_with(path, manual_flush_config()).await
}

/// Open over a [`ProbeStorage`] with `config`, returning the probe that
/// controls it.
async fn open_probed_with(
    path: &std::path::Path,
    config: LiteConfig,
) -> (Arc<NodeDbLite<ProbeStorage>>, Arc<Probe>) {
    let inner = PagedbStorageDefault::open(path, Encryption::Plaintext)
        .await
        .expect("open storage");
    let probe = Arc::new(Probe::default());
    let storage = ProbeStorage {
        inner,
        probe: Arc::clone(&probe),
    };
    let db = NodeDbLite::open_with_config(storage, config)
        .await
        .expect("open db");
    (db, probe)
}

async fn insert<S: StorageEngine>(
    db: &NodeDbLite<S>,
    collection: &str,
    range: std::ops::Range<usize>,
) {
    for i in range {
        db.vector_insert(collection, &doc_id(i), &vector(i), None)
            .await
            .expect("vector_insert");
    }
}

/// Successful `(graph, segment)` writes for `collection`.
fn writes<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> (u64, u64) {
    (
        db.flush_artifact_write_count(FlushArtifact::HnswGraph, collection),
        db.flush_artifact_write_count(FlushArtifact::VectorSegment, collection),
    )
}

fn id_map_writes<S: StorageEngine>(db: &NodeDbLite<S>) -> u64 {
    db.flush_artifact_write_count(FlushArtifact::HnswIdMap, ID_MAP_KEY)
}

fn any_dirty<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> bool {
    db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, collection)
        || db.flush_artifact_is_dirty(FlushArtifact::VectorSegment, collection)
        || db.flush_artifact_is_dirty(FlushArtifact::HnswIdMap, ID_MAP_KEY)
}

/// The top-`k` ids for each query vector, in result order.
async fn search_ids<S: StorageEngine>(
    db: &NodeDbLite<S>,
    collection: &str,
    seeds: &[usize],
    k: usize,
) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for &seed in seeds {
        let hits = db
            .vector_search(collection, &vector(seed), k, None, None)
            .await
            .expect("vector_search");
        out.push(hits.into_iter().map(|h| h.id).collect());
    }
    out
}

// ---------------------------------------------------------------------------
// Probe storage
// ---------------------------------------------------------------------------

/// A one-shot gate that parks the first matching write until released.
#[derive(Default)]
struct Gate {
    armed: Mutex<Option<String>>,
    entered: Notify,
    release: Notify,
}

impl Gate {
    fn arm(&self, collection: &str) {
        *self.armed.lock().unwrap() = Some(collection.to_string());
    }

    /// Park here when armed for `collection`. `Notify` keeps a permit, so
    /// neither side can miss the other's signal.
    async fn pass(&self, collection: &str) {
        let hit = {
            let mut armed = self.armed.lock().unwrap();
            if armed.as_deref() == Some(collection) {
                *armed = None;
                true
            } else {
                false
            }
        };
        if hit {
            self.entered.notify_one();
            self.release.notified().await;
        }
    }

    async fn wait_entered(&self) {
        tokio::time::timeout(GATE_TIMEOUT, self.entered.notified())
            .await
            .expect("the flush never reached the gated write");
    }

    fn release(&self) {
        self.release.notify_one();
    }
}

/// The `(namespace, key)` of each op in one committed batch.
type BatchKeys = Vec<(Namespace, Vec<u8>)>;

/// Test controls shared between a [`ProbeStorage`] and the test body.
#[derive(Default)]
struct Probe {
    /// `(namespace, key)` of every committed put.
    puts: Mutex<Vec<(Namespace, Vec<u8>)>>,
    /// Makes every vector segment write fail while set.
    fail_segment_writes: AtomicBool,
    /// Parks a batch that puts `hnsw:<collection>`, before it is applied.
    graph_batch_gate: Gate,
    /// Parks a vector segment write for a collection, before it is applied.
    segment_gate: Gate,
    /// Collection of every successful CSR graph segment write.
    graph_segment_writes: Mutex<Vec<String>>,
    /// Makes every CSR graph segment write fail while set.
    fail_graph_segment_writes: AtomicBool,
    /// Parks a CSR graph segment write for a collection, before it is applied.
    graph_segment_gate: Gate,
    /// `(namespace, key)` of every committed delete, in a batch or on its own.
    deletes: Mutex<Vec<(Namespace, Vec<u8>)>>,
    /// Every committed non-empty batch, as the `(namespace, key)` of each op.
    batches: Mutex<Vec<BatchKeys>>,
    /// `<collection>/<field>` of every successful spatial segment write.
    spatial_segment_writes: Mutex<Vec<String>>,
    /// Makes every spatial segment write fail while set.
    fail_spatial_segment_writes: AtomicBool,
    /// Parks a spatial segment write for a collection, before it is applied.
    spatial_segment_gate: Gate,
    /// Collection of every successful vector segment write.
    vector_segment_writes: Mutex<Vec<String>>,
    /// Index key of every successful FTS segment write.
    fts_segment_writes: Mutex<Vec<String>>,
    /// Index key of every successful FTS segment delete.
    fts_segment_deletes: Mutex<Vec<String>>,
    /// Makes every FTS segment write fail while set.
    fail_fts_segment_writes: AtomicBool,
    /// Parks an FTS segment write for an index key, before it is applied.
    fts_segment_gate: Gate,
}

impl Probe {
    fn take_puts(&self) -> Vec<(Namespace, Vec<u8>)> {
        std::mem::take(&mut *self.puts.lock().unwrap())
    }

    fn take_graph_segment_writes(&self) -> Vec<String> {
        std::mem::take(&mut *self.graph_segment_writes.lock().unwrap())
    }

    fn take_deletes(&self) -> Vec<(Namespace, Vec<u8>)> {
        std::mem::take(&mut *self.deletes.lock().unwrap())
    }

    fn take_batches(&self) -> Vec<BatchKeys> {
        std::mem::take(&mut *self.batches.lock().unwrap())
    }

    /// Successful spatial segment writes, sorted.
    fn take_spatial_segment_writes(&self) -> Vec<String> {
        let mut writes = std::mem::take(&mut *self.spatial_segment_writes.lock().unwrap());
        writes.sort();
        writes
    }

    fn take_vector_segment_writes(&self) -> Vec<String> {
        std::mem::take(&mut *self.vector_segment_writes.lock().unwrap())
    }

    /// Successful FTS segment writes, sorted.
    fn take_fts_segment_writes(&self) -> Vec<String> {
        let mut writes = std::mem::take(&mut *self.fts_segment_writes.lock().unwrap());
        writes.sort();
        writes
    }

    /// Successful FTS segment deletes, sorted.
    fn take_fts_segment_deletes(&self) -> Vec<String> {
        let mut deletes = std::mem::take(&mut *self.fts_segment_deletes.lock().unwrap());
        deletes.sort();
        deletes
    }
}

/// Puts a flush makes for HNSW data or the meta entries this unit tracks.
fn hnsw_and_meta_puts(puts: &[(Namespace, Vec<u8>)]) -> Vec<String> {
    puts.iter()
        .filter(|(ns, key)| {
            (*ns == Namespace::Vector && key.starts_with(b"hnsw"))
                || (*ns == Namespace::Meta
                    && (key.as_slice() == b"meta:hnsw_collections"
                        || key.as_slice() == b"meta:last_flushed_mid"))
        })
        .map(|(ns, key)| format!("{ns:?}/{}", String::from_utf8_lossy(key)))
        .collect()
}

/// Puts a flush makes for a CSR blob or the CSR collection list.
fn csr_puts(puts: &[(Namespace, Vec<u8>)]) -> Vec<String> {
    puts.iter()
        .filter(|(ns, key)| {
            (*ns == Namespace::Graph && key.starts_with(b"csr:"))
                || (*ns == Namespace::Meta && key.as_slice() == b"meta:csr_collections")
        })
        .map(|(ns, key)| format!("{ns:?}/{}", String::from_utf8_lossy(key)))
        .collect()
}

/// Pagedb storage that lets a test observe, block, or fail flush writes.
struct ProbeStorage {
    inner: PagedbStorageDefault,
    probe: Arc<Probe>,
}

impl ProbeStorage {
    fn record(&self, ns: Namespace, key: &[u8]) {
        self.probe.puts.lock().unwrap().push((ns, key.to_vec()));
    }

    fn segments(&self) -> Result<&dyn VectorSegmentExt, LiteError> {
        self.inner
            .as_vector_segment_ext()
            .ok_or_else(|| LiteError::Storage {
                detail: "probe: inner storage has no vector segment support".into(),
            })
    }

    fn graph_segments(&self) -> Result<&dyn GraphSegmentExt, LiteError> {
        self.inner
            .as_graph_segment_ext()
            .ok_or_else(|| LiteError::Storage {
                detail: "probe: inner storage has no graph segment support".into(),
            })
    }

    fn spatial_segments(&self) -> Result<&dyn SpatialSegmentExt, LiteError> {
        self.inner
            .as_spatial_segment_ext()
            .ok_or_else(|| LiteError::Storage {
                detail: "probe: inner storage has no spatial segment support".into(),
            })
    }

    fn fts_segments(&self) -> Result<&dyn FtsSegmentExt, LiteError> {
        self.inner
            .as_fts_segment_ext()
            .ok_or_else(|| LiteError::Storage {
                detail: "probe: inner storage has no FTS segment support".into(),
            })
    }
}

#[async_trait::async_trait]
impl StorageEngine for ProbeStorage {
    async fn get(&self, ns: Namespace, key: &[u8]) -> Result<Option<Vec<u8>>, LiteError> {
        self.inner.get(ns, key).await
    }

    async fn put(&self, ns: Namespace, key: &[u8], value: &[u8]) -> Result<(), LiteError> {
        self.inner.put(ns, key, value).await?;
        self.record(ns, key);
        Ok(())
    }

    async fn delete(&self, ns: Namespace, key: &[u8]) -> Result<(), LiteError> {
        self.inner.delete(ns, key).await?;
        self.probe.deletes.lock().unwrap().push((ns, key.to_vec()));
        Ok(())
    }

    async fn scan_prefix(&self, ns: Namespace, prefix: &[u8]) -> Result<Vec<KvPair>, LiteError> {
        self.inner.scan_prefix(ns, prefix).await
    }

    async fn batch_write(&self, ops: &[WriteOp]) -> Result<(), LiteError> {
        let graph_collections: Vec<String> = ops
            .iter()
            .filter_map(|op| match op {
                WriteOp::Put { ns, key, value: _ } if *ns == Namespace::Vector => key
                    .strip_prefix(b"hnsw:")
                    .and_then(|c| std::str::from_utf8(c).ok())
                    .map(str::to_string),
                WriteOp::Put {
                    ns: _,
                    key: _,
                    value: _,
                }
                | WriteOp::Delete { ns: _, key: _ } => None,
            })
            .collect();
        for collection in &graph_collections {
            self.probe.graph_batch_gate.pass(collection).await;
        }
        self.inner.batch_write(ops).await?;
        let mut batch: Vec<(Namespace, Vec<u8>)> = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                WriteOp::Put { ns, key, value: _ } => {
                    self.record(*ns, key);
                    batch.push((*ns, key.clone()));
                }
                WriteOp::Delete { ns, key } => {
                    self.probe.deletes.lock().unwrap().push((*ns, key.clone()));
                    batch.push((*ns, key.clone()));
                }
            }
        }
        if !batch.is_empty() {
            self.probe.batches.lock().unwrap().push(batch);
        }
        Ok(())
    }

    async fn count(&self, ns: Namespace) -> Result<u64, LiteError> {
        self.inner.count(ns).await
    }

    async fn compact(&self) -> Result<CompactionOutcome, LiteError> {
        self.inner.compact().await
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

    fn as_array_segment_ext(&self) -> Option<&dyn ArraySegmentExt> {
        self.inner.as_array_segment_ext()
    }

    fn as_fts_segment_ext(&self) -> Option<&dyn FtsSegmentExt> {
        Some(self)
    }

    fn as_columnar_segment_ext(&self) -> Option<&dyn ColumnarSegmentExt> {
        self.inner.as_columnar_segment_ext()
    }

    fn as_graph_segment_ext(&self) -> Option<&dyn GraphSegmentExt> {
        Some(self)
    }

    fn as_spatial_segment_ext(&self) -> Option<&dyn SpatialSegmentExt> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl VectorSegmentExt for ProbeStorage {
    async fn write_vector_segment(
        &self,
        collection_name: &str,
        dim: usize,
        vectors: &[Vec<f32>],
        surrogate_ids: &[u64],
    ) -> Result<(), LiteError> {
        self.probe.segment_gate.pass(collection_name).await;
        if self.probe.fail_segment_writes.load(Ordering::SeqCst) {
            return Err(LiteError::Storage {
                detail: format!("probe: injected segment write failure for {collection_name}"),
            });
        }
        self.segments()?
            .write_vector_segment(collection_name, dim, vectors, surrogate_ids)
            .await?;
        self.probe
            .vector_segment_writes
            .lock()
            .unwrap()
            .push(collection_name.to_string());
        Ok(())
    }

    async fn open_vector_segment(
        &self,
        collection_name: &str,
    ) -> Result<Option<PagedbBacking>, LiteError> {
        self.segments()?.open_vector_segment(collection_name).await
    }

    async fn delete_vector_segment(&self, collection_name: &str) -> Result<(), LiteError> {
        self.segments()?
            .delete_vector_segment(collection_name)
            .await
    }
}

#[async_trait::async_trait]
impl GraphSegmentExt for ProbeStorage {
    async fn write_graph_segment(&self, collection: &str, bytes: &[u8]) -> Result<(), LiteError> {
        self.probe.graph_segment_gate.pass(collection).await;
        if self.probe.fail_graph_segment_writes.load(Ordering::SeqCst) {
            return Err(LiteError::Storage {
                detail: format!("probe: injected graph segment write failure for {collection}"),
            });
        }
        self.graph_segments()?
            .write_graph_segment(collection, bytes)
            .await?;
        self.probe
            .graph_segment_writes
            .lock()
            .unwrap()
            .push(collection.to_string());
        Ok(())
    }

    async fn open_graph_segment(&self, collection: &str) -> Result<Option<Box<[u8]>>, LiteError> {
        self.graph_segments()?.open_graph_segment(collection).await
    }

    async fn delete_graph_segment(&self, collection: &str) -> Result<(), LiteError> {
        self.graph_segments()?
            .delete_graph_segment(collection)
            .await
    }
}

#[async_trait::async_trait]
impl SpatialSegmentExt for ProbeStorage {
    async fn write_spatial_segment(
        &self,
        collection: &str,
        field: &str,
        bytes: &[u8],
    ) -> Result<(), LiteError> {
        self.probe.spatial_segment_gate.pass(collection).await;
        if self
            .probe
            .fail_spatial_segment_writes
            .load(Ordering::SeqCst)
        {
            return Err(LiteError::Storage {
                detail: format!(
                    "probe: injected spatial segment write failure for {collection}/{field}"
                ),
            });
        }
        self.spatial_segments()?
            .write_spatial_segment(collection, field, bytes)
            .await?;
        self.probe
            .spatial_segment_writes
            .lock()
            .unwrap()
            .push(format!("{collection}/{field}"));
        Ok(())
    }

    async fn open_spatial_segment(
        &self,
        collection: &str,
        field: &str,
    ) -> Result<Option<Box<[u8]>>, LiteError> {
        self.spatial_segments()?
            .open_spatial_segment(collection, field)
            .await
    }

    async fn delete_spatial_segment(&self, collection: &str, field: &str) -> Result<(), LiteError> {
        self.spatial_segments()?
            .delete_spatial_segment(collection, field)
            .await
    }
}

#[async_trait::async_trait]
impl FtsSegmentExt for ProbeStorage {
    async fn write_fts_segment(&self, index_key: &str, bytes: &[u8]) -> Result<(), LiteError> {
        self.probe.fts_segment_gate.pass(index_key).await;
        if self.probe.fail_fts_segment_writes.load(Ordering::SeqCst) {
            return Err(LiteError::Storage {
                detail: format!("probe: injected FTS segment write failure for {index_key}"),
            });
        }
        self.fts_segments()?
            .write_fts_segment(index_key, bytes)
            .await?;
        self.probe
            .fts_segment_writes
            .lock()
            .unwrap()
            .push(index_key.to_string());
        Ok(())
    }

    async fn open_fts_segment(&self, index_key: &str) -> Result<Option<Box<[u8]>>, LiteError> {
        self.fts_segments()?.open_fts_segment(index_key).await
    }

    async fn delete_fts_segment(&self, index_key: &str) -> Result<(), LiteError> {
        self.fts_segments()?.delete_fts_segment(index_key).await?;
        self.probe
            .fts_segment_deletes
            .lock()
            .unwrap()
            .push(index_key.to_string());
        Ok(())
    }

    async fn list_fts_segments(&self, prefix: &str) -> Result<Vec<String>, LiteError> {
        self.fts_segments()?.list_fts_segments(prefix).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A second flush with no mutation in between writes no HNSW graph, id-map,
/// vector segment, or tracked meta entry.
#[tokio::test]
async fn second_flush_without_mutation_writes_no_hnsw_artifact_or_meta() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("idle.pagedb")).await;

    insert(&db, ALPHA, 0..4).await;
    db.flush().await.expect("first flush");

    let after_first = (writes(&db, ALPHA), id_map_writes(&db));
    assert_eq!(
        after_first,
        ((1, 1), 1),
        "the first flush writes the graph, the segment, and the id-map once"
    );
    assert!(
        !any_dirty(&db, ALPHA),
        "a completed flush leaves nothing dirty"
    );
    probe.take_puts();

    db.flush().await.expect("idle flush");

    assert_eq!(
        (writes(&db, ALPHA), id_map_writes(&db)),
        after_first,
        "an idle flush must not rewrite any HNSW artifact"
    );
    assert_eq!(
        hnsw_and_meta_puts(&probe.take_puts()),
        Vec::<String>::new(),
        "an idle flush must not put HNSW keys or unchanged meta entries"
    );
}

/// An insert into one collection makes the next flush rewrite that
/// collection's artifacts and the shared id-map only.
#[tokio::test]
async fn insert_rewrites_only_the_touched_collections_artifacts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_manual_flush(&dir.path().join("insert.pagedb")).await;

    insert(&db, ALPHA, 0..3).await;
    insert(&db, BETA, 100..103).await;
    db.flush().await.expect("seed flush");
    let (alpha_before, beta_before, id_map_before) =
        (writes(&db, ALPHA), writes(&db, BETA), id_map_writes(&db));

    insert(&db, ALPHA, 3..4).await;
    assert!(db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, ALPHA));
    assert!(!db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, BETA));
    db.flush().await.expect("flush after insert");

    assert_eq!(
        writes(&db, ALPHA),
        (alpha_before.0 + 1, alpha_before.1 + 1),
        "the touched collection's graph and segment are rewritten"
    );
    assert_eq!(
        writes(&db, BETA),
        beta_before,
        "an untouched collection's graph and segment are not rewritten"
    );
    assert_eq!(
        id_map_writes(&db),
        id_map_before + 1,
        "the id-map gained a binding"
    );
}

/// A delete makes the next flush rewrite the collection's graph and segment.
#[tokio::test]
async fn delete_rewrites_the_collections_graph_and_segment() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_manual_flush(&dir.path().join("delete.pagedb")).await;

    insert(&db, ALPHA, 0..4).await;
    db.flush().await.expect("seed flush");
    let before = writes(&db, ALPHA);

    db.vector_delete(ALPHA, &doc_id(1))
        .await
        .expect("vector_delete");
    db.flush().await.expect("flush after delete");

    assert_eq!(
        writes(&db, ALPHA),
        (before.0 + 1, before.1 + 1),
        "the tombstone and the removed durable row are both written"
    );
    assert!(!any_dirty(&db, ALPHA));
}

/// A mutation that lands while the graph batch is in flight keeps the graph
/// and the id-map dirty, and the next flush writes them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutation_during_the_graph_write_leaves_the_graph_dirty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("graph_race.pagedb")).await;
    insert(&db, ALPHA, 0..3).await;

    probe.graph_batch_gate.arm(ALPHA);
    let flushing = tokio::spawn({
        let db = Arc::clone(&db);
        async move { db.flush().await }
    });
    probe.graph_batch_gate.wait_entered().await;
    // The flush has captured the graph and id-map generations and is parked
    // before its batch lands.
    insert(&db, ALPHA, 3..4).await;
    probe.graph_batch_gate.release();
    flushing.await.expect("join").expect("racing flush");

    assert_eq!(
        writes(&db, ALPHA).0,
        1,
        "the in-flight graph write itself succeeded"
    );
    assert!(
        db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, ALPHA),
        "an insert after the capture must keep the graph dirty"
    );
    assert!(
        db.flush_artifact_is_dirty(FlushArtifact::HnswIdMap, ID_MAP_KEY),
        "a bind after the capture must keep the id-map dirty"
    );

    db.flush().await.expect("follow-up flush");
    assert_eq!(
        writes(&db, ALPHA).0,
        2,
        "the next flush writes the graph again"
    );
    assert!(!any_dirty(&db, ALPHA));
}

/// A durable row written after the flush read the rows for a segment keeps
/// that segment dirty, and the next flush writes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutation_during_the_segment_write_leaves_the_segment_dirty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("segment_race.pagedb")).await;
    insert(&db, ALPHA, 0..3).await;

    probe.segment_gate.arm(ALPHA);
    let flushing = tokio::spawn({
        let db = Arc::clone(&db);
        async move { db.flush().await }
    });
    probe.segment_gate.wait_entered().await;
    // The segment payload has been read from the durable rows; this row is
    // not in it.
    insert(&db, ALPHA, 3..4).await;
    probe.segment_gate.release();
    flushing.await.expect("join").expect("racing flush");

    assert_eq!(
        writes(&db, ALPHA).1,
        1,
        "the in-flight segment write itself succeeded"
    );
    assert!(
        db.flush_artifact_is_dirty(FlushArtifact::VectorSegment, ALPHA),
        "a durable row written after the capture must keep the segment dirty"
    );

    db.flush().await.expect("follow-up flush");
    assert_eq!(
        writes(&db, ALPHA).1,
        2,
        "the next flush writes the segment again"
    );
    assert!(!any_dirty(&db, ALPHA));
}

/// A segment write that fails leaves the segment dirty, and the next flush
/// retries it.
#[tokio::test]
async fn failed_segment_write_is_retried_by_the_next_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("segment_fail.pagedb")).await;
    insert(&db, ALPHA, 0..3).await;

    probe.fail_segment_writes.store(true, Ordering::SeqCst);
    db.flush()
        .await
        .expect("a segment write error is logged, not returned");
    assert_eq!(
        writes(&db, ALPHA),
        (1, 0),
        "the graph landed and the segment did not"
    );
    assert!(
        db.flush_artifact_is_dirty(FlushArtifact::VectorSegment, ALPHA),
        "a failed segment write must not be recorded as flushed"
    );
    assert!(!db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, ALPHA));

    probe.fail_segment_writes.store(false, Ordering::SeqCst);
    db.flush().await.expect("retry flush");
    assert_eq!(
        writes(&db, ALPHA),
        (1, 1),
        "the retry writes only the segment that failed"
    );
    assert!(!any_dirty(&db, ALPHA));
}

/// Search results and document ids survive a close and reopen that follows a
/// run of dirty-aware flushes, including idle ones.
#[tokio::test]
async fn reopen_after_dirty_aware_flushes_returns_identical_search_results() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reopen.pagedb");
    let seeds: Vec<usize> = vec![0, 2, 3, 4, 5];
    let live: Vec<String> = seeds.iter().map(|&i| doc_id(i)).collect();

    let before = {
        let db = open_manual_flush(&path).await;
        insert(&db, ALPHA, 0..4).await;
        db.flush().await.expect("flush 1");
        insert(&db, ALPHA, 4..6).await;
        db.vector_delete(ALPHA, &doc_id(1))
            .await
            .expect("vector_delete");
        db.flush().await.expect("flush 2");
        db.flush().await.expect("idle flush");
        let before = search_ids(&db, ALPHA, &seeds, seeds.len()).await;
        db.shutdown().await;
        before
    };

    let db = open_manual_flush(&path).await;
    let after = search_ids(&db, ALPHA, &seeds, seeds.len()).await;

    assert_eq!(after, before, "search results must survive the reopen");
    for (seed, hits) in seeds.iter().zip(&after) {
        assert_eq!(
            hits.first(),
            Some(&doc_id(*seed)),
            "an exact query finds its own document"
        );
        assert!(
            hits.iter().all(|id| live.contains(id)),
            "every hit maps to a live document id, not a slot number: {hits:?}"
        );
    }
}

/// Artifacts restored from a valid checkpoint, segment, and id-map start
/// clean, so the first flush after a reopen writes none of them.
#[tokio::test]
async fn artifacts_restored_from_valid_stored_forms_start_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("clean_open.pagedb");
    {
        let db = open_manual_flush(&path).await;
        insert(&db, ALPHA, 0..4).await;
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let db = open_manual_flush(&path).await;
    assert!(
        !any_dirty(&db, ALPHA),
        "every restored artifact starts clean"
    );
    db.flush().await.expect("first flush after reopen");
    assert_eq!((writes(&db, ALPHA), id_map_writes(&db)), ((0, 0), 0));
}

/// An index rebuilt from durable rows at open, because no checkpoint was
/// ever written, is written by the first flush.
#[tokio::test]
async fn artifacts_rebuilt_at_open_are_written_by_the_first_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rebuilt_open.pagedb");
    {
        let db = open_manual_flush(&path).await;
        insert(&db, ALPHA, 0..4).await;
        // No flush: only the durable rows reach disk.
        db.shutdown().await;
    }

    let db = open_manual_flush(&path).await;
    assert!(db.flush_artifact_is_dirty(FlushArtifact::HnswGraph, ALPHA));
    assert!(db.flush_artifact_is_dirty(FlushArtifact::VectorSegment, ALPHA));
    assert!(db.flush_artifact_is_dirty(FlushArtifact::HnswIdMap, ID_MAP_KEY));

    db.flush().await.expect("first flush after rebuild");
    assert_eq!((writes(&db, ALPHA), id_map_writes(&db)), ((1, 1), 1));
    assert!(!any_dirty(&db, ALPHA));
}

/// `flush_full()` writes every tracked artifact even when all are clean.
#[tokio::test]
async fn flush_full_writes_every_artifact_even_when_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("full.pagedb")).await;
    insert(&db, ALPHA, 0..3).await;
    db.flush().await.expect("flush");
    db.flush().await.expect("idle flush");
    assert!(!any_dirty(&db, ALPHA));
    let (alpha_before, id_map_before) = (writes(&db, ALPHA), id_map_writes(&db));
    probe.take_puts();

    db.flush_full().await.expect("flush_full");

    assert_eq!(writes(&db, ALPHA), (alpha_before.0 + 1, alpha_before.1 + 1));
    assert_eq!(id_map_writes(&db), id_map_before + 1);
    let puts = hnsw_and_meta_puts(&probe.take_puts());
    for expected in [
        "Vector/hnsw:alpha",
        "Vector/hnsw_id_map",
        "Meta/meta:hnsw_collections",
        "Meta/meta:last_flushed_mid",
    ] {
        assert!(
            puts.iter().any(|p| p == expected),
            "flush_full must put {expected}; saw {puts:?}"
        );
    }
    assert!(!any_dirty(&db, ALPHA));
}

// ---------------------------------------------------------------------------
// CSR graph
// ---------------------------------------------------------------------------

async fn add_edge<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str, src: &str, dst: &str) {
    let src = NodeId::try_new(src).expect("node id");
    let dst = NodeId::try_new(dst).expect("node id");
    db.graph_insert_edge(collection, &src, &dst, "LINK", None)
        .await
        .expect("graph_insert_edge");
}

/// Successful CSR checkpoint writes for `collection`.
fn csr_writes<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> u64 {
    db.flush_artifact_write_count(FlushArtifact::CsrGraph, collection)
}

fn csr_dirty<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> bool {
    db.flush_artifact_is_dirty(FlushArtifact::CsrGraph, collection)
}

/// Every `(src, label, dst)` edge reachable in one hop from `start`, sorted.
async fn one_hop_edges<S: StorageEngine>(
    db: &NodeDbLite<S>,
    collection: &str,
    start: &str,
) -> Vec<(String, String, String)> {
    let start = NodeId::try_new(start).expect("node id");
    let subgraph = db
        .graph_traverse(collection, &start, 1, None)
        .await
        .expect("graph_traverse");
    let mut edges: Vec<(String, String, String)> = subgraph
        .edges
        .into_iter()
        .map(|e| {
            (
                e.from.as_str().to_string(),
                e.label,
                e.to.as_str().to_string(),
            )
        })
        .collect();
    edges.sort();
    edges
}

/// A second flush with no graph mutation in between writes no CSR segment,
/// no CSR blob, and no CSR collection list.
#[tokio::test]
async fn second_flush_without_mutation_writes_no_csr_segment_or_meta() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("csr_idle.pagedb")).await;

    add_edge(&db, GRAPH_A, "a", "b").await;
    db.flush().await.expect("first flush");
    assert_eq!(
        csr_writes(&db, GRAPH_A),
        1,
        "the first flush writes the graph"
    );
    assert_eq!(probe.take_graph_segment_writes(), vec![GRAPH_A.to_string()]);
    assert!(
        !csr_dirty(&db, GRAPH_A),
        "a completed flush leaves the graph clean"
    );
    probe.take_puts();

    db.flush().await.expect("idle flush");

    assert_eq!(
        csr_writes(&db, GRAPH_A),
        1,
        "an idle flush must not rewrite the graph"
    );
    assert_eq!(
        probe.take_graph_segment_writes(),
        Vec::<String>::new(),
        "an idle flush must not write a CSR segment"
    );
    assert_eq!(
        csr_puts(&probe.take_puts()),
        Vec::<String>::new(),
        "an idle flush must not put a CSR blob or an unchanged collection list"
    );
}

/// An edge added to one graph collection makes the next flush rewrite that
/// collection's checkpoint only.
#[tokio::test]
async fn edge_rewrites_only_the_touched_collections_csr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_manual_flush(&dir.path().join("csr_edge.pagedb")).await;

    add_edge(&db, GRAPH_A, "a", "b").await;
    add_edge(&db, GRAPH_B, "x", "y").await;
    db.flush().await.expect("seed flush");
    let (a_before, b_before) = (csr_writes(&db, GRAPH_A), csr_writes(&db, GRAPH_B));

    add_edge(&db, GRAPH_A, "b", "c").await;
    assert!(csr_dirty(&db, GRAPH_A));
    assert!(!csr_dirty(&db, GRAPH_B));
    db.flush().await.expect("flush after edge");

    assert_eq!(
        csr_writes(&db, GRAPH_A),
        a_before + 1,
        "the touched collection's graph is rewritten"
    );
    assert_eq!(
        csr_writes(&db, GRAPH_B),
        b_before,
        "an untouched collection's graph is not rewritten"
    );
}

/// An edge that lands while the collection's CSR segment write is in flight
/// keeps the graph dirty, and the next flush writes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_during_the_csr_segment_write_leaves_the_graph_dirty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("csr_race.pagedb")).await;
    add_edge(&db, GRAPH_A, "a", "b").await;

    probe.graph_segment_gate.arm(GRAPH_A);
    let flushing = tokio::spawn({
        let db = Arc::clone(&db);
        async move { db.flush().await }
    });
    probe.graph_segment_gate.wait_entered().await;
    // The flush has captured the generation and serialized the checkpoint;
    // this edge is not in it.
    add_edge(&db, GRAPH_A, "b", "c").await;
    probe.graph_segment_gate.release();
    flushing.await.expect("join").expect("racing flush");

    assert_eq!(
        csr_writes(&db, GRAPH_A),
        1,
        "the in-flight segment write itself succeeded"
    );
    assert!(
        csr_dirty(&db, GRAPH_A),
        "an edge added after the capture must keep the graph dirty"
    );

    db.flush().await.expect("follow-up flush");
    assert_eq!(
        csr_writes(&db, GRAPH_A),
        2,
        "the next flush writes the graph again"
    );
    assert!(!csr_dirty(&db, GRAPH_A));
}

/// A CSR segment write that fails leaves the graph dirty, and the next flush
/// retries it.
#[tokio::test]
async fn failed_csr_segment_write_is_retried_by_the_next_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("csr_fail.pagedb")).await;
    add_edge(&db, GRAPH_A, "a", "b").await;

    probe
        .fail_graph_segment_writes
        .store(true, Ordering::SeqCst);
    db.flush()
        .await
        .expect("a CSR segment write error is logged, not returned");
    assert_eq!(csr_writes(&db, GRAPH_A), 0, "the segment did not land");
    assert!(
        csr_dirty(&db, GRAPH_A),
        "a failed segment write must not be recorded as flushed"
    );

    probe
        .fail_graph_segment_writes
        .store(false, Ordering::SeqCst);
    db.flush().await.expect("retry flush");
    assert_eq!(csr_writes(&db, GRAPH_A), 1, "the retry writes the segment");
    assert_eq!(probe.take_graph_segment_writes(), vec![GRAPH_A.to_string()]);
    assert!(!csr_dirty(&db, GRAPH_A));
}

/// A graph restored from its stored checkpoint keeps its neighbors and starts
/// clean, so the first flush after a reopen does not rewrite it.
#[tokio::test]
async fn csr_restored_from_its_checkpoint_keeps_neighbors_and_starts_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("csr_reopen.pagedb");

    let before = {
        let db = open_manual_flush(&path).await;
        add_edge(&db, GRAPH_A, "a", "b").await;
        add_edge(&db, GRAPH_A, "a", "c").await;
        add_edge(&db, GRAPH_B, "x", "y").await;
        db.flush().await.expect("flush");
        db.flush().await.expect("idle flush");
        let before = (
            one_hop_edges(&db, GRAPH_A, "a").await,
            one_hop_edges(&db, GRAPH_B, "x").await,
        );
        db.shutdown().await;
        before
    };
    assert_eq!(before.0.len(), 2, "both edges out of `a` are reachable");

    let db = open_manual_flush(&path).await;
    for collection in [GRAPH_A, GRAPH_B] {
        assert!(
            !csr_dirty(&db, collection),
            "{collection} restored from its checkpoint starts clean"
        );
    }
    let after = (
        one_hop_edges(&db, GRAPH_A, "a").await,
        one_hop_edges(&db, GRAPH_B, "x").await,
    );
    assert_eq!(after, before, "neighbors must survive the reopen");

    db.flush().await.expect("first flush after reopen");
    assert_eq!(
        (csr_writes(&db, GRAPH_A), csr_writes(&db, GRAPH_B)),
        (0, 0),
        "the first flush after a reopen must not rewrite a restored graph"
    );
}

/// A graph rebuilt from its edge documents at open, because no checkpoint
/// was listed, is written by the first flush.
#[tokio::test]
async fn csr_rebuilt_at_open_is_written_by_the_first_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("csr_rebuilt.pagedb");
    {
        let db = open_manual_flush(&path).await;
        add_edge(&db, GRAPH_A, "a", "b").await;
        db.flush().await.expect("flush");
        db.shutdown().await;
    }
    // Without the collection list, open finds no checkpoint and rebuilds the
    // graph from the edge documents.
    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .expect("open storage");
        storage
            .delete(Namespace::Meta, b"meta:csr_collections")
            .await
            .expect("delete the CSR collection list");
    }

    let db = open_manual_flush(&path).await;
    assert_eq!(
        one_hop_edges(&db, GRAPH_A, "a").await,
        vec![("a".to_string(), "LINK".to_string(), "b".to_string())],
        "the rebuild restores the edge"
    );
    assert!(
        csr_dirty(&db, GRAPH_A),
        "a graph rebuilt at open has no stored form matching it"
    );

    db.flush().await.expect("first flush after rebuild");
    assert_eq!(csr_writes(&db, GRAPH_A), 1);
    assert!(!csr_dirty(&db, GRAPH_A));
}

/// `flush_full()` writes every CSR collection and the collection list even
/// when all are clean.
#[tokio::test]
async fn flush_full_writes_every_csr_collection_and_the_meta() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("csr_full.pagedb")).await;
    add_edge(&db, GRAPH_A, "a", "b").await;
    add_edge(&db, GRAPH_B, "x", "y").await;
    db.flush().await.expect("flush");
    db.flush().await.expect("idle flush");
    assert!(!csr_dirty(&db, GRAPH_A) && !csr_dirty(&db, GRAPH_B));
    let before = (csr_writes(&db, GRAPH_A), csr_writes(&db, GRAPH_B));
    probe.take_puts();
    probe.take_graph_segment_writes();

    db.flush_full().await.expect("flush_full");

    assert_eq!(
        (csr_writes(&db, GRAPH_A), csr_writes(&db, GRAPH_B)),
        (before.0 + 1, before.1 + 1)
    );
    let mut segments = probe.take_graph_segment_writes();
    segments.sort();
    assert_eq!(segments, vec![GRAPH_A.to_string(), GRAPH_B.to_string()]);
    let puts = csr_puts(&probe.take_puts());
    assert!(
        puts.iter().any(|p| p == "Meta/meta:csr_collections"),
        "flush_full must put the CSR collection list; saw {puts:?}"
    );
    assert!(!csr_dirty(&db, GRAPH_A) && !csr_dirty(&db, GRAPH_B));
}

// ---------------------------------------------------------------------------
// Sparse vectors
// ---------------------------------------------------------------------------

const SPARSE: &str = "sparse_docs";
const DOCS: &str = "docs";

/// The key a sparse index is tracked under: `<collection>:<field>`.
fn sparse_key(collection: &str, field: &str) -> String {
    format!("{collection}:{field}")
}

fn sparse_writes<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str, field: &str) -> u64 {
    db.flush_artifact_write_count(FlushArtifact::SparseIndex, &sparse_key(collection, field))
}

fn sparse_dirty<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str, field: &str) -> bool {
    db.flush_artifact_is_dirty(FlushArtifact::SparseIndex, &sparse_key(collection, field))
}

fn sparse_insert<S: StorageEngine>(db: &NodeDbLite<S>, field: &str, doc_id: &str, dim: u32) {
    db.sparse_insert(SPARSE, field, doc_id, &[(dim, 1.0)])
        .expect("sparse_insert");
}

/// A document whose `emb` field holds a sparse-vector literal.
async fn put_sparse_document<S: StorageEngine>(db: &NodeDbLite<S>, doc_id: &str) {
    let mut doc = Document::new(doc_id);
    doc.set("emb", Value::String("{1: 0.5}".into()));
    db.document_put(DOCS, doc).await.expect("document_put");
}

/// Puts a flush makes for a sparse index or the sparse index list.
fn sparse_puts(puts: &[(Namespace, Vec<u8>)]) -> Vec<String> {
    let mut out: Vec<String> = puts
        .iter()
        .filter(|(ns, key)| *ns == Namespace::Vector && key.starts_with(b"sparse:"))
        .map(|(ns, key)| format!("{ns:?}/{}", String::from_utf8_lossy(key)))
        .collect();
    out.sort();
    out
}

/// A second flush with no sparse mutation in between writes no sparse key.
#[tokio::test]
async fn second_flush_without_mutation_writes_no_sparse_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("sparse_idle.pagedb")).await;
    sparse_insert(&db, "a", "d1", 1);

    db.flush().await.expect("first flush");
    assert_eq!(
        sparse_writes(&db, SPARSE, "a"),
        1,
        "the first flush writes it"
    );
    assert!(!sparse_dirty(&db, SPARSE, "a"));
    probe.take_puts();

    db.flush().await.expect("idle flush");

    assert_eq!(
        sparse_puts(&probe.take_puts()),
        Vec::<String>::new(),
        "an idle flush must not put a sparse index or an unchanged index list"
    );
    assert_eq!(sparse_writes(&db, SPARSE, "a"), 1);
}

/// An insert into one sparse field makes the next flush rewrite that field's
/// index only. The index list is unchanged, so it is not rewritten either.
#[tokio::test]
async fn sparse_insert_rewrites_only_its_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("sparse_insert.pagedb")).await;
    sparse_insert(&db, "a", "d1", 1);
    sparse_insert(&db, "b", "d1", 2);
    db.flush().await.expect("seed flush");
    let (a_before, b_before) = (
        sparse_writes(&db, SPARSE, "a"),
        sparse_writes(&db, SPARSE, "b"),
    );
    probe.take_puts();

    sparse_insert(&db, "a", "d2", 3);
    assert!(sparse_dirty(&db, SPARSE, "a"));
    assert!(!sparse_dirty(&db, SPARSE, "b"));
    db.flush().await.expect("flush after insert");

    assert_eq!(
        sparse_puts(&probe.take_puts()),
        vec![format!("Vector/sparse:{SPARSE}:a:docs")],
        "only the touched index is put"
    );
    assert_eq!(sparse_writes(&db, SPARSE, "a"), a_before + 1);
    assert_eq!(sparse_writes(&db, SPARSE, "b"), b_before);
}

/// Every document write reconciles the collection's sparse indexes, removing
/// the document from each field it does not carry. A write whose document is
/// in none of them changes nothing, so no index turns dirty.
#[tokio::test]
async fn non_sparse_document_write_does_not_dirty_sparse() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("sparse_precision.pagedb")).await;
    put_sparse_document(&db, "d1").await;
    db.flush().await.expect("seed flush");
    assert!(!sparse_dirty(&db, DOCS, "emb"));
    let before = sparse_writes(&db, DOCS, "emb");
    probe.take_puts();

    let mut plain = Document::new("d2");
    plain.set("title", Value::String("plain text".into()));
    db.document_put(DOCS, plain).await.expect("document_put");

    assert!(
        !sparse_dirty(&db, DOCS, "emb"),
        "a document with no sparse field must not dirty the collection's sparse index"
    );
    db.flush().await.expect("flush after the plain write");
    assert_eq!(sparse_puts(&probe.take_puts()), Vec::<String>::new());
    assert_eq!(sparse_writes(&db, DOCS, "emb"), before);
}

/// A sparse index restored from its stored blob starts clean, so the first
/// flush after a reopen does not rewrite it.
#[tokio::test]
async fn sparse_restored_from_its_checkpoint_starts_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sparse_reopen.pagedb");
    {
        let db = open_manual_flush(&path).await;
        put_sparse_document(&db, "d1").await;
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let (db, probe) = open_probed(&path).await;
    assert!(
        !sparse_dirty(&db, DOCS, "emb"),
        "an index whose blob decoded starts clean"
    );
    let hits = db
        .sparse_search(DOCS, "emb", &[(1, 1.0)], 10)
        .expect("sparse_search");
    assert_eq!(hits.len(), 1, "the restored index still answers");

    db.flush().await.expect("first flush after reopen");
    assert_eq!(sparse_writes(&db, DOCS, "emb"), 0);
    let puts = sparse_puts(&probe.take_puts());
    assert!(
        !puts.iter().any(|p| p.ends_with(":docs")),
        "the first flush after a reopen must not rewrite a restored index; saw {puts:?}"
    );
}

/// `flush_full()` writes every sparse index and the index list even when
/// all are clean.
#[tokio::test]
async fn flush_full_writes_every_sparse_index_and_the_list() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("sparse_full.pagedb")).await;
    sparse_insert(&db, "a", "d1", 1);
    sparse_insert(&db, "b", "d1", 2);
    db.flush().await.expect("flush");
    db.flush().await.expect("idle flush");
    assert!(!sparse_dirty(&db, SPARSE, "a") && !sparse_dirty(&db, SPARSE, "b"));
    let before = (
        sparse_writes(&db, SPARSE, "a"),
        sparse_writes(&db, SPARSE, "b"),
    );
    probe.take_puts();

    db.flush_full().await.expect("flush_full");

    assert_eq!(
        (
            sparse_writes(&db, SPARSE, "a"),
            sparse_writes(&db, SPARSE, "b")
        ),
        (before.0 + 1, before.1 + 1)
    );
    assert_eq!(
        sparse_puts(&probe.take_puts()),
        vec![
            "Vector/sparse:_indices".to_string(),
            format!("Vector/sparse:{SPARSE}:a:docs"),
            format!("Vector/sparse:{SPARSE}:b:docs"),
        ]
    );
}

// ---------------------------------------------------------------------------
// Spatial
// ---------------------------------------------------------------------------

const PLACES: &str = "places";
const OTHER: &str = "other";
const LOC: &str = "loc";
const GEO: &str = "geo";

fn point(i: usize) -> Geometry {
    Geometry::point(i as f64 * 0.01, i as f64 * 0.01)
}

fn rtree_writes<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str, field: &str) -> u64 {
    db.flush_artifact_write_count(
        FlushArtifact::SpatialRtree,
        &spatial_rtree_key(collection, field),
    )
}

fn rtree_dirty<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str, field: &str) -> bool {
    db.flush_artifact_is_dirty(
        FlushArtifact::SpatialRtree,
        &spatial_rtree_key(collection, field),
    )
}

fn docmap_writes<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> u64 {
    db.flush_artifact_write_count(FlushArtifact::SpatialDocMap, collection)
}

fn docmap_dirty<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> bool {
    db.flush_artifact_is_dirty(FlushArtifact::SpatialDocMap, collection)
}

/// Puts a flush makes under `Namespace::Spatial`, sorted.
fn spatial_puts(puts: &[(Namespace, Vec<u8>)]) -> Vec<String> {
    let mut out: Vec<String> = puts
        .iter()
        .filter(|(ns, _)| *ns == Namespace::Spatial)
        .map(|(ns, key)| format!("{ns:?}/{}", String::from_utf8_lossy(key)))
        .collect();
    out.sort();
    out
}

/// Committed non-empty batches carrying any `Namespace::Spatial` op.
fn spatial_batches(batches: &[BatchKeys]) -> usize {
    batches
        .iter()
        .filter(|batch| batch.iter().any(|(ns, _)| *ns == Namespace::Spatial))
        .count()
}

/// A second flush with no spatial mutation in between writes no spatial key,
/// runs no spatial batch, and writes no R-tree segment.
#[tokio::test]
async fn second_flush_without_mutation_writes_no_spatial_put_or_segment() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("spatial_idle.pagedb")).await;
    db.spatial_insert(PLACES, LOC, "p1", &point(1));

    db.flush().await.expect("first flush");
    assert_eq!(
        (rtree_writes(&db, PLACES, LOC), docmap_writes(&db, PLACES)),
        (1, 1),
        "the first flush writes the tree and its doc-map"
    );
    assert!(!rtree_dirty(&db, PLACES, LOC) && !docmap_dirty(&db, PLACES));
    probe.take_puts();
    probe.take_batches();
    probe.take_spatial_segment_writes();

    db.flush().await.expect("idle flush");

    assert_eq!(spatial_puts(&probe.take_puts()), Vec::<String>::new());
    assert_eq!(
        spatial_batches(&probe.take_batches()),
        0,
        "an idle flush must not run a spatial batch"
    );
    assert_eq!(probe.take_spatial_segment_writes(), Vec::<String>::new());
    assert_eq!(
        (rtree_writes(&db, PLACES, LOC), docmap_writes(&db, PLACES)),
        (1, 1)
    );
}

/// An insert makes the next flush write that tree's segment, its
/// collection's doc-map, and `spatial:_next_id`, and nothing for another
/// collection. The set of trees is unchanged, so the catalog is not written.
#[tokio::test]
async fn spatial_insert_writes_its_tree_its_doc_map_and_next_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("spatial_insert.pagedb")).await;
    db.spatial_insert(PLACES, LOC, "p1", &point(1));
    db.spatial_insert(OTHER, LOC, "o1", &point(2));
    db.flush().await.expect("seed flush");
    probe.take_puts();
    probe.take_spatial_segment_writes();

    db.spatial_insert(PLACES, LOC, "p2", &point(3));
    assert!(rtree_dirty(&db, PLACES, LOC) && docmap_dirty(&db, PLACES));
    assert!(!rtree_dirty(&db, OTHER, LOC) && !docmap_dirty(&db, OTHER));
    db.flush().await.expect("flush after insert");

    assert_eq!(
        probe.take_spatial_segment_writes(),
        vec![format!("{PLACES}/{LOC}")]
    );
    assert_eq!(
        spatial_puts(&probe.take_puts()),
        vec![
            "Spatial/spatial:_next_id".to_string(),
            format!("Spatial/spatial:{PLACES}:{LOC}:docmap"),
        ]
    );
    assert_eq!(
        (rtree_writes(&db, OTHER, LOC), docmap_writes(&db, OTHER)),
        (1, 1),
        "the untouched collection is not rewritten"
    );
}

/// `TRUNCATE` empties every tree of the collection, so every one of its
/// fields and its doc-map turn dirty. Another collection stays clean.
#[tokio::test]
async fn truncate_dirties_every_spatial_field_of_the_collection() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("spatial_truncate.pagedb")).await;
    // `TRUNCATE` addresses a document collection, so give it one row.
    let mut row = Document::new("row1");
    row.set("n", Value::Integer(1));
    db.document_put(PLACES, row).await.expect("document_put");
    db.spatial_insert(PLACES, LOC, "p1", &point(1));
    db.spatial_insert(PLACES, GEO, "g1", &point(2));
    db.spatial_insert(OTHER, LOC, "o1", &point(3));
    db.flush().await.expect("seed flush");
    probe.take_spatial_segment_writes();

    db.execute_sql(&format!("TRUNCATE {PLACES}"), &[])
        .await
        .expect("TRUNCATE");

    assert!(rtree_dirty(&db, PLACES, LOC), "every field is emptied");
    assert!(rtree_dirty(&db, PLACES, GEO), "every field is emptied");
    assert!(docmap_dirty(&db, PLACES));
    assert!(!rtree_dirty(&db, OTHER, LOC) && !docmap_dirty(&db, OTHER));

    db.flush().await.expect("flush after truncate");
    assert_eq!(
        probe.take_spatial_segment_writes(),
        vec![format!("{PLACES}/{GEO}"), format!("{PLACES}/{LOC}")]
    );
    assert!(!rtree_dirty(&db, PLACES, LOC) && !rtree_dirty(&db, PLACES, GEO));
    assert!(!docmap_dirty(&db, PLACES));
}

/// An insert that lands while the tree's segment write is in flight keeps
/// the tree and its doc-map dirty, and the next flush writes the tree again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_during_the_spatial_segment_write_leaves_the_tree_dirty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("spatial_race.pagedb")).await;
    db.spatial_insert(PLACES, LOC, "p1", &point(1));

    probe.spatial_segment_gate.arm(PLACES);
    let flushing = tokio::spawn({
        let db = Arc::clone(&db);
        async move { db.flush().await }
    });
    probe.spatial_segment_gate.wait_entered().await;
    // The flush has captured the tree's generation and serialized it, and
    // its doc-map batch has committed. This entry is in neither.
    db.spatial_insert(PLACES, LOC, "p2", &point(2));
    probe.spatial_segment_gate.release();
    flushing.await.expect("join").expect("racing flush");

    assert_eq!(
        rtree_writes(&db, PLACES, LOC),
        1,
        "the in-flight segment write itself succeeded"
    );
    assert!(
        rtree_dirty(&db, PLACES, LOC),
        "an insert after the capture must keep the tree dirty"
    );
    assert!(
        docmap_dirty(&db, PLACES),
        "an insert after the doc-map batch must keep the doc-map dirty"
    );

    db.flush().await.expect("follow-up flush");
    assert_eq!(rtree_writes(&db, PLACES, LOC), 2);
    assert!(!rtree_dirty(&db, PLACES, LOC) && !docmap_dirty(&db, PLACES));
}

/// A spatial segment write that fails leaves the tree dirty, and the next
/// flush retries it.
#[tokio::test]
async fn failed_spatial_segment_write_is_retried_by_the_next_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("spatial_fail.pagedb")).await;
    db.spatial_insert(PLACES, LOC, "p1", &point(1));

    probe
        .fail_spatial_segment_writes
        .store(true, Ordering::SeqCst);
    db.flush()
        .await
        .expect("a spatial segment write error is logged, not returned");
    assert_eq!(
        rtree_writes(&db, PLACES, LOC),
        0,
        "the segment did not land"
    );
    assert!(
        rtree_dirty(&db, PLACES, LOC),
        "a failed segment write must not be recorded as flushed"
    );
    assert_eq!(docmap_writes(&db, PLACES), 1, "the doc-map batch landed");
    assert!(!docmap_dirty(&db, PLACES));

    probe
        .fail_spatial_segment_writes
        .store(false, Ordering::SeqCst);
    db.flush().await.expect("retry flush");
    assert_eq!(
        rtree_writes(&db, PLACES, LOC),
        1,
        "the retry writes the tree"
    );
    assert_eq!(
        probe.take_spatial_segment_writes(),
        vec![format!("{PLACES}/{LOC}")]
    );
    assert_eq!(
        docmap_writes(&db, PLACES),
        1,
        "the clean doc-map is not rewritten"
    );
    assert!(!rtree_dirty(&db, PLACES, LOC));
}

/// Trees and doc-maps restored from their stored forms start clean, so the
/// first flush after a reopen rewrites neither.
#[tokio::test]
async fn spatial_restored_from_its_checkpoint_starts_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("spatial_reopen.pagedb");
    {
        let db = open_manual_flush(&path).await;
        db.spatial_insert(PLACES, LOC, "p1", &point(1));
        db.spatial_insert(OTHER, LOC, "o1", &point(2));
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let (db, probe) = open_probed(&path).await;
    for collection in [PLACES, OTHER] {
        assert!(
            !rtree_dirty(&db, collection, LOC) && !docmap_dirty(&db, collection),
            "{collection} restored from its checkpoint starts clean"
        );
    }
    let world = BoundingBox::new(-180.0, -90.0, 180.0, 90.0);
    assert_eq!(db.spatial_search_bbox(PLACES, LOC, &world).len(), 1);

    db.flush().await.expect("first flush after reopen");
    assert_eq!(probe.take_spatial_segment_writes(), Vec::<String>::new());
    let puts = spatial_puts(&probe.take_puts());
    assert!(
        !puts.iter().any(|p| p.ends_with(":docmap")),
        "the first flush after a reopen must not rewrite a restored doc-map; saw {puts:?}"
    );
    for collection in [PLACES, OTHER] {
        assert_eq!(
            (
                rtree_writes(&db, collection, LOC),
                docmap_writes(&db, collection)
            ),
            (0, 0)
        );
    }
}

/// `flush_full()` writes every spatial tree, every doc-map, and both catalog
/// entries even when all are clean.
#[tokio::test]
async fn flush_full_writes_every_spatial_tree_doc_map_and_catalog_entry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("spatial_full.pagedb")).await;
    db.spatial_insert(PLACES, LOC, "p1", &point(1));
    db.spatial_insert(OTHER, LOC, "o1", &point(2));
    db.flush().await.expect("flush");
    db.flush().await.expect("idle flush");
    probe.take_puts();
    probe.take_spatial_segment_writes();

    db.flush_full().await.expect("flush_full");

    assert_eq!(
        probe.take_spatial_segment_writes(),
        vec![format!("{OTHER}/{LOC}"), format!("{PLACES}/{LOC}")]
    );
    assert_eq!(
        spatial_puts(&probe.take_puts()),
        vec![
            "Spatial/spatial:_collections".to_string(),
            "Spatial/spatial:_next_id".to_string(),
            format!("Spatial/spatial:{OTHER}:{LOC}:docmap"),
            format!("Spatial/spatial:{PLACES}:{LOC}:docmap"),
        ]
    );
    for collection in [PLACES, OTHER] {
        assert_eq!(
            (
                rtree_writes(&db, collection, LOC),
                docmap_writes(&db, collection)
            ),
            (2, 2)
        );
    }
}

// ---------------------------------------------------------------------------
// Full-text search
// ---------------------------------------------------------------------------

const NOTES: &str = "notes";
const MEMOS: &str = "memos";

/// The key a collection's whole-document text index is tracked under.
fn fts_key(collection: &str) -> String {
    format!("{collection}:_doc")
}

fn fts_writes<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> u64 {
    db.flush_artifact_write_count(FlushArtifact::FtsIndex, &fts_key(collection))
}

fn fts_dirty<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str) -> bool {
    db.flush_artifact_is_dirty(FlushArtifact::FtsIndex, &fts_key(collection))
}

fn surrogate_writes<S: StorageEngine>(db: &NodeDbLite<S>) -> u64 {
    db.flush_artifact_write_count(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY)
}

fn surrogates_dirty<S: StorageEngine>(db: &NodeDbLite<S>) -> bool {
    db.flush_artifact_is_dirty(FlushArtifact::FtsSurrogates, FTS_SURROGATES_KEY)
}

/// Write a document whose only field holds `text`.
async fn put_text<S: StorageEngine>(db: &NodeDbLite<S>, collection: &str, id: &str, text: &str) {
    let mut doc = Document::new(id);
    doc.set("body", Value::String(text.into()));
    db.document_put(collection, doc)
        .await
        .expect("document_put");
}

/// Ids of the documents a text query matches, sorted.
async fn text_hits<S: StorageEngine>(
    db: &NodeDbLite<S>,
    collection: &str,
    query: &str,
) -> Vec<String> {
    let hits = db
        .text_search(collection, "", query, 10, TextSearchParams::default(), None)
        .await
        .expect("text_search");
    let mut ids: Vec<String> = hits.into_iter().map(|h| h.id).collect();
    ids.sort();
    ids
}

/// `<namespace>/<key>` of each op, in order.
fn op_names(ops: &[(Namespace, Vec<u8>)]) -> Vec<String> {
    ops.iter()
        .map(|(ns, key)| format!("{ns:?}/{}", String::from_utf8_lossy(key)))
        .collect()
}

/// Ops under `Namespace::Fts`, sorted.
fn fts_ops(ops: &[(Namespace, Vec<u8>)]) -> Vec<String> {
    let mut out: Vec<String> = ops
        .iter()
        .filter(|(ns, _)| *ns == Namespace::Fts)
        .map(|(ns, key)| format!("{ns:?}/{}", String::from_utf8_lossy(key)))
        .collect();
    out.sort();
    out
}

/// Committed non-empty batches carrying any `Namespace::Fts` op.
fn fts_batches(batches: &[BatchKeys]) -> usize {
    batches
        .iter()
        .filter(|batch| batch.iter().any(|(ns, _)| *ns == Namespace::Fts))
        .count()
}

/// A second flush with no text mutation in between writes no FTS key, runs
/// no FTS batch, and writes no FTS segment.
#[tokio::test]
async fn second_flush_without_mutation_writes_no_fts_key_or_segment() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("fts_idle.pagedb")).await;
    put_text(&db, NOTES, "n1", "phoenix rises").await;

    db.flush().await.expect("first flush");
    assert_eq!(
        (fts_writes(&db, NOTES), surrogate_writes(&db)),
        (1, 1),
        "the first flush writes the index and the surrogate map"
    );
    assert!(!fts_dirty(&db, NOTES) && !surrogates_dirty(&db));
    probe.take_puts();
    probe.take_deletes();
    probe.take_batches();
    probe.take_fts_segment_writes();

    db.flush().await.expect("idle flush");

    assert_eq!(fts_ops(&probe.take_puts()), Vec::<String>::new());
    assert_eq!(fts_ops(&probe.take_deletes()), Vec::<String>::new());
    assert_eq!(
        fts_batches(&probe.take_batches()),
        0,
        "an idle flush must not run an FTS batch"
    );
    assert_eq!(probe.take_fts_segment_writes(), Vec::<String>::new());
    assert_eq!((fts_writes(&db, NOTES), surrogate_writes(&db)), (1, 1));
}

/// A text write to one collection makes the next flush rewrite that
/// collection's index only. The set of indexes is unchanged, so the index
/// list is not written either.
#[tokio::test]
async fn text_write_rewrites_only_its_collections_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("fts_touch.pagedb")).await;
    put_text(&db, NOTES, "n1", "phoenix rises").await;
    put_text(&db, MEMOS, "m1", "ember glows").await;
    db.flush().await.expect("seed flush");
    let memos_before = fts_writes(&db, MEMOS);
    probe.take_puts();
    probe.take_fts_segment_writes();

    put_text(&db, NOTES, "n2", "ashes settle").await;
    assert!(fts_dirty(&db, NOTES));
    assert!(!fts_dirty(&db, MEMOS));
    db.flush().await.expect("flush after the text write");

    assert_eq!(probe.take_fts_segment_writes(), vec![fts_key(NOTES)]);
    let puts = fts_ops(&probe.take_puts());
    assert!(
        !puts.iter().any(|p| p.contains(&fts_key(MEMOS))),
        "an untouched collection's index must not be put; saw {puts:?}"
    );
    assert!(
        !puts.iter().any(|p| p == "Fts/fts:_collections"),
        "an unchanged index list must not be put; saw {puts:?}"
    );
    assert_eq!(fts_writes(&db, MEMOS), memos_before);
}

/// Updating a document that already has a surrogate allocates none, so the
/// surrogate map is not rewritten. The index itself is.
#[tokio::test]
async fn updating_a_document_does_not_rewrite_the_surrogate_map() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("fts_update.pagedb")).await;
    put_text(&db, NOTES, "n1", "phoenix rises").await;
    db.flush().await.expect("seed flush");
    let (index_before, surrogates_before) = (fts_writes(&db, NOTES), surrogate_writes(&db));
    probe.take_puts();

    put_text(&db, NOTES, "n1", "phoenix sleeps").await;
    assert!(fts_dirty(&db, NOTES), "the document's terms changed");
    assert!(
        !surrogates_dirty(&db),
        "an update of a known document must not dirty the surrogate map"
    );
    db.flush().await.expect("flush after the update");

    let puts = fts_ops(&probe.take_puts());
    assert!(
        !puts.iter().any(|p| p == "Fts/fts:_surrogates"),
        "the surrogate map must not be put; saw {puts:?}"
    );
    assert_eq!(surrogate_writes(&db), surrogates_before);
    assert_eq!(fts_writes(&db, NOTES), index_before + 1);
}

/// A text write that lands while the index's segment write is in flight
/// keeps the index dirty, and the next flush writes it again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_write_during_the_fts_segment_write_leaves_the_index_dirty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("fts_race.pagedb")).await;
    put_text(&db, NOTES, "n1", "phoenix rises").await;

    probe.fts_segment_gate.arm(&fts_key(NOTES));
    let flushing = tokio::spawn({
        let db = Arc::clone(&db);
        async move { db.flush().await }
    });
    probe.fts_segment_gate.wait_entered().await;
    // The flush has captured the index's generation and serialized it; this
    // document is not in it.
    put_text(&db, NOTES, "n2", "ashes settle").await;
    probe.fts_segment_gate.release();
    flushing.await.expect("join").expect("racing flush");

    assert_eq!(
        fts_writes(&db, NOTES),
        1,
        "the in-flight index write itself succeeded"
    );
    assert!(
        fts_dirty(&db, NOTES),
        "a write after the capture must keep the index dirty"
    );

    db.flush().await.expect("follow-up flush");
    assert_eq!(
        fts_writes(&db, NOTES),
        2,
        "the next flush writes the index again"
    );
    assert!(!fts_dirty(&db, NOTES));
}

/// An FTS segment write that fails leaves the index dirty and keeps its doc
/// lengths out of the batch, and the next flush retries it.
#[tokio::test]
async fn failed_fts_segment_write_is_retried_by_the_next_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("fts_fail.pagedb")).await;
    put_text(&db, NOTES, "n1", "phoenix rises").await;

    probe.fail_fts_segment_writes.store(true, Ordering::SeqCst);
    db.flush()
        .await
        .expect("an FTS segment write error is logged, not returned");
    assert_eq!(fts_writes(&db, NOTES), 0, "the segment did not land");
    assert!(
        fts_dirty(&db, NOTES),
        "a failed segment write must not be recorded as flushed"
    );
    let puts = fts_ops(&probe.take_puts());
    assert!(
        !puts
            .iter()
            .any(|p| *p == format!("Fts/fts:{}:doclens", fts_key(NOTES))),
        "the doc lengths of an index whose segment failed stay out of the batch; saw {puts:?}"
    );
    assert_eq!(surrogate_writes(&db), 1, "the surrogate map landed");

    probe.fail_fts_segment_writes.store(false, Ordering::SeqCst);
    db.flush().await.expect("retry flush");
    assert_eq!(fts_writes(&db, NOTES), 1, "the retry writes the index");
    assert_eq!(probe.take_fts_segment_writes(), vec![fts_key(NOTES)]);
    assert_eq!(
        surrogate_writes(&db),
        1,
        "the clean surrogate map is not rewritten"
    );
    assert!(!fts_dirty(&db, NOTES));
}

/// An index and surrogate map restored from their stored forms start clean,
/// so the first flush after a reopen writes no FTS key or segment.
#[tokio::test]
async fn fts_restored_from_its_checkpoint_starts_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fts_reopen.pagedb");
    {
        let db = open_manual_flush(&path).await;
        put_text(&db, NOTES, "n1", "phoenix rises").await;
        put_text(&db, MEMOS, "m1", "ember glows").await;
        db.flush().await.expect("flush");
        db.shutdown().await;
    }

    let (db, probe) = open_probed(&path).await;
    for collection in [NOTES, MEMOS] {
        assert!(
            !fts_dirty(&db, collection),
            "{collection} restored from its checkpoint starts clean"
        );
    }
    assert!(
        !surrogates_dirty(&db),
        "a decoded surrogate map starts clean"
    );
    assert_eq!(
        text_hits(&db, NOTES, "phoenix").await,
        vec!["n1".to_string()]
    );
    probe.take_puts();
    probe.take_fts_segment_writes();

    db.flush().await.expect("first flush after reopen");
    assert_eq!(probe.take_fts_segment_writes(), Vec::<String>::new());
    assert_eq!(
        fts_ops(&probe.take_puts()),
        Vec::<String>::new(),
        "the first flush after a reopen must not rewrite restored FTS state"
    );
    assert_eq!(
        (
            fts_writes(&db, NOTES),
            fts_writes(&db, MEMOS),
            surrogate_writes(&db)
        ),
        (0, 0, 0)
    );
}

/// An index rebuilt from the documents at open, because no index list was
/// stored, is written by the first flush.
#[tokio::test]
async fn fts_rebuilt_at_open_is_written_by_the_first_flush() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fts_rebuilt.pagedb");
    {
        let db = open_manual_flush(&path).await;
        put_text(&db, NOTES, "n1", "phoenix rises").await;
        db.flush().await.expect("flush");
        db.shutdown().await;
    }
    // Without the index list, open finds no checkpoint and rebuilds the
    // index from the stored documents.
    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .expect("open storage");
        storage
            .delete(Namespace::Fts, b"fts:_collections")
            .await
            .expect("delete the FTS index list");
    }

    let db = open_manual_flush(&path).await;
    assert_eq!(
        text_hits(&db, NOTES, "phoenix").await,
        vec!["n1".to_string()],
        "the rebuild restores the document"
    );
    assert!(
        fts_dirty(&db, NOTES),
        "an index rebuilt at open has no stored form matching it"
    );
    assert!(surrogates_dirty(&db));

    db.flush().await.expect("first flush after rebuild");
    assert_eq!((fts_writes(&db, NOTES), surrogate_writes(&db)), (1, 1));
    assert!(!fts_dirty(&db, NOTES) && !surrogates_dirty(&db));
}

/// `flush_full()` writes every FTS index, the surrogate map, and the index
/// list even when all are clean.
#[tokio::test]
async fn flush_full_writes_every_fts_index_and_the_surrogate_map() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed(&dir.path().join("fts_full.pagedb")).await;
    put_text(&db, NOTES, "n1", "phoenix rises").await;
    put_text(&db, MEMOS, "m1", "ember glows").await;
    db.flush().await.expect("flush");
    db.flush().await.expect("idle flush");
    let before = (
        fts_writes(&db, NOTES),
        fts_writes(&db, MEMOS),
        surrogate_writes(&db),
    );
    probe.take_puts();
    probe.take_fts_segment_writes();

    db.flush_full().await.expect("flush_full");

    assert_eq!(
        probe.take_fts_segment_writes(),
        vec![fts_key(MEMOS), fts_key(NOTES)]
    );
    let puts = fts_ops(&probe.take_puts());
    for expected in [
        "Fts/fts:_collections".to_string(),
        "Fts/fts:_surrogates".to_string(),
        format!("Fts/fts:{}:doclens", fts_key(MEMOS)),
        format!("Fts/fts:{}:doclens", fts_key(NOTES)),
    ] {
        assert!(
            puts.contains(&expected),
            "flush_full must put {expected}; saw {puts:?}"
        );
    }
    assert_eq!(
        (
            fts_writes(&db, NOTES),
            fts_writes(&db, MEMOS),
            surrogate_writes(&db)
        ),
        (before.0 + 1, before.1 + 1, before.2 + 1)
    );
}

/// Deleting the last document of an index empties it. The flush must store
/// that empty state, or the reopen restores the postings stored before the
/// delete and the document matches again.
#[tokio::test]
async fn deleting_the_last_document_does_not_resurrect_it_on_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fts_last_doc.pagedb");
    {
        let db = open_manual_flush(&path).await;
        put_text(&db, NOTES, "n1", "phoenix rises").await;
        db.flush().await.expect("flush");
        db.document_delete(NOTES, "n1")
            .await
            .expect("document_delete");
        assert_eq!(text_hits(&db, NOTES, "phoenix").await, Vec::<String>::new());
        db.flush().await.expect("flush after the delete");
        db.shutdown().await;
    }

    let db = open_manual_flush(&path).await;
    assert_eq!(
        text_hits(&db, NOTES, "phoenix").await,
        Vec::<String>::new(),
        "a deleted document must not match after a reopen"
    );
}

/// `TRUNCATE` drops the collection's index. The next flush deletes its
/// stored segment and doc lengths, and a later idle flush deletes nothing.
#[tokio::test]
async fn truncated_collection_deletes_its_stored_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fts_truncate.pagedb");
    {
        let (db, probe) = open_probed(&path).await;
        put_text(&db, NOTES, "n1", "phoenix rises").await;
        db.flush().await.expect("seed flush");
        probe.take_deletes();

        db.execute_sql(&format!("TRUNCATE {NOTES}"), &[])
            .await
            .expect("TRUNCATE");
        assert!(
            fts_dirty(&db, NOTES),
            "a dropped index must be flushed so its stored form is deleted"
        );
        db.flush().await.expect("flush after truncate");

        assert_eq!(probe.take_fts_segment_deletes(), vec![fts_key(NOTES)]);
        let deletes = fts_ops(&probe.take_deletes());
        assert!(
            deletes.contains(&format!("Fts/fts:{}:doclens", fts_key(NOTES))),
            "the dropped index's doc lengths must be deleted; saw {deletes:?}"
        );
        assert!(!fts_dirty(&db, NOTES));

        db.flush().await.expect("idle flush");
        assert_eq!(probe.take_fts_segment_deletes(), Vec::<String>::new());
        assert_eq!(fts_ops(&probe.take_deletes()), Vec::<String>::new());
        db.shutdown().await;
    }
    {
        let storage = PagedbStorageDefault::open(&path, Encryption::Plaintext)
            .await
            .expect("open storage");
        let doclens = format!("fts:{}:doclens", fts_key(NOTES));
        assert_eq!(
            storage
                .get(Namespace::Fts, doclens.as_bytes())
                .await
                .expect("get"),
            None,
            "no doc lengths stay behind for a dropped index"
        );
        let segments = storage
            .as_fts_segment_ext()
            .expect("pagedb has FTS segments");
        assert!(
            segments
                .open_fts_segment(&fts_key(NOTES))
                .await
                .expect("open_fts_segment")
                .is_none(),
            "no segment stays behind for a dropped index"
        );
    }

    let db = open_manual_flush(&path).await;
    assert_eq!(text_hits(&db, NOTES, "phoenix").await, Vec::<String>::new());
}

/// A collection dropped and created again under the same index key must not
/// bring back the documents it held before the drop.
#[tokio::test]
async fn recreated_index_does_not_resurrect_dropped_documents() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fts_recreate.pagedb");
    {
        let db = open_manual_flush(&path).await;
        put_text(&db, NOTES, "n1", "phoenix rises").await;
        db.flush().await.expect("seed flush");

        db.execute_sql(&format!("TRUNCATE {NOTES}"), &[])
            .await
            .expect("TRUNCATE");
        put_text(&db, NOTES, "n2", "ember glows").await;
        db.document_delete(NOTES, "n2")
            .await
            .expect("document_delete");
        db.flush().await.expect("flush after the re-create");
        db.shutdown().await;
    }

    let db = open_manual_flush(&path).await;
    assert_eq!(
        text_hits(&db, NOTES, "phoenix").await,
        Vec::<String>::new(),
        "a document dropped with its collection must not match after a reopen"
    );
    assert_eq!(text_hits(&db, NOTES, "ember").await, Vec::<String>::new());
}

// ---------------------------------------------------------------------------
// Every tracked engine together
// ---------------------------------------------------------------------------

/// Write through every engine a flush persists: vectors, an edge, document
/// text, a sparse vector, a geometry, and a KV entry.
async fn seed_every_engine<S: StorageEngine>(db: &NodeDbLite<S>) {
    insert(db, ALPHA, 0..4).await;
    add_edge(db, GRAPH_A, "a", "b").await;
    put_text(db, NOTES, "n1", "phoenix rises").await;
    sparse_insert(db, "a", "d1", 1);
    db.spatial_insert(PLACES, LOC, "p1", &point(1));
    db.kv_put("kv", "k1", b"v1").await.expect("kv_put");
}

/// Write counters of every dirty-tracked artifact the seed creates.
fn tracked_writes<S: StorageEngine>(db: &NodeDbLite<S>) -> Vec<u64> {
    let (graph, segment) = writes(db, ALPHA);
    vec![
        graph,
        segment,
        id_map_writes(db),
        csr_writes(db, GRAPH_A),
        sparse_writes(db, SPARSE, "a"),
        rtree_writes(db, PLACES, LOC),
        docmap_writes(db, PLACES),
        fts_writes(db, NOTES),
        surrogate_writes(db),
    ]
}

/// Seed every engine, flush, then flush again with no mutation in between.
/// The second flush must write nothing: no put, no delete, no batch, no
/// vector, graph, FTS, or spatial segment, and no CRDT snapshot or delta.
async fn assert_second_flush_writes_nothing(config: LiteConfig, file: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (db, probe) = open_probed_with(&dir.path().join(file), config).await;
    seed_every_engine(&db).await;

    db.flush().await.expect("first flush");
    let before = (
        tracked_writes(&db),
        db.crdt_snapshot_export_count(),
        db.crdt_delta_write_count(),
    );
    probe.take_puts();
    probe.take_deletes();
    probe.take_batches();
    probe.take_vector_segment_writes();
    probe.take_graph_segment_writes();
    probe.take_spatial_segment_writes();
    probe.take_fts_segment_writes();
    probe.take_fts_segment_deletes();

    db.flush().await.expect("idle flush");

    assert_eq!(
        op_names(&probe.take_puts()),
        Vec::<String>::new(),
        "an idle flush must not put anything"
    );
    assert_eq!(
        op_names(&probe.take_deletes()),
        Vec::<String>::new(),
        "an idle flush must not delete anything"
    );
    let batches: Vec<Vec<String>> = probe
        .take_batches()
        .iter()
        .map(|batch| op_names(batch.as_slice()))
        .collect();
    assert_eq!(
        batches,
        Vec::<Vec<String>>::new(),
        "an idle flush must not commit a batch"
    );
    assert_eq!(probe.take_vector_segment_writes(), Vec::<String>::new());
    assert_eq!(probe.take_graph_segment_writes(), Vec::<String>::new());
    assert_eq!(probe.take_spatial_segment_writes(), Vec::<String>::new());
    assert_eq!(probe.take_fts_segment_writes(), Vec::<String>::new());
    assert_eq!(probe.take_fts_segment_deletes(), Vec::<String>::new());
    assert_eq!(
        (
            tracked_writes(&db),
            db.crdt_snapshot_export_count(),
            db.crdt_delta_write_count(),
        ),
        before,
        "no tracked artifact, CRDT snapshot, or CRDT delta is written again"
    );
}

/// With every engine flushed, a second flush writes nothing at all.
///
/// Sync is on, the default: the store stages CRDT deltas and outbound FTS
/// and spatial entries for Origin.
#[tokio::test]
async fn second_flush_writes_nothing_at_all() {
    let config = manual_flush_config();
    assert!(config.sync_enabled, "the default config replicates");
    assert_second_flush_writes_nothing(config, "all_idle.pagedb").await;
}

/// The same with sync off, where nothing is staged for Origin.
#[tokio::test]
async fn second_flush_writes_nothing_at_all_with_sync_disabled() {
    let config = LiteConfig {
        sync_enabled: false,
        ..manual_flush_config()
    };
    assert_second_flush_writes_nothing(config, "all_idle_local.pagedb").await;
}
