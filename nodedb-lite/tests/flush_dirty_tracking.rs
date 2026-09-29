// SPDX-License-Identifier: Apache-2.0

//! Flush writes a derived HNSW or CSR artifact only when it changed.
//!
//! `flush()` runs on a timer. Rewriting every collection's graph checkpoint,
//! the vector id-map, every vector segment, and every CSR adjacency checkpoint
//! on each tick costs their full size whether or not anything changed: an
//! idle store with a large vector collection rewrote hundreds of megabytes
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
use nodedb_lite::nodedb::{FlushArtifact, ID_MAP_KEY};
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
use nodedb_types::Namespace;
use nodedb_types::id::NodeId;
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
    let inner = PagedbStorageDefault::open(path, Encryption::Plaintext)
        .await
        .expect("open storage");
    let probe = Arc::new(Probe::default());
    let storage = ProbeStorage {
        inner,
        probe: Arc::clone(&probe),
    };
    let db = NodeDbLite::open_with_config(storage, manual_flush_config())
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
}

impl Probe {
    fn take_puts(&self) -> Vec<(Namespace, Vec<u8>)> {
        std::mem::take(&mut *self.puts.lock().unwrap())
    }

    fn take_graph_segment_writes(&self) -> Vec<String> {
        std::mem::take(&mut *self.graph_segment_writes.lock().unwrap())
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
        self.inner.delete(ns, key).await
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
        for op in ops {
            if let WriteOp::Put { ns, key, value: _ } = op {
                self.record(*ns, key);
            }
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
        self.inner.as_fts_segment_ext()
    }

    fn as_columnar_segment_ext(&self) -> Option<&dyn ColumnarSegmentExt> {
        self.inner.as_columnar_segment_ext()
    }

    fn as_graph_segment_ext(&self) -> Option<&dyn GraphSegmentExt> {
        Some(self)
    }

    fn as_spatial_segment_ext(&self) -> Option<&dyn SpatialSegmentExt> {
        self.inner.as_spatial_segment_ext()
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
            .await
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
