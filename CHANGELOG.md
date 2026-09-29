# Changelog

All notable changes to NodeDB Lite are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
NodeDB Lite uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

> NodeDB Lite has not been released. No tag, no crates.io publish, no
> distributed binary. The first release will be 0.1.0, covering embedded use
> on Linux, macOS, Windows, Android, and the browser, plus pilot integration
> with NodeDB Origin.
>
> Public API and exported FFI symbol signatures stay unfrozen until that tag.

### Added

- `NodeDbLite::flush_full()` writes every dirty-tracked flush artifact
  whether or not it changed. `flush()` stays the dirty-aware pass.
- `NodeDbLite::flush_artifact_write_count(artifact, collection)` and
  `flush_artifact_is_dirty(artifact, collection)` report per-artifact flush
  writes and pending state for `FlushArtifact::{HnswGraph, HnswIdMap,
  VectorSegment, CsrGraph, SparseIndex, SpatialRtree, SpatialDocMap}`.
  `spatial_rtree_key(collection, field)` builds the key a `SpatialRtree` is
  tracked under.

### Fixed

- A reopened or lazily reloaded HNSW index no longer attaches the wrong
  vector to its nodes. The vector segment was built from the durable rows in
  document-id key order, but the graph numbers nodes in insertion order. When
  the two orders differed, search returned wrong results with no error. The
  segment is now built from the index in node order. Each slot carries a
  stamp of its bound document id, or a tombstone marker for an unbound slot.
  Open and lazy-load check every stamp against the id-map before they attach
  the segment, and rebuild from the durable rows on any mismatch. A segment
  written before this fix has no stamps, so it is rebuilt once and rewritten
  by the next flush. `TRUNCATE` and `DropIndex` now unlink the stored segment,
  so a stale one cannot attach after the same ids are inserted again.

- `flush()` no longer rewrites unchanged HNSW artifacts on every tick. Each
  collection's graph checkpoint, the vector id-map, and each vector segment
  carry a mutation generation and are written only when dirty. The
  `meta:hnsw_collections` and `meta:last_flushed_mid` entries are written only
  when their value changes. An idle store with vector data now makes no HNSW
  or meta writes per tick; before, it rewrote the full vector segment each
  time. The FTS flush path is unchanged.

- `flush()` no longer rewrites every CSR graph on every tick. Each graph
  collection's adjacency checkpoint carries a mutation generation and is
  written only when dirty, as a pagedb graph segment or a `csr:<collection>`
  blob. A failed segment write leaves the collection dirty, and the next flush
  retries it. `meta:csr_collections` is written in sorted order, and only when
  the set of collections changes. A graph restored from its stored checkpoint
  starts clean. A graph rebuilt from edge documents at open starts dirty.

- `flush()` no longer rewrites every sparse index on every tick. Each index
  carries a mutation generation and is written only when dirty. An insert
  that stores the vector a document already holds, and a document write that
  removes nothing from an index, leave it clean. `sparse:_indices` is written
  only when the set of indexes changes. An index restored from a blob that
  decoded starts clean. An absent or undecodable blob starts dirty.

- `flush()` no longer rewrites every spatial R-tree and doc-map on every
  tick. Each `(collection, field)` R-tree and each collection's doc-map carry
  a mutation generation and are written only when dirty.
  `spatial:_collections` is written in sorted order, and only when the set of
  trees changes. `spatial:_next_id` is written only when it changes. The
  doc-map and catalog batch is skipped when empty. A failed R-tree segment
  write is logged and leaves the tree dirty, and the next flush retries it.
  Before, it aborted the flush and skipped the FTS and sparse writes after
  it. A tree restored from its checkpoint starts clean. A tree rebuilt at
  open starts dirty.

- Shutdown no longer aborts an in-flight auto-flush or auto-compaction after
  5 s; it waits for the pass to finish, so a stop during a long flush cannot
  leave a half-written segment in `seg/.staging` (aql#163, NDB-AQL-40).

---

[Unreleased]: https://github.com/NodeDB-Lab/nodedb-lite/commits/main
