# Changelog

All notable changes to NodeDB Lite are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
NodeDB Lite uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] - 2026-10-16

First release. Embedded NodeDB for Linux, macOS, Windows, Android and the browser, built on NodeDB 0.6.2 crates and pagedb 0.1.

### Fixed

- **Vector index:** search found nothing after the last vector was deleted.
- **Full-text search:** each reopen rebuilt the index on top of the old memory reservation, so the memory budget ran out.

### Added

- `NodeDbLite::flush_full()` writes every dirty-tracked flush artifact
  whether or not it changed. `flush()` stays the dirty-aware pass.
- `NodeDbLite::flush_artifact_write_count(artifact, collection)` and
  `flush_artifact_is_dirty(artifact, collection)` report per-artifact flush
  writes and pending state for `FlushArtifact::{HnswGraph, HnswIdMap,
  VectorSegment}`.

### Fixed

- `flush()` no longer rewrites unchanged HNSW artifacts on every tick. Each
  collection's graph checkpoint, the vector id-map, and each vector segment
  carry a mutation generation and are written only when dirty. The
  `meta:hnsw_collections` and `meta:last_flushed_mid` entries are written only
  when their value changes. An idle store with vector data now makes no HNSW
  or meta writes per tick; before, it rewrote the full vector segment each
  time. CSR, FTS, sparse, and spatial flush paths are unchanged.

- Shutdown no longer aborts an in-flight auto-flush or auto-compaction after
  5 s; it waits for the pass to finish, so a stop during a long flush cannot
  leave a half-written segment in `seg/.staging` (aql#163, NDB-AQL-40).

---

[Unreleased]: https://github.com/NodeDB-Lab/nodedb-lite/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NodeDB-Lab/nodedb-lite/releases/tag/v0.1.0
