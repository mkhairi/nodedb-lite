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

[Unreleased]: https://github.com/NodeDB-Lab/nodedb-lite/commits/main
