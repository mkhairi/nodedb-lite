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

### Fixed

- Shutdown no longer aborts an in-flight auto-flush or auto-compaction after
  5 s; it waits for the pass to finish, so a stop during a long flush cannot
  leave a half-written segment in `seg/.staging` (aql#163, NDB-AQL-40).

---

[Unreleased]: https://github.com/NodeDB-Lab/nodedb-lite/commits/main
