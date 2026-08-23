# Contributing to NodeDB Lite

Thank you for your interest in contributing.

## Repository layout

| Crate | Description |
|---|---|
| `nodedb-lite` | Core embedded Rust library |
| `nodedb-lite-ffi` | C FFI bindings (cbindgen, Kotlin/JNI for Android) |
| `nodedb-lite-wasm` | JavaScript/TypeScript bindings via wasm-bindgen |

## Building

```bash
# Check all crates compile
cargo check --workspace

# Rust core
cargo build -p nodedb-lite

# C FFI (requires cbindgen)
cargo build -p nodedb-lite-ffi

# WASM (requires wasm-pack)
wasm-pack build --target web nodedb-lite-wasm
```

## Running tests

```bash
# Rust unit and integration tests
cargo nextest run -p nodedb-lite

# FFI tests
cargo nextest run -p nodedb-lite-ffi

# WASM tests (headless browser required)
wasm-pack test --headless --firefox nodedb-lite-wasm
```

Always use `cargo nextest run`, not `cargo test`. The test suite relies on nextest's
per-test isolation and retry configuration in `.config/nextest.toml`.

## Before opening a pull request

- [ ] `cargo fmt --all` — no formatting diffs
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — no warnings
- [ ] `cargo nextest run -p nodedb-lite` — all tests green
- [ ] New public API has at least one integration test in `tests/`
- [ ] No `.unwrap()` calls in library code — propagate errors with `?`
- [ ] Files stay under 500 lines; split by concern if needed

## Changing the FFI surface

Exported symbol signatures freeze at the **first tagged release**. Before that
tag, change a signature in place and regenerate the header. After it, add a
separately named export instead — an in-place change leaves adapters compiled
against the old declaration calling a function with a different shape.

`nodedb-lite-ffi/abi/surface.txt` records every export. Any change to the
surface must be re-recorded in the same commit, so it lands in the diff where a
reviewer sees it:

```bash
UPDATE_ABI_SNAPSHOT=1 cargo nextest run -p nodedb-lite-ffi -E 'binary(abi_surface)'
```

Adding an export is backward-compatible. Changing or removing one is breaking —
bump `NODEDB_ABI_VERSION` in `nodedb-lite-ffi/src/version.rs` in the same
commit, or the snapshot test fails.

Adding a JNI export means adding its Kotlin `external fun` too; the parity
tests fail on a symbol declared on one side only, or declared with the wrong
number of arguments.

Some contract changes are invisible to all of this. Changing who owns a
returned pointer, or whether an out-parameter is written on failure, keeps
every signature identical and breaks every caller. Those changes need a note in
`docs/ffi-abi.md` and a reviewer who reads it.

## Commit messages

Conventional commits are encouraged: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
Use the imperative mood in the subject line ("Add X", "Fix Y", not "Added X").
Keep the subject under 72 characters. Reference the relevant issue number if one exists.

## Code of conduct

This project follows the [Contributor Covenant 2.1](./CODE_OF_CONDUCT.md).
