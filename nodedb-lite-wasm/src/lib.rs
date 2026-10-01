//! JavaScript/TypeScript bindings for NodeDB-Lite via wasm-bindgen.
//!
//! # In-memory storage
//!
//! ```js
//! const db = await NodeDbLiteWasm.openInMemory();
//! // Legacy alias:
//! const db = await NodeDbLiteWasm.open();
//! ```
//!
//! In-memory databases lose their data when the process ends.
//!
//! # Durability
//!
//! Constructors start the background flush task from the resolved config.
//! `auto_flush_ms` controls its interval, not a maximum write-to-storage delay.
//! Call `flush()` to persist buffered writes before continuing.
//! In-memory storage remains volatile after flushing.
//!
//! # Persistent storage
//!
//! Persistent constructors require the `opfs` feature and a browser with OPFS support.
//! PageDB performs synchronous filesystem operations inside a dedicated JavaScript worker.
//! Serve PageDB's worker source at a URL, then pass that URL to the constructor.
//! The source is available as `pagedb::vfs::opfs::OPFS_WORKER_JS` on `wasm32` with `opfs` enabled.
//! The worker runs JavaScript and loads no Rust/WASM module.
//!
//! ```js
//! const db = await NodeDbLiteWasm.openPersistent(
//!     "mydb.pagedb",      // OPFS directory name
//!     "./opfs_worker.js", // Served PageDB worker source
//!     "my-passphrase",   // An empty string explicitly selects plaintext
//! );
//! ```
//!
//! `filename` selects an isolated OPFS directory within the browser origin.
//! Reopening the same directory retains its data.
//! The name must exclude `/`, `\`, and NUL, and cannot be empty, `.` or `..`.
//! Persistent data survives page reloads and browser restarts.
//!
//! # Corruption recovery
//!
//! Normal persistent constructors return an error and preserve unreadable state.
//! `openPersistentDiscardingCorruptState` discards damaged CRDT state and loses its unsynced writes.
//! Use that constructor only when another source can restore the data.
//! OPFS provides no rename primitive, so this constructor cannot replace an unreadable store wholesale.

pub mod array;
pub mod document;
pub mod graph;
pub mod maintenance;
pub mod open;
pub mod query;
pub mod search;
pub mod types;
pub mod udf;
pub mod vector;

pub use types::NodeDbLiteWasm;
pub use udf::register_wasm_udf;

pub(crate) use types::dispatch;
