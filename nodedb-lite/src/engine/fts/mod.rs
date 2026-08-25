pub mod analyzer;
pub mod checkpoint;
pub mod manager;
pub mod search;
pub mod state;

pub use manager::FtsCollectionManager;
pub(crate) use search::run_text_search;
pub use state::FtsState;

// Re-export types callers need.
pub use nodedb_fts::FtsIndex;
pub use nodedb_fts::backend::FtsBackend;
pub use nodedb_fts::backend::memory::MemoryBackend;
pub use nodedb_fts::posting::{MatchOffset, Posting, QueryMode, TextSearchResult};

/// Type alias for Lite's persistent FTS index (serialized to KV store on flush).
pub type LiteFtsIndex = FtsIndex<MemoryBackend>;

/// Memtable thresholds for Lite's FTS indexes: never spill.
///
/// Lite runs `nodedb-fts` on `MemoryBackend`, whose LSM segments live in
/// process memory, and [`checkpoint::serialize_fts`] persists the memtable
/// only. A spill therefore moves postings from a durable place to a
/// non-durable one at no memory saving: everything drained before a checkpoint
/// is absent from it and gone at the next open (NDB-AQL-37).
///
/// Ceiling: the checkpoint is not incremental, so it re-serializes the whole
/// vocabulary on every flush — cost now grows with total terms rather than
/// being capped at the old 100k spill threshold. If that write cost starts to
/// matter, the upgrade path is a dirty-term checkpoint, or teaching
/// `serialize_fts` to persist the backend's segments too; it is not to start
/// dropping postings again.
pub(crate) const LITE_MEMTABLE_CONFIG: nodedb_fts::MemtableConfig = nodedb_fts::MemtableConfig {
    max_postings: usize::MAX,
    max_terms: usize::MAX,
};
