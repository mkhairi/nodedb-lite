pub mod analyzer;
pub(crate) mod catalog;
pub mod checkpoint;
pub(crate) mod coordinator;
pub(crate) mod maintain;
pub mod manager;
pub(crate) mod rebuild;
pub mod search;
pub mod state;

pub use manager::FtsCollectionManager;
pub(crate) use search::{TextSearchRequest, run_text_search};
pub use state::{FtsState, LiteFtsIndex};

// Re-export types callers need.
pub use nodedb_fts::FtsIndex;
pub use nodedb_fts::backend::FtsBackend;
pub use nodedb_fts::backend::memory::MemoryBackend;
pub use nodedb_fts::posting::{MatchOffset, Posting, QueryMode, TextSearchResult};
