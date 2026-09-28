//! Error types for NodeDB-Lite.

/// Errors specific to the Lite embedded engine.
#[derive(Debug, thiserror::Error)]
pub enum LiteError {
    #[error("storage error: {detail}")]
    Storage { detail: String },

    #[error("storage backend returned poison lock")]
    LockPoisoned,

    #[error("async task join failed: {detail}")]
    JoinError { detail: String },

    #[error("serialization error: {detail}")]
    Serialization { detail: String },

    #[error("namespace {ns} not recognized")]
    InvalidNamespace { ns: u8 },

    #[error("bad request: {detail}")]
    BadRequest { detail: String },

    /// A write that would duplicate a declared unique key. Maps to SQLSTATE
    /// `23505` at the SQL boundary.
    #[error("duplicate key value violates unique constraint on '{collection}': {detail}")]
    UniqueViolation { collection: String, detail: String },

    /// A write that would store NULL in a NOT NULL column. Maps to SQLSTATE
    /// `23502` at the SQL boundary.
    #[error("null value in column '{column}' of '{collection}' violates not-null constraint")]
    NotNullViolation { collection: String, column: String },

    #[error("sync error: {detail}")]
    Sync { detail: String },

    #[error("query error: {0}")]
    Query(String),

    #[error("Arrow type conversion: expected {expected}, got {got}")]
    ArrowTypeConversion { expected: String, got: String },

    #[error("backpressure: {detail}")]
    Backpressure { detail: String },

    /// Feature or SQL construct not supported in this Lite beta release.
    #[error("unsupported: {detail}")]
    Unsupported { detail: String },

    /// The OPFS worker bridge failed to start or encountered an IPC error.
    ///
    /// This variant is produced when `PagedbStorage::open_opfs` cannot spawn
    /// the dedicated Web Worker or when the worker signals a corruption-class
    /// failure that cannot be recovered automatically (OPFS has no rename).
    #[error("OPFS worker bridge failed: {detail}")]
    WorkerFailed { detail: String },

    /// An error during key derivation, salt I/O, or encryption setup.
    #[error("encryption error: {detail}")]
    Encryption { detail: String },

    /// A corruption-class failure surfaced from the storage backend.
    ///
    /// Kept as a distinct variant (rather than folded into [`LiteError::Storage`])
    /// so the corruption signal survives the error-type conversions and reaches
    /// the open-sequence recovery driver, which renames the corrupt store aside,
    /// recreates a fresh one, and retries the open exactly once.
    #[error("storage corrupted: {detail}")]
    Corrupted { detail: String },

    /// A full-text index update failed while writing a document.
    ///
    /// The write that triggered it must fail too: the row and its postings are
    /// written together, and nothing re-indexes the gap afterwards — the next
    /// write to the document only indexes its new text, so a swallowed failure
    /// leaves the document permanently missing from (or stale in) the index.
    #[error("full-text index update failed for {collection}: {detail}")]
    FtsIndex { collection: String, detail: String },

    /// A KV counter atomic read a stored value it cannot parse as a number,
    /// or computed a result out of range. Maps to SQLSTATE `22P02` or `22003`
    /// at the SQL boundary, the same as Origin.
    #[error("{fault} on {collection}")]
    CounterFault {
        collection: String,
        fault: nodedb_physical::kv_atomic::CounterFault,
    },

    /// A KV atomic found a typed row with no column of the type it reads.
    #[error("type mismatch on {collection}: {detail}")]
    TypeMismatch { collection: String, detail: String },

    /// An input the engine cannot compute on: a vector of the wrong
    /// dimension, or index input it cannot use. The same public error
    /// Origin raises for the same fault, SQLSTATE `22000`.
    #[error("{detail}")]
    DataException { detail: String },
}

/// A vector engine error, classified as Origin classifies it: an input
/// vector of the wrong dimension, or index input the engine cannot use, is
/// the caller's data error; a memory budget refusal is backpressure; a
/// stored-data or checkpoint failure is corruption.
impl From<nodedb_vector::error::VectorError> for LiteError {
    fn from(e: nodedb_vector::error::VectorError) -> Self {
        use nodedb_vector::error::VectorError as Ve;
        let detail = e.to_string();
        match e {
            Ve::DimensionMismatch { .. } | Ve::InvalidInput { .. } => {
                Self::DataException { detail }
            }
            Ve::BudgetExhausted(_) => Self::Backpressure { detail },
            Ve::SegmentIo(_) | Ve::InvalidFilterBitmap { .. } => Self::Storage { detail },
            // `VectorError` is `#[non_exhaustive]`; every other variant is a
            // stored-data or checkpoint failure.
            _ => Self::Corrupted { detail },
        }
    }
}

/// Returns `true` when `e` is the corruption-class variant that should drive
/// the store-recovery retry (rename the corrupt store, recreate fresh, reopen).
pub(crate) fn is_corruption(e: &LiteError) -> bool {
    matches!(e, LiteError::Corrupted { .. })
}

/// Expression evaluation failure: division or modulo by a zero divisor, a
/// call to a function no evaluator implements, or a function argument it
/// cannot compute on (vectors of different dimensions, an argument of the
/// wrong type, a malformed JSONPath). SQL requires each to fail the
/// statement rather than fold the row to `NULL`. Mapped to
/// [`LiteError::Query`] so it surfaces to the caller instead of silently
/// filtering rows out.
impl From<nodedb_query::EvalError> for LiteError {
    fn from(e: nodedb_query::EvalError) -> Self {
        Self::Query(e.to_string())
    }
}

impl From<nodedb_types::columnar::SchemaError> for LiteError {
    fn from(e: nodedb_types::columnar::SchemaError) -> Self {
        Self::BadRequest {
            detail: e.to_string(),
        }
    }
}

impl From<LiteError> for nodedb_types::error::NodeDbError {
    fn from(e: LiteError) -> Self {
        use nodedb_types::error::NodeDbError;
        match e {
            // The same public errors Origin builds for the same faults.
            LiteError::CounterFault { collection, fault } => {
                NodeDbError::kv_counter_fault(collection, fault, fault.is_out_of_range())
            }
            LiteError::TypeMismatch { collection, detail } => {
                NodeDbError::type_mismatch(collection, detail)
            }
            LiteError::DataException { detail } => NodeDbError::data_exception(detail),
            e if is_corruption(&e) => NodeDbError::segment_corrupted(e.to_string()),
            e => NodeDbError::storage(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lite_error_display() {
        let e = LiteError::Storage {
            detail: "disk full".into(),
        };
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn lite_error_converts_to_nodedb_error() {
        let e = LiteError::Storage {
            detail: "test".into(),
        };
        let ndb: nodedb_types::error::NodeDbError = e.into();
        assert!(ndb.to_string().contains("test"));
    }

    #[test]
    fn lite_error_encryption_display_and_convert() {
        let e = LiteError::Encryption {
            detail: "argon2 key derivation failed".into(),
        };
        let rendered = e.to_string();
        assert!(rendered.contains("encryption error"));
        assert!(rendered.contains("argon2 key derivation failed"));

        let ndb: nodedb_types::error::NodeDbError = e.into();
        assert!(ndb.to_string().contains("argon2 key derivation failed"));
    }

    #[test]
    fn lite_error_backpressure_display() {
        let e = LiteError::Backpressure {
            detail: "outbound queue full".into(),
        };
        assert!(e.to_string().contains("backpressure"));
        assert!(e.to_string().contains("outbound queue full"));
    }
}
