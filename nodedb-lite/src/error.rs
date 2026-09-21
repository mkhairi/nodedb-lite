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

    /// An integrity constraint refused the write.
    ///
    /// Kept distinct from [`LiteError::BadRequest`], which is also what an
    /// unknown collection or a malformed payload returns, so a caller can
    /// tell a duplicate key from a client mistake. The conversion below hands
    /// it to `NodeDbError::constraint_violation` rather than flattening it
    /// into a storage error, which sets `ErrorCode::CONSTRAINT_VIOLATION` —
    /// and that numeric code is what Origin's pgwire layer maps to SQLSTATE
    /// 23505. `constraint` (`"unique"`, `"not_null"`, ...) rides along in the
    /// details for a caller to read; nothing upstream branches on it.
    #[error("constraint violation on {collection}: {detail}")]
    ConstraintViolation {
        collection: String,
        constraint: String,
        detail: String,
    },
}

/// Returns `true` when `e` is the corruption-class variant that should drive
/// the store-recovery retry (rename the corrupt store, recreate fresh, reopen).
pub(crate) fn is_corruption(e: &LiteError) -> bool {
    matches!(e, LiteError::Corrupted { .. })
}

/// Expression evaluation failure — currently only division/modulo by a zero
/// divisor, which SQL requires to fail the statement (SQLSTATE `22012`) rather
/// than fold the row to `NULL`. Mapped to [`LiteError::Query`] so it surfaces
/// to the caller instead of silently filtering rows out.
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
        if is_corruption(&e) {
            return nodedb_types::error::NodeDbError::segment_corrupted(e.to_string());
        }
        match e {
            LiteError::ConstraintViolation {
                ref collection,
                ref constraint,
                ref detail,
            } => nodedb_types::error::NodeDbError::constraint_violation(
                collection.clone(),
                constraint.clone(),
                detail.clone(),
            ),
            _ => nodedb_types::error::NodeDbError::storage(e),
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
    fn constraint_violation_keeps_its_kind_through_the_conversion() {
        let e = LiteError::ConstraintViolation {
            collection: "users".into(),
            constraint: "unique".into(),
            detail: "duplicate key value violates the primary key on 'users'".into(),
        };
        let ndb: nodedb_types::error::NodeDbError = e.into();
        assert!(
            ndb.is_constraint_violation(),
            "must not flatten into a storage error: {ndb}"
        );
        assert!(ndb.to_string().contains("users"));
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
