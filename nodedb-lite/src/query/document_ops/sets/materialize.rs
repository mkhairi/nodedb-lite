// SPDX-License-Identifier: Apache-2.0
use super::super::is_strict;
use super::super::reads::loro_value_to_ndb_value;
use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::query::msgpack_helpers::{write_array_header, write_bin, write_str, write_u32};
use crate::query::value_utils::value_to_string;
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;
use std::collections::HashMap;
/// MaterializeScan: cursor-paginated full collection scan for the clone materializer.
///
/// Lite is single-node — there is no distributed cursor executor. Instead,
/// every document in the collection is enumerated in insertion order. When
/// `cursor` is non-empty its bytes are interpreted as the UTF-8 ID of the
/// last-seen document; scanning resumes from the ID that follows it
/// lexicographically. Returns at most `count` entries per call. When fewer
/// than `count` entries are returned the next-cursor is empty, signalling
/// scan completion.
///
/// The response payload is msgpack-encoded as a 2-element array:
/// `[next_cursor: bin, entries: [[doc_id: str, surrogate: u32, value_bytes: bin], ...]]`
/// packed into `QueryResult { columns: ["payload"], rows: [[Value::Bytes(payload)]] }`.
pub async fn materialize_scan<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    cursor: &[u8],
    count: usize,
) -> Result<QueryResult, LiteError> {
    let cursor_str = if cursor.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(cursor).into_owned())
    };

    // Collect (doc_id, value_bytes) pairs, resuming from cursor if present.
    let pairs: Vec<(String, Vec<u8>)> = if is_strict(engine, collection) {
        let schema = engine
            .strict
            .schema(collection)
            .ok_or_else(|| LiteError::collection_not_found("strict", collection))?;
        let pk_idx = schema
            .columns
            .iter()
            .position(|c| c.primary_key)
            .ok_or_else(|| LiteError::BadRequest {
                detail: format!("strict collection '{collection}' has no primary key"),
            })?;
        let columns = schema.columns;
        let all_rows = engine.strict.list_rows(collection).await?;
        let mut out = Vec::new();
        let mut past_cursor = cursor_str.is_none();
        for row in all_rows {
            let pk = value_to_string(&row[pk_idx]);
            if !past_cursor {
                if let Some(ref c) = cursor_str
                    && &pk == c
                {
                    past_cursor = true;
                }
                continue;
            }
            if out.len() >= count {
                break;
            }
            let map: HashMap<String, Value> = columns
                .iter()
                .enumerate()
                .filter_map(|(i, col)| {
                    if i < row.len() {
                        Some((col.name.clone(), row[i].clone()))
                    } else {
                        None
                    }
                })
                .collect();
            let bytes = zerompk::to_msgpack_vec(&Value::Object(map)).map_err(|e| {
                LiteError::Serialization {
                    detail: format!("materialize_scan serialize row: {e}"),
                }
            })?;
            out.push((pk, bytes));
        }
        out
    } else {
        let crdt = engine.crdt.lock().map_err(|_| LiteError::LockPoisoned)?;
        let ids = crdt.list_ids(collection);
        let mut out = Vec::new();
        let mut past_cursor = cursor_str.is_none();
        for id in ids {
            if !past_cursor {
                if let Some(ref c) = cursor_str
                    && &id == c
                {
                    past_cursor = true;
                }
                continue;
            }
            if out.len() >= count {
                break;
            }
            if let Some(val) = crdt.read(collection, &id) {
                let ndb_val = loro_value_to_ndb_value(&val);
                let bytes =
                    zerompk::to_msgpack_vec(&ndb_val).map_err(|e| LiteError::Serialization {
                        detail: format!("materialize_scan serialize crdt row: {e}"),
                    })?;
                out.push((id, bytes));
            }
        }
        drop(crdt);
        out
    };

    // Build the same msgpack response shape as Origin:
    // [next_cursor: bin, entries: [[doc_id: str, 0u32, value_bytes: bin], ...]]
    let next_cursor: Vec<u8> = if pairs.len() < count {
        Vec::new()
    } else {
        pairs
            .last()
            .map(|(id, _)| id.as_bytes().to_vec())
            .unwrap_or_default()
    };

    let payload = encode_materialize_payload(&next_cursor, &pairs);

    Ok(QueryResult {
        columns: vec!["payload".into()],
        rows: vec![vec![Value::Bytes(payload)]],
        rows_affected: 0,
        command: None,
    })
}

/// Encode the MaterializeScan response payload in the same msgpack shape as Origin.
fn encode_materialize_payload(next_cursor: &[u8], pairs: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    write_array_header(&mut out, 2);
    write_bin(&mut out, next_cursor);
    write_array_header(&mut out, pairs.len());
    for (doc_id, value_bytes) in pairs {
        write_array_header(&mut out, 3);
        write_str(&mut out, doc_id.as_bytes());
        // Lite has no catalog-assigned surrogates; emit 0 as a sentinel.
        write_u32(&mut out, 0u32);
        write_bin(&mut out, value_bytes);
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::NodeDbLite;
    use crate::PagedbStorageMem;

    async fn make_db() -> std::sync::Arc<NodeDbLite<PagedbStorageMem>> {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        NodeDbLite::open(storage).await.unwrap()
    }

    /// encode_materialize_payload emits a 2-element fixarray (0x92) as outer header.
    #[test]
    fn materialize_scan_payload_envelope_shape() {
        let payload = super::encode_materialize_payload(&[], &[]);
        // 0x92 = msgpack fixarray len=2
        assert_eq!(payload[0], 0x92, "outer envelope must be fixarray(2)");
    }

    /// encode_materialize_payload with one entry encodes doc_id as msgpack str,
    /// surrogate as u32 (0xce prefix), and value_bytes as bin.
    #[test]
    fn materialize_scan_payload_one_entry() {
        let doc_id = "abc".to_string();
        let val_bytes = b"data".to_vec();
        let payload =
            super::encode_materialize_payload(&[], &[(doc_id.clone(), val_bytes.clone())]);
        // The outer array is len=2; first element is empty bin (cursor); second is
        // fixarray(1) wrapping fixarray(3) = [str, u32, bin].
        assert!(payload.len() > 10, "payload must have content");
        // Scan for the doc_id bytes within the payload.
        let needle = doc_id.as_bytes();
        let found = payload.windows(needle.len()).any(|w| w == needle);
        assert!(found, "doc_id must appear in payload");
    }

    /// materialize_scan on an empty schemaless collection returns a payload row.
    #[tokio::test]
    async fn materialize_scan_empty_collection() {
        let db = make_db().await;
        let result = super::materialize_scan(&db.query_engine, "nonexistent_coll", &[], 10)
            .await
            .unwrap();
        assert_eq!(result.columns, vec!["payload"]);
        assert_eq!(result.rows.len(), 1);
        // Payload must be non-empty — it contains the msgpack envelope at minimum.
        if let nodedb_types::value::Value::Bytes(payload) = &result.rows[0][0] {
            assert!(!payload.is_empty());
        } else {
            panic!("expected Value::Bytes for MaterializeScan result");
        }
    }
}
