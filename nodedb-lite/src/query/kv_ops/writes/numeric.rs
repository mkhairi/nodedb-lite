// SPDX-License-Identifier: Apache-2.0
//! Numeric and compare-and-swap writes for the KV engine: incr, incr_float,
//! cas, get_set.
//!
//! Every stored value comes from `nodedb_physical::kv_atomic::compute`, the
//! functions Origin's KV engine calls. Lite and Origin store the same bytes
//! and fail with the same faults for the same op.

use nodedb_physical::kv_atomic::{AtomicComputeError, compute};
use nodedb_physical::physical_plan::KvCounterShape;
use nodedb_types::Namespace;
use nodedb_types::result::QueryResult;
use nodedb_types::value::Value;

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;

use super::super::reads::{decode_value, encode_value, is_expired, kv_key};

/// The live body stored under `rkey` and its expiry deadline. An absent or
/// expired entry reads as `None` with no deadline.
async fn read_live<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    rkey: &[u8],
) -> Result<(Option<Vec<u8>>, u64), LiteError> {
    let stored = engine
        .storage
        .get(Namespace::Kv, rkey)
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })?;
    let Some(raw) = stored else {
        return Ok((None, 0));
    };
    let (deadline, body) = decode_value(&raw).ok_or(LiteError::Storage {
        detail: "corrupt KV entry".into(),
    })?;
    if is_expired(deadline) {
        Ok((None, 0))
    } else {
        Ok((Some(body.to_vec()), deadline))
    }
}

/// Store `body` under `rkey` with the expiry `deadline`.
async fn write_body<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    rkey: &[u8],
    deadline: u64,
    body: &[u8],
) -> Result<(), LiteError> {
    engine
        .storage
        .put(Namespace::Kv, rkey, &encode_value(deadline, body))
        .await
        .map_err(|e| LiteError::Storage {
            detail: e.to_string(),
        })
}

/// The `LiteError` for an atomic that computed no value. The variants carry
/// the same collection and fault Origin's Data Plane error carries.
pub(crate) fn atomic_error(collection: &str, error: AtomicComputeError) -> LiteError {
    match error {
        AtomicComputeError::TypeMismatch { detail } => LiteError::TypeMismatch {
            collection: collection.to_owned(),
            detail,
        },
        AtomicComputeError::Counter(fault) => LiteError::CounterFault {
            collection: collection.to_owned(),
            fault,
        },
        AtomicComputeError::Encode { detail } => LiteError::Serialization { detail },
    }
}

/// Incr: atomic integer counter increment. Returns the new value.
///
/// - A raw body is decimal text in and decimal text out, by the Redis rules.
/// - A typed row moves its first integer column in key order.
/// - An absent key starts at 0 and stores the row `shape` names.
/// - `ttl_ms > 0` installs a fresh expiry. `ttl_ms == 0` keeps the old one.
pub async fn kv_incr<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    delta: i64,
    ttl_ms: u64,
    shape: &KvCounterShape,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let (current, old_deadline) = read_live(engine, &rkey).await?;
    let (value, written) =
        compute::incr(current.as_deref(), delta, shape).map_err(|e| atomic_error(collection, e))?;
    let deadline = if ttl_ms > 0 {
        crate::runtime::now_millis().saturating_add(ttl_ms)
    } else {
        old_deadline
    };
    write_body(engine, &rkey, deadline, &written).await?;

    Ok(QueryResult {
        columns: vec!["value".into()],
        rows: vec![vec![Value::Integer(value)]],
        rows_affected: 1,
        command: None,
    })
}

/// IncrFloat: atomic float increment. Returns the new value.
///
/// - `delta` is the client's decimal text.
/// - A raw body is decimal text in and decimal text out, added exactly.
/// - A typed row moves its first numeric column in key order, in `f64`.
/// - An absent key starts at 0 and stores the row `shape` names.
/// - The key keeps its expiry.
pub async fn kv_incr_float<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    delta: &str,
    shape: &KvCounterShape,
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let (current, old_deadline) = read_live(engine, &rkey).await?;
    let (value, written) = compute::incr_float(current.as_deref(), delta, shape)
        .map_err(|e| atomic_error(collection, e))?;
    write_body(engine, &rkey, old_deadline, &written).await?;

    Ok(QueryResult {
        columns: vec!["value".into()],
        rows: vec![vec![Value::Float(value)]],
        rows_affected: 1,
        command: None,
    })
}

/// Cas: compare-and-swap.
///
/// The current value matches when its bytes equal `expected`, or when it is a
/// typed row whose string column holds `expected`. A typed row swaps only
/// that column. An absent key matches an empty `expected` and is created.
pub async fn kv_cas<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    expected: &[u8],
    new_value: &[u8],
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let (current, old_deadline) = read_live(engine, &rkey).await?;
    let (success, written) = compute::cas(current.as_deref(), expected, new_value)
        .map_err(|e| atomic_error(collection, e))?;
    if success {
        write_body(engine, &rkey, old_deadline, &written).await?;
    }

    Ok(QueryResult {
        columns: vec!["success".into(), "current_value".into()],
        rows: vec![vec![
            Value::Bool(success),
            Value::Bytes(current.unwrap_or_default()),
        ]],
        rows_affected: if success { 1 } else { 0 },
        command: None,
    })
}

/// GetSet: atomically set a new value and return the old one.
///
/// A typed row swaps only its string column. The key keeps its expiry.
pub async fn kv_get_set<S: StorageEngine>(
    engine: &LiteQueryEngine<S>,
    collection: &str,
    key: &[u8],
    new_value: &[u8],
) -> Result<QueryResult, LiteError> {
    let rkey = kv_key(collection, key);
    let (old, old_deadline) = read_live(engine, &rkey).await?;
    let written =
        compute::getset(old.as_deref(), new_value).map_err(|e| atomic_error(collection, e))?;
    write_body(engine, &rkey, old_deadline, &written).await?;

    Ok(QueryResult {
        columns: vec!["old_value".into()],
        rows: vec![vec![old.map_or(Value::Null, Value::Bytes)]],
        rows_affected: 1,
        command: None,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nodedb_physical::kv_atomic::CounterFault;
    use nodedb_types::error::NodeDbError;

    use super::*;
    use crate::query::engine::test_engine;
    use crate::query::kv_ops::reads::kv_get;
    use crate::query::kv_ops::writes::kv_put;

    const COLL: &str = "counters";

    fn row(fields: &[(&str, Value)]) -> Vec<u8> {
        let map: HashMap<String, Value> = fields
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect();
        nodedb_types::value_to_msgpack(&Value::Object(map)).expect("encode row")
    }

    fn typed_shape(column: Option<&str>, rest: &[(&str, Value)]) -> KvCounterShape {
        KvCounterShape::Typed {
            column: column.map(str::to_string),
            template: row(rest),
        }
    }

    fn columns(bytes: &[u8]) -> HashMap<String, Value> {
        match nodedb_types::value_from_msgpack(bytes).expect("decode row") {
            Value::Object(map) => map,
            other => panic!("stored body is not a typed row: {other:?}"),
        }
    }

    async fn stored<S: StorageEngine>(engine: &LiteQueryEngine<S>, key: &[u8]) -> Vec<u8> {
        let r = kv_get(engine, COLL, key, None).await.expect("get");
        match &r.rows[0][1] {
            Value::Bytes(bytes) => bytes.clone(),
            other => panic!("kv_get value column is not bytes: {other:?}"),
        }
    }

    fn fault_of(err: LiteError) -> CounterFault {
        match err {
            LiteError::CounterFault { collection, fault } => {
                assert_eq!(collection, COLL);
                fault
            }
            other => panic!("expected a counter fault, got {other}"),
        }
    }

    fn int_of(r: &QueryResult) -> i64 {
        match r.rows[0][0] {
            Value::Integer(v) => v,
            ref other => panic!("incr value is not an integer: {other:?}"),
        }
    }

    fn float_of(r: &QueryResult) -> f64 {
        match r.rows[0][0] {
            Value::Float(v) => v,
            ref other => panic!("incr_float value is not a float: {other:?}"),
        }
    }

    #[tokio::test]
    async fn incr_on_a_raw_body_stores_decimal_text() {
        let engine = test_engine().await;
        let raw = KvCounterShape::Raw;
        let r = kv_incr(&engine, COLL, b"k", 4, 0, &raw)
            .await
            .expect("incr");
        assert_eq!(int_of(&r), 4);
        assert_eq!(stored(&engine, b"k").await, b"4".to_vec());

        let r = kv_incr(&engine, COLL, b"k", -10, 0, &raw)
            .await
            .expect("incr");
        assert_eq!(int_of(&r), -6);
        assert_eq!(stored(&engine, b"k").await, b"-6".to_vec());
    }

    #[tokio::test]
    async fn incr_on_an_absent_key_under_a_typed_shape_creates_the_typed_row() {
        let engine = test_engine().await;
        let shape = typed_shape(Some("n"), &[("status", Value::String("new".into()))]);
        let r = kv_incr(&engine, COLL, b"k", 7, 0, &shape)
            .await
            .expect("incr");
        assert_eq!(int_of(&r), 7);
        let cols = columns(&stored(&engine, b"k").await);
        assert_eq!(cols.get("n"), Some(&Value::Integer(7)));
        assert_eq!(cols.get("status"), Some(&Value::String("new".into())));
    }

    #[tokio::test]
    async fn incr_on_a_typed_row_moves_its_integer_column() {
        let engine = test_engine().await;
        let current = row(&[
            ("n", Value::Integer(5)),
            ("label", Value::String("x".into())),
        ]);
        kv_put(&engine, COLL, b"k", &current, 0)
            .await
            .expect("seed");
        let r = kv_incr(&engine, COLL, b"k", 3, 0, &KvCounterShape::Raw)
            .await
            .expect("incr");
        assert_eq!(int_of(&r), 8);
        let cols = columns(&stored(&engine, b"k").await);
        assert_eq!(cols.get("n"), Some(&Value::Integer(8)));
        assert_eq!(cols.get("label"), Some(&Value::String("x".into())));
    }

    #[tokio::test]
    async fn a_typed_shape_without_a_column_is_a_type_mismatch() {
        let engine = test_engine().await;
        let shape = typed_shape(None, &[]);
        let err = kv_incr(&engine, COLL, b"k", 1, 0, &shape)
            .await
            .expect_err("no integer column");
        assert!(
            matches!(&err, LiteError::TypeMismatch { collection, .. } if collection == COLL),
            "{err}"
        );
        let err = kv_incr_float(&engine, COLL, b"k", "1", &shape)
            .await
            .expect_err("no numeric column");
        assert!(matches!(err, LiteError::TypeMismatch { .. }), "{err}");
    }

    #[tokio::test]
    async fn incr_past_the_i64_range_is_an_overflow_and_writes_nothing() {
        let engine = test_engine().await;
        let max = i64::MAX.to_string();
        kv_put(&engine, COLL, b"k", max.as_bytes(), 0)
            .await
            .expect("seed");
        let err = kv_incr(&engine, COLL, b"k", 1, 0, &KvCounterShape::Raw)
            .await
            .expect_err("overflow");
        assert_eq!(fault_of(err), CounterFault::IntegerOverflow);
        assert_eq!(stored(&engine, b"k").await, max.into_bytes());
    }

    #[tokio::test]
    async fn a_non_numeric_stored_value_is_a_parse_fault() {
        let engine = test_engine().await;
        kv_put(&engine, COLL, b"k", b"abc", 0).await.expect("seed");
        let err = kv_incr(&engine, COLL, b"k", 1, 0, &KvCounterShape::Raw)
            .await
            .expect_err("not an integer");
        assert_eq!(fault_of(err), CounterFault::NotAnInteger);
        let err = kv_incr_float(&engine, COLL, b"k", "1", &KvCounterShape::Raw)
            .await
            .expect_err("not a float");
        assert_eq!(fault_of(err), CounterFault::NotAFloat);
        assert_eq!(stored(&engine, b"k").await, b"abc".to_vec());
    }

    #[tokio::test]
    async fn incr_float_adds_decimal_text_exactly() {
        let engine = test_engine().await;
        kv_put(&engine, COLL, b"k", b"0.1", 0).await.expect("seed");
        let r = kv_incr_float(&engine, COLL, b"k", "0.2", &KvCounterShape::Raw)
            .await
            .expect("incr_float");
        assert_eq!(float_of(&r), 0.3);
        assert_eq!(stored(&engine, b"k").await, b"0.3".to_vec());

        let r = kv_incr_float(&engine, COLL, b"fresh", "2.5", &KvCounterShape::Raw)
            .await
            .expect("incr_float");
        assert_eq!(float_of(&r), 2.5);
        assert_eq!(stored(&engine, b"fresh").await, b"2.5".to_vec());
    }

    #[tokio::test]
    async fn incr_float_under_a_typed_shape_creates_a_float_column() {
        let engine = test_engine().await;
        let shape = typed_shape(Some("score"), &[]);
        let r = kv_incr_float(&engine, COLL, b"k", "2.5", &shape)
            .await
            .expect("incr_float");
        assert_eq!(float_of(&r), 2.5);
        let cols = columns(&stored(&engine, b"k").await);
        assert_eq!(cols.get("score"), Some(&Value::Float(2.5)));
    }

    #[tokio::test]
    async fn incr_float_to_infinity_is_non_finite() {
        let engine = test_engine().await;
        let max = f64::MAX.to_string();
        kv_put(&engine, COLL, b"k", max.as_bytes(), 0)
            .await
            .expect("seed");
        let err = kv_incr_float(&engine, COLL, b"k", &max, &KvCounterShape::Raw)
            .await
            .expect_err("non-finite");
        assert_eq!(fault_of(err), CounterFault::NonFinite);
    }

    #[test]
    fn counter_faults_become_the_public_errors_origin_builds() {
        for fault in [
            CounterFault::NotAnInteger,
            CounterFault::NotAFloat,
            CounterFault::IntegerOverflow,
            CounterFault::NonFinite,
        ] {
            let lite: NodeDbError = LiteError::CounterFault {
                collection: COLL.into(),
                fault,
            }
            .into();
            let origin =
                NodeDbError::kv_counter_fault(COLL, fault.message(), fault.is_out_of_range());
            assert_eq!(lite.code(), origin.code(), "{fault:?}");
            assert_eq!(lite.message(), origin.message(), "{fault:?}");
            assert_eq!(lite.details(), origin.details(), "{fault:?}");
        }
    }

    #[tokio::test]
    async fn cas_on_a_typed_row_swaps_its_string_column() {
        let engine = test_engine().await;
        let current = row(&[("state", Value::String("idle".into()))]);
        kv_put(&engine, COLL, b"k", &current, 0)
            .await
            .expect("seed");
        let r = kv_cas(&engine, COLL, b"k", b"idle", b"busy")
            .await
            .expect("cas");
        assert_eq!(r.rows[0][0], Value::Bool(true));
        let cols = columns(&stored(&engine, b"k").await);
        assert_eq!(cols.get("state"), Some(&Value::String("busy".into())));
    }
}
