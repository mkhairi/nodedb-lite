// SPDX-License-Identifier: Apache-2.0
//! Shared scalar-to-string conversion for key components.

use nodedb_types::value::Value;

/// Convert a scalar `Value` into the canonical string form used as a key
/// component. Non-scalar variants collapse to the empty string so they can
/// still produce a deterministic key segment.
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Uuid(s) => s.clone(),
        Value::Null => String::new(),
        _ => String::new(),
    }
}
