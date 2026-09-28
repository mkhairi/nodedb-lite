// SPDX-License-Identifier: Apache-2.0

//! Shared edge-preparation helpers: `EdgeId` allocation, CRDT field lists,
//! and bitemporal history values.
//!
//! Used by both the single-edge insert path
//! (`nodedb::trait_impl::graph::graph_insert_edge_impl`) and the batch path
//! (`nodedb::batch::batch_graph_insert_edges`) so the two write the same
//! `EdgeId`, the same CRDT fields, and the same history payload for the
//! same input.

use std::collections::HashMap;

use loro::LoroValue;

use nodedb_types::document::Document;
use nodedb_types::error::{NodeDbError, NodeDbResult};
use nodedb_types::id::{EdgeId, NodeId};
use nodedb_types::value::Value;

use crate::nodedb::convert::value_to_loro;

/// Returns the CRDT collection name for edges belonging to a graph collection.
pub(crate) fn edge_crdt_collection(collection: &str) -> String {
    format!("__edges__{collection}")
}

/// Allocate the `EdgeId` for a `(from, to, edge_type)` insert.
///
/// `seq` is always `0`: parallel-edge disambiguation by sequence number is
/// not yet wired into either the single-edge or batch insert path, so both
/// use `EdgeId::try_first`.
pub(crate) fn edge_id_for(from: &NodeId, to: &NodeId, edge_type: &str) -> NodeDbResult<EdgeId> {
    EdgeId::try_first(from.clone(), to.clone(), edge_type).map_err(|e| {
        NodeDbError::storage(format!("edge_store: invalid edge label '{edge_type}': {e}"))
    })
}

/// Build the CRDT field list for an edge row: `src`, `dst`, `label`, plus
/// any user-supplied properties.
pub(crate) fn edge_crdt_fields<'a>(
    from: &'a NodeId,
    to: &'a NodeId,
    edge_type: &'a str,
    properties: &'a Option<Document>,
) -> Vec<(&'a str, LoroValue)> {
    let mut fields: Vec<(&str, LoroValue)> = vec![
        ("src", LoroValue::String(from.as_str().into())),
        ("dst", LoroValue::String(to.as_str().into())),
        ("label", LoroValue::String(edge_type.into())),
    ];
    if let Some(props) = properties {
        for (k, v) in &props.fields {
            fields.push((k.as_str(), value_to_loro(v)));
        }
    }
    fields
}

/// Build the bitemporal history payload for an edge insert: `src`, `dst`,
/// `label`, plus any user-supplied properties, as a `Value::Object`.
pub(crate) fn edge_history_value(
    from: &NodeId,
    to: &NodeId,
    edge_type: &str,
    properties: &Option<Document>,
) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("src".to_string(), Value::String(from.as_str().to_string()));
    m.insert("dst".to_string(), Value::String(to.as_str().to_string()));
    m.insert("label".to_string(), Value::String(edge_type.to_string()));
    if let Some(props) = properties {
        for (k, v) in &props.fields {
            m.insert(k.clone(), v.clone());
        }
    }
    Value::Object(m)
}
