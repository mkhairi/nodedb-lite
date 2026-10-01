// SPDX-License-Identifier: Apache-2.0

//! JNI graph mutation and traversal entry points.

use jni::JNIEnv;
use jni::objects::{JObject, JString};
use jni::sys::{jint, jlong, jstring};

use super::core::get_handle;
use crate::error::record_error;
use crate::{NODEDB_ERR_FAILED, NODEDB_OK, ffi_guard};

/// Insert a directed edge, return its id as a Java string, null on error.
///
/// `nativeGraphDeleteEdge` needs that id. Never discard it.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nodedb_lite_NodeDbLite_nativeGraphInsertEdge(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    collection: JString,
    from: JString,
    to: JString,
    edge_type: JString,
) -> jstring {
    ffi_guard(std::ptr::null_mut(), || {
        let Some(h) = get_handle(handle) else {
            return std::ptr::null_mut();
        };
        let collection: String = match env.get_string(&collection) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return std::ptr::null_mut();
            }
        };
        let from: String = match env.get_string(&from) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return std::ptr::null_mut();
            }
        };
        let to: String = match env.get_string(&to) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return std::ptr::null_mut();
            }
        };
        let edge_type: String = match env.get_string(&edge_type) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return std::ptr::null_mut();
            }
        };

        use nodedb_client::NodeDb;
        let from_id = match nodedb_types::id::NodeId::try_new(from) {
            Ok(id) => id,
            Err(e) => {
                record_error(e);
                return std::ptr::null_mut();
            }
        };
        let to_id = match nodedb_types::id::NodeId::try_new(to) {
            Ok(id) => id,
            Err(e) => {
                record_error(e);
                return std::ptr::null_mut();
            }
        };
        let edge_id = match h.rt.block_on(h.db.graph_insert_edge(
            &collection,
            &from_id,
            &to_id,
            &edge_type,
            None,
        )) {
            Ok(id) => id,
            Err(e) => {
                record_error(e);
                return std::ptr::null_mut();
            }
        };
        match env.new_string(edge_id.to_string()) {
            Ok(s) => s.into_raw(),
            Err(_) => {
                let _ = env.exception_clear();
                std::ptr::null_mut()
            }
        }
    })
}

/// Delete a graph edge by the id from `nativeGraphInsertEdge`.
///
/// Deletion is idempotent: an id naming no live edge returns `NODEDB_OK`.
/// A malformed id returns `NODEDB_ERR_FAILED`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nodedb_lite_NodeDbLite_nativeGraphDeleteEdge(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    collection: JString,
    edge_id: JString,
) -> jint {
    ffi_guard(NODEDB_ERR_FAILED, || {
        let Some(h) = get_handle(handle) else {
            return NODEDB_ERR_FAILED;
        };
        let collection: String = match env.get_string(&collection) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return NODEDB_ERR_FAILED;
            }
        };
        let edge_id: String = match env.get_string(&edge_id) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return NODEDB_ERR_FAILED;
            }
        };

        use nodedb_client::NodeDb;
        let eid: nodedb_types::id::EdgeId = match edge_id.parse() {
            Ok(id) => id,
            Err(e) => {
                record_error(e);
                return NODEDB_ERR_FAILED;
            }
        };
        match h.rt.block_on(h.db.graph_delete_edge(&collection, &eid)) {
            Ok(()) => NODEDB_OK,
            Err(e) => {
                record_error(e);
                NODEDB_ERR_FAILED
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nodedb_lite_NodeDbLite_nativeGraphTraverse(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    collection: JString,
    start: JString,
    depth: jint,
) -> jstring {
    ffi_guard(std::ptr::null_mut(), || {
        let h = match get_handle(handle) {
            Some(h) => h,
            None => return std::ptr::null_mut(),
        };
        let collection: String = match env.get_string(&collection) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return std::ptr::null_mut();
            }
        };
        let start: String = match env.get_string(&start) {
            Ok(s) => s.into(),
            Err(_) => {
                let _ = env.exception_clear();
                return std::ptr::null_mut();
            }
        };

        use nodedb_client::NodeDb;
        let start_id = match nodedb_types::id::NodeId::try_new(start) {
            Ok(id) => id,
            Err(e) => {
                record_error(e);
                return std::ptr::null_mut();
            }
        };
        let subgraph = match h.rt.block_on(h.db.graph_traverse(
            &collection,
            &start_id,
            depth as u8,
            nodedb_types::graph::Direction::Out,
            None,
        )) {
            Ok(sg) => sg,
            Err(e) => {
                record_error(e);
                return std::ptr::null_mut();
            }
        };

        let json = serde_json::json!({
            "nodes": subgraph.nodes.iter().map(|n| serde_json::json!({"id": n.id.as_str(), "depth": n.depth})).collect::<Vec<_>>(),
            "edges": subgraph.edges.iter().map(|e| serde_json::json!({"from": e.from.as_str(), "to": e.to.as_str(), "label": e.label})).collect::<Vec<_>>(),
        });
        let json_str = serde_json::to_string(&json).unwrap_or_else(|_| "{}".into());
        match env.new_string(&json_str) {
            Ok(s) => s.into_raw(),
            Err(_) => {
                let _ = env.exception_clear();
                std::ptr::null_mut()
            }
        }
    })
}
