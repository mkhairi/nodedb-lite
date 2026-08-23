use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use nodedb_lite_ffi::*;

#[test]
fn open_close_in_memory() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());
        nodedb_close(handle);
    }
}

#[test]
fn null_handle_returns_error() {
    unsafe {
        assert_eq!(nodedb_flush(std::ptr::null_mut()), NODEDB_ERR_NULL);
    }
}

#[test]
fn close_null_is_noop() {
    unsafe {
        nodedb_close(std::ptr::null_mut());
    }
}

#[test]
fn vector_insert_and_search() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());

        let coll = CString::new("vecs").unwrap();
        let id = CString::new("v1").unwrap();
        let emb = [1.0f32, 0.0, 0.0];

        let rc = nodedb_vector_insert(handle, coll.as_ptr(), id.as_ptr(), emb.as_ptr(), 3);
        assert_eq!(rc, NODEDB_OK);

        let query = [1.0f32, 0.0, 0.0];
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = nodedb_vector_search(handle, coll.as_ptr(), query.as_ptr(), 3, 5, &mut out);
        assert_eq!(rc, NODEDB_OK);
        assert!(!out.is_null());

        let json = CStr::from_ptr(out).to_str().unwrap();
        assert!(json.contains("v1"));
        nodedb_free_string(out);

        nodedb_close(handle);
    }
}

#[test]
fn graph_insert_and_traverse() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());

        let collection = CString::new("social").unwrap();
        let from = CString::new("alice").unwrap();
        let to = CString::new("bob").unwrap();
        let label = CString::new("KNOWS").unwrap();

        let rc = nodedb_graph_insert_edge(
            handle,
            collection.as_ptr(),
            from.as_ptr(),
            to.as_ptr(),
            label.as_ptr(),
            std::ptr::null_mut(),
        );
        assert_eq!(rc, NODEDB_OK);

        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = nodedb_graph_traverse(handle, collection.as_ptr(), from.as_ptr(), 2, &mut out);
        assert_eq!(rc, NODEDB_OK);
        assert!(!out.is_null());

        let json = CStr::from_ptr(out).to_str().unwrap();
        assert!(json.contains("alice"));
        assert!(json.contains("bob"));
        nodedb_free_string(out);

        nodedb_close(handle);
    }
}

/// The id returned by `nodedb_graph_insert_edge` must delete that exact edge.
///
/// Deletion is idempotent, so `NODEDB_OK` proves nothing. Traversal decides.
#[test]
fn graph_insert_edge_returns_id_that_deletes_that_edge() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());

        let collection = CString::new("social").unwrap();
        let from = CString::new("alice").unwrap();
        let to = CString::new("bob").unwrap();
        let label = CString::new("KNOWS").unwrap();

        let mut edge_id: *mut c_char = std::ptr::null_mut();
        let rc = nodedb_graph_insert_edge(
            handle,
            collection.as_ptr(),
            from.as_ptr(),
            to.as_ptr(),
            label.as_ptr(),
            &mut edge_id,
        );
        assert_eq!(rc, NODEDB_OK);
        assert!(!edge_id.is_null(), "edge id must be returned");

        // Length-prefixed form: "{src_len}:{src}|{label_len}:{label}|{dst_len}:{dst}|{seq}".
        let edge_id_str = CStr::from_ptr(edge_id).to_str().unwrap();
        assert_eq!(edge_id_str, "5:alice|5:KNOWS|3:bob|0");

        // A well-formed id for another edge must not touch this one.
        let other_id = CString::new("5:alice|5:KNOWS|5:carol|0").unwrap();
        let rc = nodedb_graph_delete_edge(handle, collection.as_ptr(), other_id.as_ptr());
        assert_eq!(rc, NODEDB_OK, "deleting an absent edge is idempotent");
        assert!(
            graph_traverse_json(handle, &collection, &from).contains("bob"),
            "alice -KNOWS-> bob must survive an unrelated delete"
        );

        let rc = nodedb_graph_delete_edge(handle, collection.as_ptr(), edge_id);
        assert_eq!(rc, NODEDB_OK);
        assert!(
            !graph_traverse_json(handle, &collection, &from).contains("bob"),
            "alice -KNOWS-> bob must be gone after deleting its id"
        );

        nodedb_free_string(edge_id);
        nodedb_close(handle);
    }
}

/// A malformed edge id must report an error, never silent success.
#[test]
fn graph_delete_edge_rejects_malformed_id() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        let collection = CString::new("social").unwrap();
        let bogus = CString::new("not-an-edge-id").unwrap();
        assert_eq!(
            nodedb_graph_delete_edge(handle, collection.as_ptr(), bogus.as_ptr()),
            NODEDB_ERR_FAILED
        );
        nodedb_close(handle);
    }
}

/// Traverse depth 2 from `start`, return the JSON body.
unsafe fn graph_traverse_json(
    handle: *mut NodeDbHandle,
    collection: &CString,
    start: &CString,
) -> String {
    unsafe {
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = nodedb_graph_traverse(handle, collection.as_ptr(), start.as_ptr(), 2, &mut out);
        assert_eq!(rc, NODEDB_OK);
        assert!(!out.is_null());
        let json = CStr::from_ptr(out).to_str().unwrap().to_string();
        nodedb_free_string(out);
        json
    }
}

#[test]
fn document_crud_via_ffi() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());

        let coll = CString::new("notes").unwrap();
        let body = CString::new(r#"{"id":"n1","fields":{"title":{"String":"Hello"}}}"#).unwrap();

        let rc = nodedb_document_put(handle, coll.as_ptr(), body.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, NODEDB_OK);

        let id = CString::new("n1").unwrap();
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = nodedb_document_get(handle, coll.as_ptr(), id.as_ptr(), &mut out);
        assert_eq!(rc, NODEDB_OK);
        assert!(!out.is_null());

        let json = CStr::from_ptr(out).to_str().unwrap();
        assert!(json.contains("n1"));
        nodedb_free_string(out);

        let rc = nodedb_document_delete(handle, coll.as_ptr(), id.as_ptr());
        assert_eq!(rc, NODEDB_OK);

        let rc = nodedb_document_get(handle, coll.as_ptr(), id.as_ptr(), &mut out);
        assert_eq!(rc, NODEDB_ERR_NOT_FOUND);

        nodedb_close(handle);
    }
}

#[test]
fn sql_execute_returns_json() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());

        // A constant-expression query is always supported.
        let sql = CString::new("SELECT 1 + 1 AS result").unwrap();
        let mut out: *mut c_char = std::ptr::null_mut();
        let rc = nodedb_lite_ffi::nodedb_execute_sql(handle, sql.as_ptr(), &mut out);
        assert_eq!(rc, NODEDB_OK);
        assert!(!out.is_null());

        let json = CStr::from_ptr(out).to_str().unwrap();
        assert!(json.contains("columns") || json.contains("rows"));
        nodedb_free_string(out);

        nodedb_close(handle);
    }
}

/// REPLICATE #12: open failures must be distinguishable via nodedb_last_error.
/// Today nodedb_open returns NULL for every failure mode (wrong passphrase,
/// corrupt store, bad path) with no error-detail surface at all.
#[test]
fn open_failure_records_last_error() {
    unsafe {
        // A non-:memory: path with NULL passphrase is refused (persistent
        // plaintext storage is not allowed) — NULL handle, and the reason
        // must be retrievable.
        let path = CString::new("/tmp/nodedb-ffi-no-such-dir-x/store").unwrap();
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(handle.is_null());

        let err = nodedb_last_error(handle);
        assert!(
            !err.is_null(),
            "nodedb_last_error must return a reason after open failure"
        );
        let msg = CStr::from_ptr(err).to_str().unwrap();
        assert!(!msg.is_empty());
        nodedb_free_string(err);

        // The error slot must be cleared after a successful call, so a stale
        // failure is never attributed to a later successful operation.
        let mem = CString::new(":memory:").unwrap();
        let ok_handle = nodedb_open(mem.as_ptr(), std::ptr::null());
        assert!(!ok_handle.is_null());
        let err2 = nodedb_last_error(ok_handle);
        assert!(err2.is_null(), "error slot must be cleared after success");
        nodedb_close(ok_handle);
    }
}

#[test]
fn free_null_string_is_noop() {
    unsafe {
        nodedb_free_string(std::ptr::null_mut());
    }
}

/// REPLICATE #13: the cdylib must export `nodedb_version` so bindings can
/// detect library/ABI skew at runtime. Today no version symbol exists —
/// bindings pin the build by comparing sha256sum of the .so, which can only
/// say "different", not "which one".
#[test]
fn version_export_exists() {
    unsafe {
        let version = nodedb_version();
        assert!(!version.is_null(), "nodedb_version must be exported");
        let v = CStr::from_ptr(version).to_str().unwrap();
        assert!(!v.is_empty(), "version string must not be empty");
        // e.g. "0.1.0" or "0.1.0+<sha>"
        assert!(
            v.split(['+', '-']).next().unwrap().split('.').count() >= 2,
            "version must be semver-ish, got: {v}"
        );
    }
}

/// REPLICATE #13 (companion): an integer ABI version lets bindings fail fast
/// on breaking FFI changes instead of dying on a missing symbol.
#[test]
fn abi_version_export_exists() {
    let abi = nodedb_abi_version();
    assert!(abi > 0, "nodedb_abi_version must be a positive integer");
}
