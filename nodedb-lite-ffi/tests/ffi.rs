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

/// A failed open records a reason; a successful one leaves no stale reason.
///
/// `nodedb_open` returns NULL for every failure mode, so the message is the
/// only way an embedder tells "wrong passphrase" from "corrupt store".
#[test]
fn open_failure_records_last_error() {
    unsafe {
        // A persistent path with a NULL passphrase is refused: silent plaintext
        // storage is not allowed. The refusal must name itself.
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
fn open_null_path_records_null_error() {
    unsafe {
        // NULL path is a programming error and must not be reported as an
        // invalid-UTF-8 input error.
        let handle = nodedb_open(std::ptr::null(), std::ptr::null());
        assert!(handle.is_null());
        let err = nodedb_last_error(handle);
        assert!(!err.is_null());
        let msg = CStr::from_ptr(err).to_str().unwrap();
        assert_eq!(msg, "path is NULL");
        nodedb_free_string(err);
    }
}

/// Closing a handle with active sync must not crash.
///
/// A sync task torn down with the runtime crashed the process. Close now
/// stops the database's tasks first.
///
/// Sync points at a local TCP listener and the test waits for the accept, so
/// the task has provably reached the connect path before the close. Closing
/// before the task is ever polled would test nothing.
#[test]
fn close_with_active_sync_does_not_crash() {
    unsafe {
        for _ in 0..10 {
            let path = CString::new(":memory:").unwrap();
            let handle = nodedb_open(path.as_ptr(), std::ptr::null());
            assert!(!handle.is_null());

            // Reachable endpoint: plain TCP listener. The client connects and
            // starts the WebSocket handshake, then stalls waiting for our
            // (never sent) response — the mid-connect state that crashed.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("ws://{}", listener.local_addr().unwrap());
            let url = CString::new(url).unwrap();
            let jwt = CString::new("some-jwt").unwrap();
            let rc = nodedb_start_sync(handle, url.as_ptr(), jwt.as_ptr());
            assert_eq!(rc, NODEDB_OK);

            listener.set_nonblocking(true).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let mut accepted = false;
            while std::time::Instant::now() < deadline {
                if listener.accept().is_ok() {
                    accepted = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert!(accepted, "sync task never reached the connect path");

            nodedb_close(handle);
        }
    }
}

/// `nodedb_stop_sync` quiesces sync without closing the handle.
///
/// A second stop reports no task.
#[test]
fn stop_sync_quiesces_and_handle_stays_usable() {
    unsafe {
        let path = CString::new(":memory:").unwrap();
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());

        let url = CString::new("ws://127.0.0.1:1").unwrap();
        let jwt = CString::new("some-jwt").unwrap();
        let rc = nodedb_start_sync(handle, url.as_ptr(), jwt.as_ptr());
        assert_eq!(rc, NODEDB_OK);

        // Stopping returns OK and does not invalidate the handle.
        let rc = nodedb_stop_sync(handle);
        assert_eq!(rc, NODEDB_OK);

        // Second stop: no sync task left.
        let rc = nodedb_stop_sync(handle);
        assert_eq!(rc, NODEDB_ERR_FAILED);

        // Handle still works after stop.
        let coll = CString::new("notes").unwrap();
        let body = CString::new(r#"{"id":"n1","fields":{"title":{"String":"Hello"}}}"#).unwrap();
        let rc = nodedb_document_put(handle, coll.as_ptr(), body.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, NODEDB_OK);

        nodedb_close(handle);
    }
}

/// A second `nodedb_start_sync` on a running handle must be rejected, not
/// silently detach the running loop (which would make it unstoppable).
#[test]
fn second_start_sync_is_rejected() {
    unsafe {
        let path = CString::new(":memory:").unwrap();
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());

        let url = CString::new("ws://127.0.0.1:1").unwrap();
        let jwt = CString::new("some-jwt").unwrap();
        let rc = nodedb_start_sync(handle, url.as_ptr(), jwt.as_ptr());
        assert_eq!(rc, NODEDB_OK);

        // Second start while the first is still running: rejected, with a
        // reason the embedder can read.
        let rc = nodedb_start_sync(handle, url.as_ptr(), jwt.as_ptr());
        assert_eq!(rc, NODEDB_ERR_FAILED);
        let err = nodedb_last_error(handle);
        assert!(!err.is_null(), "the rejection must record a reason");
        let msg = CStr::from_ptr(err).to_str().unwrap();
        assert!(msg.contains("already running"), "unexpected reason: {msg}");
        nodedb_free_string(err);

        // After a stop, starting again succeeds.
        let rc = nodedb_stop_sync(handle);
        assert_eq!(rc, NODEDB_OK);
        let rc = nodedb_start_sync(handle, url.as_ptr(), jwt.as_ptr());
        assert_eq!(rc, NODEDB_OK);

        nodedb_close(handle);
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

/// An operational failure records the engine's reason, not just `-3`.
#[test]
fn operation_failure_records_last_error() {
    let path = CString::new(":memory:").unwrap();
    unsafe {
        let handle = nodedb_open(path.as_ptr(), std::ptr::null());
        assert!(!handle.is_null());

        let coll = CString::new("notes").unwrap();
        let body = CString::new("{ this is not json }").unwrap();
        let rc = nodedb_document_put(handle, coll.as_ptr(), body.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, NODEDB_ERR_FAILED);

        let err = nodedb_last_error(handle);
        assert!(!err.is_null(), "a failed put must record its reason");
        let msg = CStr::from_ptr(err).to_str().unwrap();
        assert!(!msg.is_empty(), "reason must not be empty");
        nodedb_free_string(err);

        nodedb_close(handle);
    }
}

/// An argument rejected before the engine runs names the argument.
#[test]
fn unknown_id_type_records_last_error() {
    unsafe {
        let id_type = CString::new("uuidv9").unwrap();
        let mut out: *mut c_char = std::ptr::null_mut();
        assert_eq!(
            nodedb_generate_id_typed(id_type.as_ptr(), &mut out),
            NODEDB_ERR_FAILED
        );

        let err = nodedb_last_error(std::ptr::null_mut());
        assert!(!err.is_null());
        let msg = CStr::from_ptr(err).to_str().unwrap();
        assert!(
            msg.contains("uuidv9") && msg.contains("uuidv7"),
            "message must name the rejected type and the supported set, got: {msg}"
        );
        nodedb_free_string(err);
    }
}
