// SPDX-License-Identifier: Apache-2.0

use nodedb_lite_ffi::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

type Search = unsafe extern "C" fn(*mut NodeDbHandle, *const c_char, *mut *mut c_char) -> i32;

struct Db(*mut NodeDbHandle);
impl Db {
    fn open() -> Self {
        let path = CString::new(":memory:").unwrap();
        let handle = unsafe { nodedb_open(path.as_ptr(), std::ptr::null()) };
        assert!(!handle.is_null());
        Self(handle)
    }
    fn search(&self, search: Search, params: &str) -> Vec<nodedb_types::result::SearchResult> {
        let params = CString::new(params).unwrap();
        let mut out = std::ptr::null_mut();
        unsafe {
            assert_eq!(search(self.0, params.as_ptr(), &mut out), NODEDB_OK);
            let results = sonic_rs::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap();
            nodedb_free_string(out);
            results
        }
    }
}
impl Drop for Db {
    fn drop(&mut self) {
        unsafe {
            nodedb_close(self.0);
        }
    }
}

#[test]
fn hybrid_json_preserves_field_scope_and_allowed_ids() {
    let db = Db::open();
    let collection = CString::new("docs").unwrap();
    let body = CString::new(r#"{"id":"a","fields":{"title":"rust","body":"context"}}"#).unwrap();
    unsafe {
        assert_eq!(
            nodedb_document_put(
                db.0,
                collection.as_ptr(),
                body.as_ptr(),
                std::ptr::null_mut()
            ),
            NODEDB_OK
        );
    }
    assert_eq!(db.search(nodedb_hybrid_search, r#"{"collection":"docs","vector_k":0,"query_text":"rust","text_field":"title","allowed_ids":null}"#)[0].id, "a");
    assert!(
        db.search(
            nodedb_hybrid_search,
            r#"{"collection":"docs","vector_k":0,"query_text":"rust","text_field":"body"}"#
        )
        .is_empty()
    );
    for allowed in ["[]", "[\"unknown\"]"] {
        assert!(db.search(nodedb_hybrid_search, &format!(r#"{{"collection":"docs","vector_k":0,"query_text":"rust","allowed_ids":{allowed}}}"#)).is_empty());
    }
}

#[test]
fn graph_json_preserves_seeds_depth_and_allowed_ids() {
    let db = Db::open();
    let results = db.search(nodedb_graph_rag_search, r#"{"collection":"docs","query":[],"vector_k":0,"graph_depth":0,"top_k":1,"rrf_k":30,"seed_nodes":["a","a"],"allowed_ids":["a"]}"#);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "a");
    assert!(
        db.search(
            nodedb_graph_rag_search,
            r#"{"vector_k":0,"seed_nodes":["a"],"allowed_ids":[]}"#
        )
        .is_empty()
    );
}

#[test]
fn search_errors_preserve_output_and_record_reason() {
    let db = Db::open();
    for (search, json) in [
        (nodedb_hybrid_search as Search, "{"),
        (
            nodedb_graph_rag_search as Search,
            r#"{"vector_k":0,"seed_nodes":[""]}"#,
        ),
    ] {
        let params = CString::new(json).unwrap();
        let sentinel = std::ptr::dangling_mut::<c_char>();
        let mut out = sentinel;
        unsafe {
            assert_eq!(search(db.0, params.as_ptr(), &mut out), NODEDB_ERR_FAILED);
            assert_eq!(out, sentinel);
            let error = nodedb_last_error(db.0);
            assert!(!error.is_null());
            assert!(!CStr::from_ptr(error).to_bytes().is_empty());
            nodedb_free_string(error);
            assert_eq!(search(db.0, std::ptr::null(), &mut out), NODEDB_ERR_NULL);
            assert_eq!(
                search(std::ptr::null_mut(), params.as_ptr(), &mut out),
                NODEDB_ERR_NULL
            );
            assert_eq!(
                search(db.0, params.as_ptr(), std::ptr::null_mut()),
                NODEDB_ERR_NULL
            );
            let invalid_utf8 = [255_u8, 0];
            assert_eq!(
                search(db.0, invalid_utf8.as_ptr().cast(), &mut out),
                NODEDB_ERR_UTF8
            );
        }
    }
}

#[test]
fn hybrid_json_fuses_sources_and_deserializes_metadata_filters() {
    let db = Db::open();
    let collection = CString::new("docs").unwrap();
    for (id, title, embedding) in [
        ("both", "rust", [1.0_f32, 0.0]),
        ("vector", "other", [0.8, 0.2]),
        ("text", "rust", [0.0, 1.0]),
    ] {
        let body = CString::new(format!(
            r#"{{"id":"{id}","fields":{{"title":"{title}","category":"keep"}}}}"#
        ))
        .unwrap();
        let id = CString::new(id).unwrap();
        unsafe {
            assert_eq!(
                nodedb_document_put(
                    db.0,
                    collection.as_ptr(),
                    body.as_ptr(),
                    std::ptr::null_mut()
                ),
                NODEDB_OK
            );
            assert_eq!(
                nodedb_vector_insert(
                    db.0,
                    collection.as_ptr(),
                    id.as_ptr(),
                    embedding.as_ptr(),
                    embedding.len()
                ),
                NODEDB_OK
            );
        }
    }
    let results = db.search(nodedb_hybrid_search, r#"{"collection":"docs","query_embedding":[1,0],"query_text":"rust","text_field":"title","vector_k":2,"text_k":2,"top_k":3,"filter":{"Eq":{"field":"category","value":"keep"}},"allowed_ids":["both","vector","text"]}"#);
    assert_eq!(results[0].id, "both");
    assert_eq!(results.len(), 3);
    let filtered = db.search(nodedb_hybrid_search, r#"{"collection":"docs","query_embedding":[1,0],"vector_k":3,"text_k":0,"filter":{"Eq":{"field":"title","value":"other"}}}"#);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, "vector");
}
