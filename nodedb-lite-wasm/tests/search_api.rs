// SPDX-License-Identifier: Apache-2.0

use nodedb_lite_wasm::NodeDbLiteWasm;
use nodedb_types::result::SearchResult;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn params(json: &str) -> JsValue {
    js_sys::JSON::parse(json).unwrap()
}

#[wasm_bindgen_test]
async fn hybrid_parameters_preserve_field_scope_and_allowed_ids() {
    let db = NodeDbLiteWasm::open_in_memory().await.unwrap();
    db.document_put("docs", "a", r#"{"title":"rust","body":"context"}"#)
        .await
        .unwrap();
    let result = db.hybrid_search(params(r#"{"collection":"docs","vector_k":0,"text_k":1,"top_k":1,"query_text":"rust","text_field":"title","allowed_ids":["a"]}"#)).await.unwrap();
    let results: Vec<SearchResult> = serde_wasm_bindgen::from_value(result).unwrap();
    assert_eq!(results[0].id, "a");
    for extra in [
        r#""allowed_ids":[]"#,
        r#""allowed_ids":["unknown"]"#,
        r#""text_field":"body""#,
    ] {
        let result = db
            .hybrid_search(params(&format!(
                r#"{{"collection":"docs","vector_k":0,"query_text":"rust",{extra}}}"#
            )))
            .await
            .unwrap();
        assert!(
            serde_wasm_bindgen::from_value::<Vec<SearchResult>>(result)
                .unwrap()
                .is_empty()
        );
    }
}

#[wasm_bindgen_test]
async fn graph_parameters_preserve_seeds_depth_and_reject_invalid_ids() {
    let db = NodeDbLiteWasm::open_in_memory().await.unwrap();
    db.graph_insert_edge("docs", "a", "b", "LINK")
        .await
        .unwrap();
    let result = db.graph_rag_search(params(r#"{"collection":"docs","query":[],"vector_k":0,"graph_depth":1,"top_k":2,"rrf_k":30,"seed_nodes":["a","a"],"allowed_ids":null}"#)).await.unwrap();
    let results: Vec<SearchResult> = serde_wasm_bindgen::from_value(result).unwrap();
    assert_eq!(
        results.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert!(
        db.graph_rag_search(params(r#"{"vector_k":0,"seed_nodes":[""]}"#))
            .await
            .is_err()
    );
    assert!(
        db.hybrid_search(params(r#"{"top_k":"invalid"}"#))
            .await
            .is_err()
    );
}
