// SPDX-License-Identifier: Apache-2.0

//! Readers for the FFI's declared surface.
//!
//! Two surfaces have to agree with the symbols the library exports: the
//! cbindgen-generated C header, and the hand-written Kotlin `external fun`
//! declarations. Both are read here so a single snapshot can pin them.

use std::path::{Path, PathBuf};

/// The crate root, so tests can reach the header, the Kotlin file and the
/// snapshot without depending on the working directory.
pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Strip `/* … */` and `// …` comments, leaving code positions intact enough
/// for statement splitting.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"/*") {
            match src[i..].find("*/") {
                Some(end) => i += end + 2,
                None => break,
            }
        } else if bytes[i..].starts_with(b"//") {
            match src[i..].find('\n') {
                Some(end) => i += end,
                None => break,
            }
        } else {
            out.push(src[i..].chars().next().unwrap_or(' '));
            i += src[i..].chars().next().map_or(1, |c| c.len_utf8());
        }
    }
    out
}

/// Collapse every run of whitespace to a single space and trim.
fn normalize(decl: &str) -> String {
    decl.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every function declaration in the cbindgen-generated C header, normalized
/// to one line each and sorted.
///
/// Declarations, not just names: a return type widening from `uint32_t` to
/// `uint64_t`, or a parameter losing its `const`, keeps the name and the
/// argument count while breaking every caller.
pub fn c_declarations(header: &Path) -> Vec<String> {
    let src = std::fs::read_to_string(header)
        .unwrap_or_else(|e| panic!("read {}: {e}", header.display()));
    let src = strip_comments(&src);

    let mut decls: Vec<String> = src
        .split(';')
        .map(normalize)
        .filter(|stmt| {
            // A function declaration, not a typedef or a macro line.
            stmt.contains('(')
                && !stmt.starts_with('#')
                && !stmt.starts_with("typedef")
                && stmt.contains("nodedb_")
        })
        // Drop anything the preprocessor lines left glued to the front.
        .map(|stmt| match stmt.rfind('#') {
            Some(hash) => normalize(&stmt[hash..][stmt[hash..].find('\n').unwrap_or(0)..]),
            None => stmt,
        })
        .filter(|stmt| !stmt.is_empty())
        .collect();
    decls.sort();
    decls.dedup();
    decls
}

/// One exported JNI entry point.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct JniExport {
    /// Full symbol, e.g. `Java_com_nodedb_lite_NodeDbLite_nativeFlush`.
    pub symbol: String,
    /// The Java-visible method name, e.g. `nativeFlush`.
    pub method: String,
    /// Parameter types after `JNIEnv` and the class/object receiver.
    pub params: Vec<String>,
    /// Declared return type, or `void` when there is none.
    pub ret: String,
}

/// Every `extern "system"` JNI export declared under `src/jni_bridge`.
///
/// Parsed from source because JNI has no generated header to read — which is
/// the reason this surface is the one that has drifted before.
pub fn jni_exports(src_dir: &Path) -> Vec<JniExport> {
    let mut exports = Vec::new();
    let bridge = src_dir.join("jni_bridge");
    let entries =
        std::fs::read_dir(&bridge).unwrap_or_else(|e| panic!("read {}: {e}", bridge.display()));

    for entry in entries {
        let path = entry.expect("read dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        exports.extend(parse_jni_exports(&src));
    }
    exports.sort();
    exports
}

/// Pull the JNI exports out of one source file.
fn parse_jni_exports(src: &str) -> Vec<JniExport> {
    const MARKER: &str = "pub extern \"system\" fn Java_";
    let mut exports = Vec::new();
    let mut rest = src;

    while let Some(start) = rest.find(MARKER) {
        let after = &rest[start + MARKER.len() - "Java_".len()..];
        let Some(open) = after.find('(') else { break };
        let symbol = after[..open].trim().to_string();
        let Some(close) = after.find(')') else { break };

        let params: Vec<String> = strip_comments(&after[open + 1..close])
            .split(',')
            .filter_map(|param| {
                let (_, ty) = param.split_once(':')?;
                Some(normalize(ty))
            })
            // The JNIEnv and the class/object receiver are supplied by the JVM,
            // not by the Kotlin caller.
            .skip(2)
            .collect();

        // `-> T {` when there is a return type; a bare `{` when there is not.
        let tail = &after[close + 1..];
        let ret = match (tail.find("->"), tail.find('{')) {
            (Some(arrow), Some(brace)) if arrow < brace => normalize(&tail[arrow + 2..brace]),
            _ => "void".to_string(),
        };

        let method = symbol.rsplit('_').next().unwrap_or(&symbol).to_string();
        exports.push(JniExport {
            symbol,
            method,
            params,
            ret,
        });
        rest = &after[close..];
    }
    exports
}

/// One `private external fun` declared in the Kotlin binding.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct KotlinExtern {
    pub name: String,
    pub params: Vec<String>,
}

/// Every `external fun` in the Kotlin binding, with its parameter types.
pub fn kotlin_externs(kotlin_file: &Path) -> Vec<KotlinExtern> {
    let src = std::fs::read_to_string(kotlin_file)
        .unwrap_or_else(|e| panic!("read {}: {e}", kotlin_file.display()));
    let src = strip_comments(&src);

    let mut externs = Vec::new();
    let mut rest = src.as_str();
    while let Some(start) = rest.find("external fun ") {
        let after = &rest[start + "external fun ".len()..];
        let Some(open) = after.find('(') else { break };
        let Some(close) = after.find(')') else { break };
        let name = after[..open].trim().to_string();
        let params: Vec<String> = after[open + 1..close]
            .split(',')
            .filter_map(|param| {
                let (_, ty) = param.split_once(':')?;
                Some(normalize(ty))
            })
            .collect();
        externs.push(KotlinExtern { name, params });
        rest = &after[close..];
    }
    externs.sort();
    externs
}
