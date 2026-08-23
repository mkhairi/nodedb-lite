// SPDX-License-Identifier: Apache-2.0

//! The exported surface is a contract, and this file is where it is pinned.
//!
//! Adapters in other languages and other repositories compile against these
//! symbols. A signature that changes without anyone noticing gives those
//! callers a library whose functions no longer match the declarations they
//! built against — the same symbol, a different shape, and a crash at the
//! first call rather than a load error.
//!
//! `abi/surface.txt` records the whole declared surface. Changing an export
//! means updating that file in the same commit, which puts the change in the
//! diff where a reviewer sees it.
//!
//! To re-record after a deliberate change:
//!
//! ```text
//! UPDATE_ABI_SNAPSHOT=1 cargo nextest run -p nodedb-lite-ffi -E 'binary(abi_surface)'
//! ```

mod common;

use std::path::PathBuf;

use common::{JniExport, KotlinExtern, c_declarations, crate_dir, jni_exports, kotlin_externs};

/// Set this to rewrite the snapshot instead of failing on a difference.
const UPDATE_VAR: &str = "UPDATE_ABI_SNAPSHOT";

fn snapshot_path() -> PathBuf {
    crate_dir().join("abi/surface.txt")
}

/// Render the current surface in the snapshot's format.
fn render_surface() -> String {
    let dir = crate_dir();
    let mut out = String::new();

    out.push_str(
        "# NodeDB-Lite exported FFI surface. Generated — see tests/abi_surface.rs.\n\
         #\n\
         # Regenerate: UPDATE_ABI_SNAPSHOT=1 cargo nextest run -p nodedb-lite-ffi -E 'binary(abi_surface)'\n\
         #\n\
         # Adding an export is backward-compatible and needs no abi_version bump.\n\
         # Changing or removing one is breaking: bump nodedb_abi_version in the\n\
         # same commit. Signatures freeze at the first tagged release.\n\n",
    );

    out.push_str(&format!(
        "abi_version {}\n\n",
        nodedb_lite_ffi::nodedb_abi_version()
    ));

    out.push_str("[c]\n");
    for decl in c_declarations(&dir.join("include/nodedb_lite.h")) {
        out.push_str(&decl);
        out.push('\n');
    }

    out.push_str("\n[jni]\n");
    for export in jni_exports(&dir.join("src")) {
        out.push_str(&format!(
            "{}({}) -> {}\n",
            export.symbol,
            export.params.join(", "),
            export.ret
        ));
    }

    out
}

/// The exported surface must match what is recorded in `abi/surface.txt`.
#[test]
fn surface_matches_snapshot() {
    let current = render_surface();
    let path = snapshot_path();

    if std::env::var_os(UPDATE_VAR).is_some() {
        std::fs::create_dir_all(path.parent().expect("snapshot has a parent"))
            .expect("create abi directory");
        std::fs::write(&path, &current).expect("write snapshot");
        return;
    }

    let recorded = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\n\nRecord it with:\n  {UPDATE_VAR}=1 cargo nextest run \
             -p nodedb-lite-ffi -E 'binary(abi_surface)'",
            path.display()
        )
    });

    if recorded != current {
        panic!(
            "The exported FFI surface no longer matches {}.\n\n\
             If the change is deliberate, re-record it and review the diff:\n  \
             {UPDATE_VAR}=1 cargo nextest run -p nodedb-lite-ffi -E 'binary(abi_surface)'\n\n\
             Adding an export is backward-compatible. Changing or removing one is \
             breaking — bump nodedb_abi_version in the same commit.\n\n\
             --- recorded\n{recorded}\n--- current\n{current}",
            path.display()
        );
    }
}

/// The snapshot must record the ABI version the library actually reports.
///
/// Without this the recorded number drifts from the exported one, and an
/// adapter comparing against the snapshot reads a version no build returns.
#[test]
fn snapshot_records_the_exported_abi_version() {
    let recorded = std::fs::read_to_string(snapshot_path()).expect("read snapshot");
    let line = recorded
        .lines()
        .find_map(|line| line.strip_prefix("abi_version "))
        .expect("snapshot must carry an abi_version line");
    assert_eq!(
        line.trim()
            .parse::<u32>()
            .expect("abi_version is an integer"),
        nodedb_lite_ffi::nodedb_abi_version(),
        "snapshot abi_version disagrees with nodedb_abi_version()"
    );
}

/// Every JNI export must have a Kotlin declaration, and vice versa.
///
/// The C header is generated, so it cannot disagree with the symbols. The
/// Kotlin declarations are written by hand against symbols nothing checks —
/// which is how an `external fun` once lost a parameter the native side still
/// read, turning every call into a wild pointer read.
#[test]
fn kotlin_declares_every_jni_export() {
    let exports = jni_exports(&crate_dir().join("src"));
    let externs = kotlin_externs(&crate_dir().join("kotlin/com/nodedb/lite/NodeDbLite.kt"));

    let exported: Vec<&str> = exports.iter().map(|e| e.method.as_str()).collect();
    let declared: Vec<&str> = externs.iter().map(|e| e.name.as_str()).collect();

    let undeclared: Vec<&&str> = exported
        .iter()
        .filter(|method| !declared.contains(method))
        .collect();
    assert!(
        undeclared.is_empty(),
        "JNI exports with no Kotlin declaration — Android cannot reach them: {undeclared:?}"
    );

    let unbacked: Vec<&&str> = declared
        .iter()
        .filter(|name| !exported.contains(name))
        .collect();
    assert!(
        unbacked.is_empty(),
        "Kotlin declares externs with no JNI export — these fail at link time: {unbacked:?}"
    );
}

/// Each Kotlin declaration must take exactly the arguments its JNI export reads.
///
/// The JVM supplies `JNIEnv` and the receiver, so the Kotlin parameter count is
/// the native one minus those two. A short declaration leaves the native side
/// reading a register the caller never set.
#[test]
fn kotlin_arity_matches_every_jni_export() {
    let exports = jni_exports(&crate_dir().join("src"));
    let externs = kotlin_externs(&crate_dir().join("kotlin/com/nodedb/lite/NodeDbLite.kt"));

    let mismatched: Vec<String> = exports
        .iter()
        .filter_map(|export: &JniExport| {
            let declared: &KotlinExtern = externs.iter().find(|e| e.name == export.method)?;
            (declared.params.len() != export.params.len()).then(|| {
                format!(
                    "{}: native takes {} argument(s) {:?}, Kotlin declares {}",
                    export.method,
                    export.params.len(),
                    export.params,
                    declared.params.len()
                )
            })
        })
        .collect();

    assert!(
        mismatched.is_empty(),
        "Kotlin declarations disagree with their JNI exports: {mismatched:#?}"
    );
}
