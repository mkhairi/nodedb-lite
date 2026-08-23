use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Emitting any rerun-if-changed replaces cargo's default "rerun when any
    // file in the package changed", so every input this script reads must be
    // listed. Missing one leaves the checked-in header stale.
    println!("cargo:rerun-if-changed={crate_dir}/src");
    println!("cargo:rerun-if-changed={crate_dir}/cbindgen.toml");
    println!("cargo:rerun-if-changed={crate_dir}/Cargo.toml");

    emit_git_rerun_paths(&crate_dir);
    emit_git_sha(&crate_dir);

    let _ = std::fs::create_dir_all(format!("{crate_dir}/include"));

    let config =
        cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml")).unwrap_or_default();

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(format!("{crate_dir}/include/nodedb_lite.h"));
        }
        Err(e) => {
            eprintln!("cargo:warning=cbindgen failed to generate C bindings: {e}");
        }
    }
}

/// Track the files that change when HEAD moves, so the embedded sha stays fresh.
///
/// A source tarball has no git directory and tracks nothing here.
fn emit_git_rerun_paths(crate_dir: &str) {
    // The git directory sits above this crate, and is a `worktrees/<name>` path
    // under a linked worktree. Ask git rather than guessing at `<crate>/.git`.
    let Some(git_dir) = git_path(crate_dir, "--absolute-git-dir") else {
        return;
    };
    emit_if_exists(&git_dir.join("HEAD"));

    // A linked worktree keeps its own HEAD but shares refs with the main
    // checkout, so refs resolve against the common directory.
    let common_dir = git_path(crate_dir, "--git-common-dir").unwrap_or(git_dir);
    emit_if_exists(&common_dir.join("packed-refs"));

    // The single ref HEAD points at. Tracking all of `refs/` would rebuild on
    // every fetch that moves an unrelated remote-tracking branch.
    if let Some(head_ref) = git_output(crate_dir, &["rev-parse", "--symbolic-full-name", "HEAD"]) {
        emit_if_exists(&common_dir.join(head_ref));
    }
}

/// Embed the git short-sha for `nodedb_version()`. Absent git omits the suffix.
fn emit_git_sha(crate_dir: &str) {
    if let Some(sha) = git_output(crate_dir, &["rev-parse", "--short", "HEAD"]) {
        println!("cargo:rustc-env=NODEDB_GIT_SHA={sha}");
    }
}

/// Resolve a `git rev-parse` path option to an absolute path.
///
/// git answers some of these relative to the working directory.
fn git_path(crate_dir: &str, opt: &str) -> Option<PathBuf> {
    let raw = git_output(crate_dir, &["rev-parse", opt])?;
    let path = Path::new(&raw);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        Path::new(crate_dir).join(path)
    };
    path.canonicalize().ok()
}

/// Run `git` in `crate_dir`, returning trimmed stdout when it succeeds.
fn git_output(crate_dir: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(crate_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn emit_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
