fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Re-run when the git HEAD moves so the embedded SHA never goes stale in
    // incremental builds. .git/HEAD covers branch switches and detached-HEAD
    // rewrites; .git/refs covers ref updates (e.g. new commits on the current
    // branch); packed-refs is used when refs are packed. Absent .git (source
    // tarballs) skips all three silently.
    for git_meta in [".git/HEAD", ".git/refs", ".git/packed-refs"] {
        let p = std::path::Path::new(&crate_dir).join(git_meta);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }

    // Embed the git short-sha for nodedb_version() (best effort — builds from
    // a source tarball have no .git and just omit the suffix).
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&crate_dir)
        .output()
        && output.status.success()
    {
        let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !sha.is_empty() {
            println!("cargo:rustc-env=NODEDB_GIT_SHA={sha}");
        }
    }

    // Create include directory.
    let _ = std::fs::create_dir_all(format!("{crate_dir}/include"));

    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

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
