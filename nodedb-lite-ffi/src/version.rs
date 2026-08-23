// SPDX-License-Identifier: Apache-2.0

//! Version and ABI exports for the C FFI surface.
//!
//! Bindings load the library by name (`-lnodedb_lite_ffi` / dlopen) and
//! previously had no way to detect which build they were talking to — a stale
//! `.so` on the library path failed late and confusingly. These exports let
//! bindings check version/ABI skew at runtime, before any other call.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::OnceLock;

/// Integer ABI version, bumped only on breaking FFI changes.
///
/// A binding can compare this against the version it was compiled against and
/// fail fast with a clear message instead of dying on a missing symbol.
const NODEDB_ABI_VERSION: u32 = 1;

/// The full version string, e.g. `"0.1.0+ee9ccdd"` (CARGO_PKG_VERSION plus the
/// git short-sha when the build ran inside a git checkout).
///
/// Allocated once on first call; the pointer is valid for the process
/// lifetime and must NOT be freed by the caller.
fn version_string() -> &'static CString {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION.get_or_init(|| {
        let s = match option_env!("NODEDB_GIT_SHA") {
            Some(sha) => format!("{}+{}", env!("CARGO_PKG_VERSION"), sha),
            None => env!("CARGO_PKG_VERSION").to_string(),
        };
        // Version strings are controlled by Cargo/build.rs — no interior NULs.
        CString::new(s).expect("version string must not contain NUL")
    })
}

/// Return the library version as a static string, e.g. `"0.1.0+ee9ccdd"`.
///
/// Safe to call before any open; no allocation per call.
/// The returned pointer is owned by the library — do not free it.
#[unsafe(no_mangle)]
pub extern "C" fn nodedb_version() -> *const c_char {
    version_string().as_ptr()
}

/// Return the ABI version as an integer.
///
/// Bumped on breaking FFI changes. Bindings should compare this against their
/// compile-time expectation and refuse to run on mismatch.
#[unsafe(no_mangle)]
pub extern "C" fn nodedb_abi_version() -> u32 {
    NODEDB_ABI_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn version_is_semver_shaped() {
        let v = unsafe { CStr::from_ptr(nodedb_version()) }
            .to_str()
            .unwrap();
        let base = v.split(['+', '-']).next().unwrap();
        let parts: Vec<&str> = base.split('.').collect();
        assert!(parts.len() >= 2, "expected semver-ish version, got: {v}");
        assert!(
            parts
                .iter()
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        );
    }

    #[test]
    fn abi_version_is_positive() {
        assert!(nodedb_abi_version() > 0);
    }

    #[test]
    fn version_is_stable_pointer() {
        // The pointer must be stable across calls (static storage).
        assert_eq!(nodedb_version(), nodedb_version());
    }
}
