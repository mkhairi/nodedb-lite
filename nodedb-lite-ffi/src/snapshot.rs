// SPDX-License-Identifier: Apache-2.0

//! Native snapshot creation and restored database handles.

use std::os::raw::c_char;

use nodedb_lite::NodeDbLite;

use crate::error::record_error;
use crate::handle::NodeDbHandle;
use crate::handle_registry;
use crate::open::{config_with_memory, create_runtime};
use crate::status::{NODEDB_ERR_FAILED, NODEDB_ERR_NULL, NODEDB_ERR_UTF8, NODEDB_OK};
use crate::util::{ffi_guard, handle_ref, ptr_to_str, resolve_encryption};

fn persistent_path<'a>(path: *const c_char, argument: &str) -> Result<&'a str, i32> {
    let Some(path) = ptr_to_str(path) else {
        return if path.is_null() {
            record_error(format!("{argument} is NULL"));
            Err(NODEDB_ERR_NULL)
        } else {
            record_error(format!("{argument} is not valid UTF-8"));
            Err(NODEDB_ERR_UTF8)
        };
    };
    if path == ":memory:" {
        record_error(format!(
            "{argument} is :memory:. Pass a persistent directory path"
        ));
        return Err(NODEDB_ERR_FAILED);
    }
    Ok(path)
}

/// Write a snapshot of a persistent database to a new destination directory.
///
/// Returns an existing status code. Read `nodedb_last_error` for the error reason.
/// `:memory:` handles and destinations are refused.
///
/// # Safety
/// `handle` must be a database token, or NULL. Unknown tokens are refused.
/// `destination` must be NULL or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_snapshot_to(
    handle: *mut NodeDbHandle,
    destination: *const c_char,
) -> i32 {
    ffi_guard(NODEDB_ERR_FAILED, || {
        let Some(h) = handle_ref(handle) else {
            record_error("handle is NULL or unknown. Pass an open database token");
            return NODEDB_ERR_NULL;
        };
        let destination = match persistent_path(destination, "destination") {
            Ok(path) => path,
            Err(status) => return status,
        };
        if h._tmpdir.is_some() {
            record_error("snapshot source is :memory:. Open a persistent database");
            return NODEDB_ERR_FAILED;
        }
        match h.rt.block_on(h.db.snapshot_to(destination)) {
            Ok(()) => NODEDB_OK,
            Err(e) => {
                record_error(e);
                NODEDB_ERR_FAILED
            }
        }
    })
}

/// Restore a snapshot into a new persistent directory and return an open handle.
///
/// Returns NULL on error. Read `nodedb_last_error` for the error reason.
/// The caller frees the returned handle with `nodedb_close`.
/// `memory_mb` of 0 uses the default memory budget.
/// `:memory:` source and destination paths are refused.
///
/// Encryption follows `nodedb_open`: NULL is refused for persistent storage.
/// An empty passphrase explicitly selects plaintext. A nonempty passphrase selects encryption.
///
/// # Safety
/// `source` and `destination` must be NULL or valid null-terminated C strings.
/// `passphrase` must be NULL or a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_restore_from(
    source: *const c_char,
    destination: *const c_char,
    memory_mb: u64,
    passphrase: *const c_char,
) -> *mut NodeDbHandle {
    ffi_guard(std::ptr::null_mut(), || {
        let source = match persistent_path(source, "source") {
            Ok(path) => path,
            Err(_) => return std::ptr::null_mut(),
        };
        let destination = match persistent_path(destination, "destination") {
            Ok(path) => path,
            Err(_) => return std::ptr::null_mut(),
        };
        let Some(encryption) = resolve_encryption(passphrase, false) else {
            if passphrase.is_null() {
                record_error(
                    "persistent plaintext storage refused: pass a passphrase or an empty \
                     string to opt out explicitly",
                );
            } else {
                record_error("passphrase is not valid UTF-8");
            }
            return std::ptr::null_mut();
        };
        let Some(rt) = create_runtime() else {
            return std::ptr::null_mut();
        };
        let db = match rt.block_on(NodeDbLite::restore_from(
            source,
            destination,
            encryption,
            config_with_memory(memory_mb),
        )) {
            Ok(db) => db,
            Err(e) => {
                record_error(e);
                return std::ptr::null_mut();
            }
        };
        handle_registry::insert(NodeDbHandle {
            db,
            rt,
            _tmpdir: None,
        }) as *mut NodeDbHandle
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};
    use std::path::Path;

    use super::*;
    use crate::error::last_error_message;
    use crate::handle::OwnedTempDir;
    use crate::{
        nodedb_close, nodedb_document_get, nodedb_document_put, nodedb_flush, nodedb_free_string,
        nodedb_open,
    };

    struct TestHandle(*mut NodeDbHandle);

    impl Drop for TestHandle {
        fn drop(&mut self) {
            unsafe { nodedb_close(self.0) };
        }
    }

    fn path_string(path: &Path) -> CString {
        CString::new(path.to_str().expect("UTF-8 temporary path")).expect("path without NUL")
    }

    fn open(path: &CString) -> TestHandle {
        let handle = unsafe { nodedb_open(path.as_ptr(), c"".as_ptr()) };
        assert!(!handle.is_null(), "{:?}", last_error_message());
        TestHandle(handle)
    }

    fn assert_error_contains(expected: &str) {
        let message = last_error_message().expect("recorded error");
        assert!(message.contains(expected), "{message}");
    }

    #[test]
    fn snapshot_round_trip_restores_registered_handle_and_documents() {
        let temp = OwnedTempDir::new().expect("temporary directory");
        let source = path_string(&temp.0.join("source"));
        let snapshot = path_string(&temp.0.join("snapshot"));
        let destination = path_string(&temp.0.join("restored"));
        let source_handle = open(&source);
        unsafe {
            assert_eq!(
                nodedb_document_put(
                    source_handle.0,
                    c"notes".as_ptr(),
                    c"{\"id\":\"n1\",\"fields\":{\"title\":\"Hello\"}}".as_ptr(),
                    std::ptr::null_mut(),
                ),
                NODEDB_OK
            );
            assert_eq!(
                nodedb_snapshot_to(source_handle.0, snapshot.as_ptr()),
                NODEDB_OK,
                "{:?}",
                last_error_message()
            );
        }
        drop(source_handle);
        let restored = TestHandle(unsafe {
            nodedb_restore_from(snapshot.as_ptr(), destination.as_ptr(), 32, c"".as_ptr())
        });
        assert!(!restored.0.is_null(), "{:?}", last_error_message());
        assert!(handle_registry::get(restored.0 as u64).is_some());
        assert!(last_error_message().is_none());
        unsafe {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                nodedb_document_get(restored.0, c"notes".as_ptr(), c"n1".as_ptr(), &mut output),
                NODEDB_OK
            );
            assert!(!output.is_null());
            let json = CStr::from_ptr(output).to_str().expect("UTF-8 document");
            let document: nodedb_types::document::Document =
                sonic_rs::from_str(json).expect("document JSON");
            nodedb_free_string(output);
            assert_eq!(
                document.get("title"),
                Some(&nodedb_types::Value::from("Hello"))
            );
            assert_eq!(nodedb_flush(restored.0), NODEDB_OK);
        }
        let token = restored.0;
        drop(restored);
        assert!(handle_registry::get(token as u64).is_none());
        assert_eq!(unsafe { nodedb_flush(token) }, NODEDB_ERR_NULL);
    }

    #[test]
    fn snapshot_refuses_unknown_handles_and_invalid_destinations() {
        let temp = OwnedTempDir::new().expect("temporary directory");
        let source = path_string(&temp.0.join("source"));
        let handle = open(&source);
        let invalid_utf8 = [0xff_u8, 0];
        unsafe {
            assert_eq!(
                nodedb_snapshot_to(std::ptr::null_mut(), c"unused".as_ptr()),
                NODEDB_ERR_NULL
            );
            assert_error_contains("handle");
            assert_eq!(
                nodedb_snapshot_to(u64::MAX as *mut NodeDbHandle, c"unused".as_ptr()),
                NODEDB_ERR_NULL
            );
            assert_error_contains("unknown");
            assert_eq!(
                nodedb_snapshot_to(handle.0, std::ptr::null()),
                NODEDB_ERR_NULL
            );
            assert_error_contains("destination is NULL");
            assert_eq!(
                nodedb_snapshot_to(handle.0, invalid_utf8.as_ptr().cast()),
                NODEDB_ERR_UTF8
            );
            assert_error_contains("destination is not valid UTF-8");
            assert_eq!(
                nodedb_snapshot_to(handle.0, c":memory:".as_ptr()),
                NODEDB_ERR_FAILED
            );
            assert_error_contains("destination is :memory:");
        }
    }

    #[test]
    fn snapshot_refuses_memory_source_and_existing_destination() {
        let temp = OwnedTempDir::new().expect("temporary directory");
        let destination = path_string(&temp.0.join("snapshot"));
        let memory = open(&CString::new(":memory:").expect("memory path"));
        assert_eq!(
            unsafe { nodedb_snapshot_to(memory.0, destination.as_ptr()) },
            NODEDB_ERR_FAILED
        );
        assert_error_contains("source is :memory:");
        assert!(!temp.0.join("snapshot").exists());

        let source = path_string(&temp.0.join("source"));
        let handle = open(&source);
        std::fs::create_dir(temp.0.join("snapshot")).expect("destination directory");
        assert_eq!(
            unsafe { nodedb_snapshot_to(handle.0, destination.as_ptr()) },
            NODEDB_ERR_FAILED
        );
        assert!(last_error_message().is_some());
    }

    #[test]
    fn restore_refuses_null_utf8_memory_and_implicit_plaintext_inputs() {
        let invalid_utf8 = [0xff_u8, 0];
        let cases = [
            (
                std::ptr::null(),
                c"restored".as_ptr(),
                c"".as_ptr(),
                "source is NULL",
            ),
            (
                c"snapshot".as_ptr(),
                std::ptr::null(),
                c"".as_ptr(),
                "destination is NULL",
            ),
            (
                invalid_utf8.as_ptr().cast(),
                c"restored".as_ptr(),
                c"".as_ptr(),
                "source is not valid UTF-8",
            ),
            (
                c"snapshot".as_ptr(),
                invalid_utf8.as_ptr().cast(),
                c"".as_ptr(),
                "destination is not valid UTF-8",
            ),
            (
                c":memory:".as_ptr(),
                c"restored".as_ptr(),
                c"".as_ptr(),
                "source is :memory:",
            ),
            (
                c"snapshot".as_ptr(),
                c":memory:".as_ptr(),
                c"".as_ptr(),
                "destination is :memory:",
            ),
            (
                c"snapshot".as_ptr(),
                c"restored".as_ptr(),
                std::ptr::null(),
                "persistent plaintext storage refused",
            ),
            (
                c"snapshot".as_ptr(),
                c"restored".as_ptr(),
                invalid_utf8.as_ptr().cast(),
                "passphrase is not valid UTF-8",
            ),
        ];
        for (source, destination, passphrase, expected) in cases {
            assert!(unsafe { nodedb_restore_from(source, destination, 0, passphrase) }.is_null());
            assert_error_contains(expected);
        }
    }

    #[test]
    fn restore_records_missing_source_without_creating_destination() {
        let temp = OwnedTempDir::new().expect("temporary directory");
        let source = path_string(&temp.0.join("missing"));
        let destination = path_string(&temp.0.join("restored"));
        assert!(
            unsafe { nodedb_restore_from(source.as_ptr(), destination.as_ptr(), 0, c"".as_ptr()) }
                .is_null()
        );
        assert!(last_error_message().is_some());
        assert!(!temp.0.join("restored").exists());
    }
}
