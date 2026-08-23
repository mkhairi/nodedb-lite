// SPDX-License-Identifier: Apache-2.0

//! Last-error surface for the C FFI.
//!
//! Every FFI entry point returns a bare status code (`-3` for almost every
//! failure mode) and `nodedb_open` returns NULL for every failure mode, so an
//! embedder cannot distinguish "wrong passphrase" from "corrupt store" from
//! "bad path". This module records the `Display` of the error that produced
//! the last non-OK result in a thread-local slot, retrievable via
//! `nodedb_last_error`.

use std::cell::RefCell;
use std::ffi::CString;
use std::fmt::Display;
use std::os::raw::c_char;

use crate::ffi_guard_keep_error;
use crate::handle::NodeDbHandle;

thread_local! {
    /// Most recent error message on this thread, if any.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Record the error message for the most recent failure on this thread.
///
/// Called by every FFI entry point before it returns a non-OK status.
pub(crate) fn record_error(err: impl Display) {
    let msg = err.to_string();
    // Truncate defensively: error strings must stay small enough for an
    // embedder to surface in a UI/log line. Interior NULs are replaced rather
    // than rejected so a bad message still surfaces (CString would otherwise
    // reject the whole string and the slot would stay empty).
    let truncated: String = msg
        .chars()
        .take(512)
        .map(|c| if c == '\0' { '?' } else { c })
        .collect();
    if let Ok(cs) = CString::new(truncated) {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(cs));
    }
}

/// Clear the recorded error (called before an operation starts, so a stale
/// error from a previous call is never attributed to a successful one).
pub(crate) fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Return the most recent error message on this thread as an owned `String`,
/// or `None` when no error is recorded.
///
/// Used by bindings that marshal strings themselves (the JNI bridge), so they
/// do not have to round-trip through the C allocation.
pub(crate) fn last_error_message() -> Option<String> {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|cs| cs.to_string_lossy().into_owned())
    })
}

/// Return the most recent error message on this thread as a C string the
/// caller frees with `nodedb_free_string`, or NULL when no error is recorded.
///
/// The returned string is a copy owned by the caller — the thread-local slot
/// can be overwritten by later calls without dangling the pointer.
///
/// `handle` may be NULL or a handle from a failed `nodedb_open` (open failures
/// happen before a valid handle exists); the error slot is thread-local, so it
/// is ignored.
///
/// # Safety
/// `handle` may be NULL or any token; it is never dereferenced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nodedb_last_error(handle: *mut NodeDbHandle) -> *mut c_char {
    // Named, not `_handle`: cbindgen copies the parameter name into the public
    // header. The slot is thread-local, so the token itself is unused.
    let _ = handle;
    ffi_guard_keep_error(std::ptr::null_mut(), || {
        LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
            // Clone: the caller owns this allocation and frees it with
            // nodedb_free_string. The thread-local slot stays untouched.
            Some(cs) => cs.clone().into_raw(),
            None => std::ptr::null_mut(),
        })
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn record_then_retrieve() {
        clear_error();
        assert!(unsafe { nodedb_last_error(std::ptr::null_mut()) }.is_null());

        record_error("wrong passphrase");
        let ptr = unsafe { nodedb_last_error(std::ptr::null_mut()) };
        assert!(!ptr.is_null());
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "wrong passphrase");
        unsafe { drop(CString::from_raw(ptr)) };
    }

    #[test]
    fn clear_after_record() {
        record_error("corrupt store");
        clear_error();
        assert!(unsafe { nodedb_last_error(std::ptr::null_mut()) }.is_null());
    }

    #[test]
    fn copy_is_independent_of_slot() {
        record_error("first");
        let a = unsafe { nodedb_last_error(std::ptr::null_mut()) };
        record_error("second");
        let b = unsafe { nodedb_last_error(std::ptr::null_mut()) };
        // The first copy must still be readable after the slot moved on.
        assert_eq!(unsafe { CStr::from_ptr(a) }.to_str().unwrap(), "first");
        assert_eq!(unsafe { CStr::from_ptr(b) }.to_str().unwrap(), "second");
        unsafe {
            drop(CString::from_raw(a));
            drop(CString::from_raw(b));
        }
    }

    #[test]
    fn interior_nul_is_replaced_not_dropped() {
        clear_error();
        record_error("bad\0nul");
        let ptr = unsafe { nodedb_last_error(std::ptr::null_mut()) };
        assert!(
            !ptr.is_null(),
            "NUL-containing message must still be recorded"
        );
        let got = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(got, "bad?nul");
        unsafe { drop(CString::from_raw(ptr)) };
    }
}
