# FFI ABI compatibility

For anyone writing an adapter — Swift, Crystal, Python, C#, Go — against
`libnodedb_lite_ffi`.

## What is guaranteed

Exported symbol signatures freeze at the **first tagged release**.

Before that tag there are no guarantees: signatures change in place and the
generated header changes with them. After it:

- An existing export never changes shape and never goes away.
- New capability arrives as a new export, never as a new parameter on an
  existing one.

That second rule is what makes an adapter safe against an upgraded library.
A caller compiled against a five-argument function resolves the same symbol
after an in-place change to six, then the callee reads a sixth argument the
caller never set — an arbitrary value treated as a pointer.

## Checking compatibility at load

Call `nodedb_abi_version()` before anything else. It takes no handle and does
not allocate.

```c
#define MY_ADAPTER_ABI 1

if (nodedb_abi_version() != MY_ADAPTER_ABI) {
    /* Refuse to run. Every later call is a guess. */
    return ADAPTER_INCOMPATIBLE;
}
```

Compare for equality, not for "at least". Adding an export does not move the
number, so an equal value means every symbol you know still has the shape you
built against. A different value means at least one of them does not, and
which one is not encoded in the integer.

`nodedb_version()` returns a display string such as `0.1.0+a26ffcd` — use it in
logs and bug reports. Do not parse it for compatibility decisions.

## Knowing what changed between releases

Each release ships `abi/surface.txt` beside the header. It records the full
declaration of every export, C and JNI, and the ABI version they belong to:

```
abi_version 1

[c]
char *nodedb_last_error(struct NodeDbNodeDbHandle *handle)
int32_t nodedb_flush(struct NodeDbNodeDbHandle *handle)
...
```

Diff two releases' copies to see exactly what moved. Lines added are new
exports and are safe to ignore. Lines changed or removed are breaking, and the
`abi_version` line will have moved with them.

## Memory rules

These are not visible in a signature, and they do not change without a major
version.

- A `char *` or `uint8_t *` written to an out-parameter is owned by the caller.
  Release it with `nodedb_free_string`, or `nodedb_free_buf` for a buffer
  (which takes the length the call wrote alongside it).
- Out-parameters are written **only on success**. Initialise them to NULL and
  do not read them after a non-`NODEDB_OK` return.
- Strings passed in are borrowed for the duration of the call. The library
  copies whatever it keeps.
- A handle from `nodedb_open` is released with `nodedb_close`. Closing blocks
  until background tasks stop, so keep it off a UI thread.

## Error detail

Every entry point returns a status code. When it is not `NODEDB_OK`, call
`nodedb_last_error(handle)` for the reason as a caller-owned string, freed with
`nodedb_free_string`.

The slot is thread-local and is cleared at the start of every call. Read it on
the thread that made the failing call, before that thread calls anything else.
`nodedb_open` records its reason too, and accepts a NULL handle — an open that
fails has no handle to pass.
