// SPDX-License-Identifier: Apache-2.0

//! Key layout of the durable per-document vector rows.
//!
//! Key: `vr:` + the index key's byte length as 8 lowercase hex digits + the
//! index key + the document id. The index key is `"{collection}"` for a base
//! vector and `"{collection}:{field}"` for a named one. The length prefix
//! fixes where the index key ends, so no index key is a key prefix of
//! another: `chat`, `chat2` and `chat:emb` each scan only their own rows,
//! whatever bytes the document ids contain.
//!
//! The `vr:` prefix is disjoint from the other keys in `Namespace::Vector`
//! (`hnsw:<name>` checkpoints, `hnsw_id_map`, `sidecar:<name>`), and from the
//! earlier `v:<index_key>:<doc_id>` layout that `migrate` rewrites.

/// Key prefix of every row in the current layout.
pub(crate) const ROW_PREFIX: &str = "vr:";

/// Key prefix of every row in the earlier, ambiguous layout.
pub(crate) const LEGACY_ROW_PREFIX: &str = "v:";

/// Hex digits of the index-key length field.
const LEN_DIGITS: usize = 8;

/// Key-space prefix of one index's rows.
pub(crate) fn index_prefix(index_key: &str) -> Vec<u8> {
    format!("{ROW_PREFIX}{:08x}{index_key}", index_key.len()).into_bytes()
}

/// Durable key of one document's vector in one index.
pub(crate) fn key(index_key: &str, doc_id: &str) -> Vec<u8> {
    let mut k = index_prefix(index_key);
    k.extend_from_slice(doc_id.as_bytes());
    k
}

/// Split a current-layout key into `(index_key, doc_id)`. `None` for a key
/// of another layout or a malformed one.
pub(crate) fn parse_key(row_key: &[u8]) -> Option<(&str, &str)> {
    let rest = row_key.strip_prefix(ROW_PREFIX.as_bytes())?;
    let len_field = std::str::from_utf8(rest.get(..LEN_DIGITS)?).ok()?;
    let len = usize::from_str_radix(len_field, 16).ok()?;
    let body = rest.get(LEN_DIGITS..)?;
    let index_key = std::str::from_utf8(body.get(..len)?).ok()?;
    let doc_id = std::str::from_utf8(body.get(len..)?).ok()?;
    Some((index_key, doc_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip_with_separators_in_ids() {
        let k = key("chat:emb", "a:b");
        assert_eq!(parse_key(&k), Some(("chat:emb", "a:b")));
    }

    #[test]
    fn base_and_named_prefixes_do_not_alias() {
        let base = index_prefix("chat");
        assert!(!key("chat:emb", "b").starts_with(&base));
        assert!(!key("chat2", "b").starts_with(&base));
        assert!(key("chat", "emb:b").starts_with(&base));
    }

    #[test]
    fn prefix_is_disjoint_from_other_vector_keys() {
        let k = key("entries", "abc");
        assert!(!k.starts_with(b"hnsw:"));
        assert!(!k.starts_with(LEGACY_ROW_PREFIX.as_bytes()));
        assert_ne!(k.as_slice(), b"hnsw_id_map");
        assert_eq!(parse_key(b"v:entries:abc"), None);
    }
}
