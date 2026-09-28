// SPDX-License-Identifier: Apache-2.0

//! Bidirectional binding between HNSW node ids and document ids.
//!
//! Each HNSW index (keyed by its index key: `"{collection}"` for the base
//! vector, `"{collection}:{field}"` for a named one) binds every document id
//! to at most one live node, and every node to one document id. Both
//! directions live behind one type, so no caller can update one side only.
//!
//! Bindings are grouped per index key rather than flattened into
//! `"{index_key}:{node}"` strings. A flat string key makes `chat` a prefix of
//! `chat2:*` and of the named key `chat:field:*`, so a prefix match leaks
//! nodes of one index into another. The flat form survives only as the
//! persisted layout ([`VectorIdMap::to_entries`] / [`VectorIdMap::from_entries`]).

use std::collections::HashMap;

/// The node ↔ document bindings of one HNSW index.
#[derive(Debug, Default, Clone)]
pub struct IndexIdMap {
    by_node: HashMap<u32, String>,
    by_doc: HashMap<String, u32>,
}

impl IndexIdMap {
    /// Bind `doc_id` to `node`. Returns the node `doc_id` was bound to before,
    /// when that was a different node: the caller tombstones it. A document
    /// previously bound to `node` loses its binding.
    pub fn bind(&mut self, doc_id: &str, node: u32) -> Option<u32> {
        if let Some(prev_doc) = self.by_node.insert(node, doc_id.to_owned())
            && prev_doc != doc_id
            && self.by_doc.get(&prev_doc) == Some(&node)
        {
            self.by_doc.remove(&prev_doc);
        }
        match self.by_doc.insert(doc_id.to_owned(), node) {
            Some(old) if old != node => {
                self.by_node.remove(&old);
                Some(old)
            }
            _ => None,
        }
    }

    /// The node bound to `doc_id`.
    pub fn node(&self, doc_id: &str) -> Option<u32> {
        self.by_doc.get(doc_id).copied()
    }

    /// The document bound to `node`.
    pub fn doc_id(&self, node: u32) -> Option<&str> {
        self.by_node.get(&node).map(String::as_str)
    }

    /// Remove the binding of `doc_id`. Returns its node.
    pub fn unbind_doc(&mut self, doc_id: &str) -> Option<u32> {
        let node = self.by_doc.remove(doc_id)?;
        self.by_node.remove(&node);
        Some(node)
    }

    /// Remove the binding of `node`. Returns its document id.
    pub fn unbind_node(&mut self, node: u32) -> Option<String> {
        let doc_id = self.by_node.remove(&node)?;
        self.by_doc.remove(&doc_id);
        Some(doc_id)
    }

    /// Number of bound documents.
    pub fn len(&self) -> usize {
        self.by_doc.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_doc.is_empty()
    }

    /// Every `(doc_id, node)` binding, in no fixed order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.by_doc.iter().map(|(doc, node)| (doc.as_str(), *node))
    }
}

/// The node ↔ document bindings of every HNSW index, keyed by index key.
#[derive(Debug, Default, Clone)]
pub struct VectorIdMap {
    indexes: HashMap<String, IndexIdMap>,
}

/// The vectors attached to one document id of one collection.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AttachedVectors {
    /// The base vector, in index `"{collection}"`.
    pub base: bool,
    /// The named vectors' field names, in indexes `"{collection}:{field}"`,
    /// sorted.
    pub named: Vec<String>,
}

impl AttachedVectors {
    /// Whether any vector is attached.
    pub fn any(&self) -> bool {
        self.base || !self.named.is_empty()
    }
}

/// One persisted binding: `("{index_key}:{node}", doc_id, node)`.
pub type PersistedIdEntry = (String, String, u32);

/// The index key of one persisted binding: its composite key minus the
/// `":{node}"` suffix. `None` when the composite key does not end in its own
/// node id.
pub fn persisted_index_key(entry: &PersistedIdEntry) -> Option<&str> {
    let (composite, _, node) = entry;
    composite.strip_suffix(&format!(":{node}"))
}

impl VectorIdMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// The bindings of `index_key`, when it has any.
    pub fn index(&self, index_key: &str) -> Option<&IndexIdMap> {
        self.indexes.get(index_key)
    }

    /// Bind `doc_id` to `node` in `index_key`. Returns the node `doc_id` was
    /// bound to before, when different: the caller tombstones it.
    pub fn bind(&mut self, index_key: &str, doc_id: &str, node: u32) -> Option<u32> {
        self.indexes
            .entry(index_key.to_owned())
            .or_default()
            .bind(doc_id, node)
    }

    /// The node bound to `doc_id` in `index_key`.
    pub fn node(&self, index_key: &str, doc_id: &str) -> Option<u32> {
        self.indexes.get(index_key)?.node(doc_id)
    }

    /// The document bound to `node` in `index_key`.
    pub fn doc_id(&self, index_key: &str, node: u32) -> Option<&str> {
        self.indexes.get(index_key)?.doc_id(node)
    }

    /// Remove the binding of `doc_id` in `index_key`. Returns its node.
    pub fn unbind_doc(&mut self, index_key: &str, doc_id: &str) -> Option<u32> {
        let ids = self.indexes.get_mut(index_key)?;
        let node = ids.unbind_doc(doc_id);
        if ids.is_empty() {
            self.indexes.remove(index_key);
        }
        node
    }

    /// Remove the binding of `node` in `index_key`. Returns its document id.
    pub fn unbind_node(&mut self, index_key: &str, node: u32) -> Option<String> {
        let ids = self.indexes.get_mut(index_key)?;
        let doc_id = ids.unbind_node(node);
        if ids.is_empty() {
            self.indexes.remove(index_key);
        }
        doc_id
    }

    /// The vectors bound to `doc_id` across the base index `collection` and
    /// its named indexes `"{collection}:{field}"`.
    pub fn attached_vectors(&self, collection: &str, doc_id: &str) -> AttachedVectors {
        let mut attached = AttachedVectors::default();
        for (index_key, ids) in &self.indexes {
            if ids.node(doc_id).is_none() {
                continue;
            }
            if index_key == collection {
                attached.base = true;
            } else if let Some(field) = index_key
                .strip_prefix(collection)
                .and_then(|rest| rest.strip_prefix(':'))
            {
                attached.named.push(field.to_owned());
            }
        }
        attached.named.sort();
        attached
    }

    /// Remove every binding of `index_key`.
    pub fn remove_index(&mut self, index_key: &str) {
        self.indexes.remove(index_key);
    }

    /// Replace every binding of `index_key` with `ids`.
    pub fn replace_index(&mut self, index_key: &str, ids: IndexIdMap) {
        if ids.is_empty() {
            self.indexes.remove(index_key);
        } else {
            self.indexes.insert(index_key.to_owned(), ids);
        }
    }

    /// Total bindings across every index.
    pub fn len(&self) -> usize {
        self.indexes.values().map(IndexIdMap::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// The persisted layout: one `("{index_key}:{node}", doc_id, node)` per
    /// binding.
    pub fn to_entries(&self) -> Vec<PersistedIdEntry> {
        let mut out = Vec::with_capacity(self.len());
        for (index_key, ids) in &self.indexes {
            for (doc_id, node) in ids.iter() {
                out.push((format!("{index_key}:{node}"), doc_id.to_owned(), node));
            }
        }
        out
    }

    /// Rebuild from the persisted layout.
    ///
    /// The index key is the composite key minus its `":{node}"` suffix; an
    /// entry whose composite key does not end in its own node id is skipped.
    /// When one document holds several nodes in one index, the highest node
    /// wins: node ids grow with each insert, so it is the latest vector. The
    /// other nodes are returned as `(index_key, node)` for the caller to
    /// tombstone.
    pub fn from_entries(entries: Vec<PersistedIdEntry>) -> (Self, Vec<(String, u32)>) {
        let mut parsed: Vec<(String, String, u32)> = entries
            .into_iter()
            .filter_map(|entry| {
                let index_key = persisted_index_key(&entry)?.to_owned();
                let (_, doc_id, node) = entry;
                Some((index_key, doc_id, node))
            })
            .collect();
        parsed.sort_by_key(|(_, _, node)| *node);
        let mut map = Self::new();
        let mut displaced = Vec::new();
        for (index_key, doc_id, node) in parsed {
            if let Some(old) = map.bind(&index_key, &doc_id, node) {
                displaced.push((index_key, old));
            }
        }
        (map, displaced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebinding_a_document_returns_the_displaced_node() {
        let mut map = VectorIdMap::new();
        assert_eq!(map.bind("c", "a", 0), None);
        assert_eq!(map.bind("c", "a", 1), Some(0));
        assert_eq!(map.node("c", "a"), Some(1));
        assert_eq!(map.doc_id("c", 0), None);
        assert_eq!(map.doc_id("c", 1), Some("a"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn indexes_with_shared_prefixes_stay_separate() {
        let mut map = VectorIdMap::new();
        map.bind("chat", "a", 0);
        map.bind("chat2", "b", 0);
        map.bind("chat:emb", "c", 0);
        assert_eq!(map.doc_id("chat", 0), Some("a"));
        assert_eq!(map.index("chat").map(IndexIdMap::len), Some(1));
        map.remove_index("chat");
        assert_eq!(map.doc_id("chat2", 0), Some("b"));
        assert_eq!(map.doc_id("chat:emb", 0), Some("c"));
    }

    #[test]
    fn unbind_removes_both_directions() {
        let mut map = VectorIdMap::new();
        map.bind("c", "a", 3);
        assert_eq!(map.unbind_doc("c", "a"), Some(3));
        assert_eq!(map.doc_id("c", 3), None);
        map.bind("c", "b", 4);
        assert_eq!(map.unbind_node("c", 4), Some("b".to_owned()));
        assert_eq!(map.node("c", "b"), None);
        assert!(map.is_empty());
    }

    #[test]
    fn persisted_entries_round_trip_named_keys() {
        let mut map = VectorIdMap::new();
        map.bind("chat", "x:1", 0);
        map.bind("chat:emb", "y", 7);
        let (restored, displaced) = VectorIdMap::from_entries(map.to_entries());
        assert!(displaced.is_empty());
        assert_eq!(restored.doc_id("chat", 0), Some("x:1"));
        assert_eq!(restored.doc_id("chat:emb", 7), Some("y"));
    }

    #[test]
    fn attached_vectors_cover_base_and_named_indexes_only() {
        let mut map = VectorIdMap::new();
        map.bind("chat", "a", 0);
        map.bind("chat:emb", "a", 0);
        map.bind("chat2", "a", 0);
        map.bind("chat:title", "b", 0);
        assert_eq!(
            map.attached_vectors("chat", "a"),
            AttachedVectors {
                base: true,
                named: vec!["emb".to_owned()],
            }
        );
        assert!(!map.attached_vectors("chat", "zzz").any());
    }

    #[test]
    fn persisted_duplicates_keep_the_latest_node() {
        let entries = vec![
            ("c:5".to_owned(), "a".to_owned(), 5),
            ("c:2".to_owned(), "a".to_owned(), 2),
        ];
        let (map, displaced) = VectorIdMap::from_entries(entries);
        assert_eq!(map.node("c", "a"), Some(5));
        assert_eq!(displaced, vec![("c".to_owned(), 2)]);
    }
}
