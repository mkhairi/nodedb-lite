// SPDX-License-Identifier: Apache-2.0

//! The HNSW slot ↔ document-id mapping, indexed both ways.
//!
//! Search needs slot → document id: HNSW answers in integer node ids, and the
//! caller asked about documents. Insert needs the other direction — "does this
//! collection already hold a vector for this document?" — because an insert
//! that cannot answer it appends, and the same document then comes back twice
//! from one ANN query.
//!
//! Both directions are kept here, behind methods, so they cannot drift: a
//! reverse index maintained by hand at each of the six call sites that bind a
//! slot would go stale at the first one that forgot, and a stale reverse index
//! silently reintroduces the duplicate it exists to prevent.

use std::collections::HashMap;

/// Slot ↔ document id for every loaded vector index.
///
/// Keys are composite (`{index_key}:{slot}` one way, `{index_key}:{doc_id}` the
/// other) rather than nested maps, matching the flat form the flush blob has
/// always been written in.
#[derive(Debug, Default)]
pub(crate) struct VectorIdMap {
    /// `{index_key}:{slot}` → (document id, slot)
    by_slot: HashMap<String, (String, u32)>,
    /// `{index_key}:{doc_id}` → slot
    by_doc: HashMap<String, u32>,
}

impl VectorIdMap {
    /// Rebuild from the flat slot-keyed form the flush blob stores.
    ///
    /// The reverse direction is derived rather than persisted: it is a pure
    /// function of the forward map, so storing it would add a second thing to
    /// keep consistent across restarts for no information gained.
    pub(crate) fn from_slots(by_slot: HashMap<String, (String, u32)>) -> Self {
        let mut map = Self {
            by_doc: HashMap::with_capacity(by_slot.len()),
            by_slot,
        };
        map.rebuild_by_doc();
        map
    }

    fn rebuild_by_doc(&mut self) {
        self.by_doc.clear();
        for (composite, (doc_id, slot)) in &self.by_slot {
            let Some((index_key, _)) = composite.rsplit_once(':') else {
                continue;
            };
            self.by_doc.insert(format!("{index_key}:{doc_id}"), *slot);
        }
    }

    /// Bind `slot` in `index_key` to `doc_id`, replacing any earlier binding
    /// for that document.
    pub(crate) fn bind(&mut self, index_key: &str, doc_id: &str, slot: u32) {
        if let Some(previous) = self.by_doc.insert(format!("{index_key}:{doc_id}"), slot)
            && previous != slot
        {
            self.by_slot.remove(&format!("{index_key}:{previous}"));
        }
        self.by_slot
            .insert(format!("{index_key}:{slot}"), (doc_id.to_string(), slot));
    }

    /// The slot currently holding `doc_id` in `index_key`, if any.
    pub(crate) fn slot_of(&self, index_key: &str, doc_id: &str) -> Option<u32> {
        self.by_doc.get(&format!("{index_key}:{doc_id}")).copied()
    }

    /// The document a slot belongs to, by the composite `{index_key}:{slot}`.
    pub(crate) fn get(&self, composite: &str) -> Option<&(String, u32)> {
        self.by_slot.get(composite)
    }

    /// Drop every binding belonging to `index_key`.
    pub(crate) fn clear_index(&mut self, index_key: &str) {
        let prefix = format!("{index_key}:");
        self.by_slot.retain(|k, _| !k.starts_with(&prefix));
        self.by_doc.retain(|k, _| !k.starts_with(&prefix));
    }

    /// Replace every binding for `index_key` with `entries`, keyed by the same
    /// composite `{index_key}:{slot}` form. Used when an index is rebuilt from
    /// durable vectors and its slots are renumbered wholesale.
    pub(crate) fn replace_index(
        &mut self,
        index_key: &str,
        entries: HashMap<String, (String, u32)>,
    ) {
        self.clear_index(index_key);
        for (composite, (doc_id, slot)) in entries {
            self.by_doc.insert(format!("{index_key}:{doc_id}"), slot);
            self.by_slot.insert(composite, (doc_id, slot));
        }
    }

    /// Every binding, slot-keyed by the composite `{index_key}:{slot}`.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &(String, u32))> {
        self.by_slot.iter()
    }

    /// Every composite slot key.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &String> {
        self.by_slot.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebinding_a_document_frees_its_old_slot() {
        let mut map = VectorIdMap::default();
        map.bind("docs", "01AAA", 0);
        map.bind("docs", "01AAA", 1);

        assert_eq!(map.slot_of("docs", "01AAA"), Some(1));
        assert_eq!(
            map.get("docs:0"),
            None,
            "the superseded slot must not still name the document"
        );
        assert_eq!(map.iter().count(), 1, "one document occupies one slot");
    }

    #[test]
    fn the_same_document_id_in_two_indexes_is_two_bindings() {
        let mut map = VectorIdMap::default();
        map.bind("docs", "01AAA", 0);
        map.bind("notes", "01AAA", 0);

        assert_eq!(map.slot_of("docs", "01AAA"), Some(0));
        assert_eq!(map.slot_of("notes", "01AAA"), Some(0));
        map.clear_index("docs");
        assert_eq!(
            map.slot_of("notes", "01AAA"),
            Some(0),
            "clearing one index must leave the other's binding alone"
        );
        assert_eq!(map.slot_of("docs", "01AAA"), None);
    }

    #[test]
    fn the_reverse_index_survives_a_restore() {
        let mut stored = HashMap::new();
        stored.insert("docs:7".to_string(), ("01AAA".to_string(), 7));
        let map = VectorIdMap::from_slots(stored);

        assert_eq!(map.slot_of("docs", "01AAA"), Some(7));
    }
}
