// SPDX-License-Identifier: BUSL-1.1

//! Read paths, collection-name registry, and conflict-policy binding.

use loro::LoroValue;

use super::types::CrdtEngine;

impl CrdtEngine {
    /// Read a single field from a row without cloning the entire row.
    ///
    /// Fast path for KV reads: avoids `get_deep_value()` and returns
    /// only the requested field.
    pub fn read_field(&self, collection: &str, doc_id: &str, field: &str) -> Option<LoroValue> {
        self.states
            .get(collection)?
            .read_field(collection, doc_id, field)
    }

    // ─── Reads ───────────────────────────────────────────────────────

    /// Read a document's fields.
    pub fn read(&self, collection: &str, doc_id: &str) -> Option<LoroValue> {
        self.states.get(collection)?.read_row(collection, doc_id)
    }

    /// Check if a document exists.
    pub fn exists(&self, collection: &str, doc_id: &str) -> bool {
        self.states
            .get(collection)
            .is_some_and(|s| s.row_exists(collection, doc_id))
    }

    /// List all document IDs in a collection.
    pub fn list_ids(&self, collection: &str) -> Vec<String> {
        self.states
            .get(collection)
            .map(|s| s.row_ids(collection))
            .unwrap_or_default()
    }
    /// The Loro peer id a collection's document authors under, if it has one.
    ///
    /// Derived from this engine's base peer id, so it changes with a peer-id
    /// rotation — exposed so callers can assert the rotation actually reached
    /// the documents rather than only the engine's own field.
    pub fn state_peer_id(&self, collection: &str) -> Option<u64> {
        self.states.get(collection).map(|s| s.peer_id())
    }

    /// Register a collection name so it appears in `collection_names()` even
    /// before any document has been inserted into it.
    ///
    /// This is needed for bitemporal document collections created via DDL: the
    /// bitemporal flag is persisted to `Namespace::Meta`, but the collection
    /// has no Loro document until the first `upsert`.  Calling this
    /// method ensures the SQL catalog can resolve the collection name immediately
    /// after `CREATE COLLECTION … WITH (bitemporal=true)`.
    pub fn register_collection(&mut self, name: &str) {
        self.registered_collections.insert(name.to_owned());
    }

    /// List all known collection names.
    ///
    /// Merges names that own a Loro document (i.e. collections that have been
    /// written to) with names that were explicitly registered via
    /// `register_collection` (i.e. collections created via DDL but not yet
    /// populated).
    pub fn collection_names(&self) -> Vec<String> {
        let mut names: std::collections::BTreeSet<String> = self.states.keys().cloned().collect();
        names.extend(self.registered_collections.iter().cloned());
        names.into_iter().collect()
    }

    /// Set conflict resolution policy for a collection.
    pub fn set_policy(&mut self, collection: &str, policy: nodedb_crdt::CollectionPolicy) {
        self.policies.set(collection, policy);
    }

    /// Get the policy registry (for sync conflict resolution).
    pub fn policies(&self) -> &nodedb_crdt::PolicyRegistry {
        &self.policies
    }
}

#[cfg(test)]
mod tests {
    use super::CrdtEngine;
    use loro::LoroValue;

    #[test]
    fn read_after_upsert() {
        let mut engine = CrdtEngine::new(1).unwrap();
        engine
            .upsert("users", "u1", &[("age", LoroValue::I64(30))])
            .unwrap();

        assert!(engine.exists("users", "u1"));
        let val = engine.read("users", "u1").unwrap();
        // The value should be a map containing "age": 30.
        assert!(format!("{val:?}").contains("30"));
    }

    #[test]
    fn list_ids() {
        let mut engine = CrdtEngine::new(1).unwrap();
        engine
            .upsert("col", "a", &[("x", LoroValue::I64(1))])
            .unwrap();
        engine
            .upsert("col", "b", &[("x", LoroValue::I64(2))])
            .unwrap();

        let mut ids = engine.list_ids("col");
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
