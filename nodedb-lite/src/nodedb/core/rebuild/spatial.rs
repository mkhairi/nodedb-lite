// SPDX-License-Identifier: Apache-2.0

//! Spatial cold recovery from applied CRDT fields.

use crate::{
    nodedb::{core::types::NodeDbLite, lock_ext::LockExt},
    storage::engine::StorageEngine,
};

impl<S: StorageEngine> NodeDbLite<S> {
    /// Rebuild spatial indices from CRDT state (cold start fallback).
    ///
    /// Scans all collections for geometry-valued fields and indexes them.
    /// Called when checkpoint restore produces empty spatial indices.
    pub(crate) fn rebuild_spatial_indices(&self) {
        let crdt = self.crdt.lock_or_recover();
        let collections = crdt.collection_names();
        let mut spatial = self.spatial.lock_or_recover();

        for collection in &collections {
            if collection.starts_with("__") {
                continue;
            }
            let ids = crdt.list_ids(collection);
            for id in &ids {
                if let Some(loro_val) = crdt.read(collection, id) {
                    let doc = crate::nodedb::convert::loro_value_to_document(id, &loro_val);
                    for (field, value) in &doc.fields {
                        // Geometry fields are stored as GeoJSON strings.
                        if let nodedb_types::Value::String(s) = value
                            && let Ok(geom) =
                                sonic_rs::from_str::<nodedb_types::geometry::Geometry>(s)
                        {
                            spatial.index_document(collection, field, id, &geom);
                        }
                    }
                }
            }
        }
    }
}
