// SPDX-License-Identifier: Apache-2.0

//! Turning reranked HNSW nodes into search results: resolve each node's
//! document id, attach its row fields, and apply the metadata post-filter.

use std::collections::HashMap;

use nodedb_types::filter::MetadataFilter;
use nodedb_types::result::SearchResult;
use nodedb_vector::rerank::Ranked;

use crate::engine::crdt::CrdtEngine;
use crate::engine::vector::IndexIdMap;

use super::filter::{fields_match, row_fields};

/// What hydration reads and how it shapes each result.
pub(super) struct Hydration<'a> {
    /// The searched index's node bindings.
    pub ids: Option<&'a IndexIdMap>,
    pub crdt: &'a CrdtEngine,
    pub collection: &'a str,
    pub filter: Option<&'a MetadataFilter>,
    /// Vector-internal fields left out of the metadata.
    pub exclude_fields: &'a [&'a str],
    /// Return no metadata. A post-filter still reads the row.
    pub skip_payload_fetch: bool,
}

impl Hydration<'_> {
    /// The results for `ranked`, in rank order, without rows the filter drops.
    pub(super) fn results(&self, ranked: Vec<Ranked>) -> Vec<SearchResult> {
        ranked.into_iter().filter_map(|r| self.result(r)).collect()
    }

    fn result(&self, r: Ranked) -> Option<SearchResult> {
        let doc_id = self
            .ids
            .and_then(|ids| ids.doc_id(r.id))
            .map_or_else(|| r.id.to_string(), str::to_owned);

        let needs_payload = !self.skip_payload_fetch || self.filter.is_some();
        let metadata = if needs_payload {
            row_fields(self.crdt, self.collection, &doc_id, self.exclude_fields).unwrap_or_default()
        } else {
            HashMap::new()
        };

        if let Some(f) = self.filter
            && !fields_match(&metadata, f)
        {
            return None;
        }
        // Honor skip_payload_fetch even when the filter forced a read: the
        // caller asked for no payload in the result.
        let metadata = if self.skip_payload_fetch {
            HashMap::new()
        } else {
            metadata
        };

        Some(SearchResult {
            id: doc_id,
            node_id: None,
            distance: r.distance,
            metadata,
        })
    }
}
