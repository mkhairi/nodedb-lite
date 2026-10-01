// SPDX-License-Identifier: Apache-2.0

//! Graph engine helpers for `NodeDbLite`.

use std::collections::{HashMap, HashSet};

use nodedb_types::document::Document;
use nodedb_types::error::{NodeDbError, NodeDbResult};
use nodedb_types::graph::GraphStats;
use nodedb_types::id::{EdgeId, NodeId};
use nodedb_types::value::Value;

use nodedb_graph::params::{AlgoParams, GraphAlgorithm};

use crate::engine::graph::edge::{
    edge_crdt_collection, edge_crdt_fields, edge_history_value, edge_id_for,
};
use crate::engine::graph::history;
use crate::engine::graph::index::CsrIndex;
use crate::nodedb::LockExt;
use crate::nodedb::NodeDbLite;
use crate::nodedb::convert::loro_value_to_document;
use crate::query::graph_ops::algorithms;
use crate::runtime::now_millis_i64;
use crate::storage::engine::StorageEngine;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Insert an edge into the collection-scoped CSR adjacency index and persist
    /// a corresponding CRDT document holding `src`, `dst`, `label`, and any
    /// user-supplied properties. Edges are stored under a per-collection CRDT
    /// namespace so that collections are fully isolated from one another.
    pub(super) async fn graph_insert_edge_impl(
        &self,
        collection: &str,
        from: &NodeId,
        to: &NodeId,
        edge_type: &str,
        properties: Option<Document>,
    ) -> NodeDbResult<EdgeId> {
        if self.governor.worst_engine_pressure() == nodedb_mem::PressureLevel::Emergency {
            return Err(NodeDbError::storage(
                crate::error::LiteError::Backpressure {
                    detail: "graph edge insert rejected: memory governor is at Emergency pressure"
                        .into(),
                },
            ));
        }

        {
            let memory = self.memory_for(nodedb_mem::EngineId::Graph);
            let mut csr_map = self.csr.lock_or_recover();
            let csr = csr_map
                .entry(collection.to_string())
                .or_insert_with(|| CsrIndex::new(memory));
            let _ = csr.add_edge(from.as_str(), edge_type, to.as_str());
        }

        let edge_id = edge_id_for(from, to, edge_type)?;
        let edge_key = format!("{edge_id}");
        let edge_coll = edge_crdt_collection(collection);

        {
            let mut crdt = self.crdt.lock_or_recover();
            let fields = edge_crdt_fields(from, to, edge_type, &properties);
            crdt.upsert(&edge_coll, &edge_key, &fields)
                .map_err(NodeDbError::storage)?;
        }

        // Record edge birth in the bitemporal history table if the collection
        // has bitemporal tracking enabled.
        let bitemporal = history::is_bitemporal(self.storage.as_ref(), collection)
            .await
            .unwrap_or(false);
        if bitemporal {
            let system_from_ms = now_millis_i64();
            let props_value = edge_history_value(from, to, edge_type, &properties);
            let _ = history::record_edge_insert(
                self.storage.as_ref(),
                collection,
                &edge_key,
                &props_value,
                system_from_ms,
            )
            .await;
        }

        self.update_memory_stats();
        Ok(edge_id)
    }

    /// Remove an edge from both the collection-scoped CSR index and the
    /// collection-scoped CRDT edge document store.
    pub(super) async fn graph_delete_edge_impl(
        &self,
        collection: &str,
        edge_id: &EdgeId,
    ) -> NodeDbResult<()> {
        let src = edge_id.src.as_str();
        let dst = edge_id.dst.as_str();
        let label = &edge_id.label;
        {
            let mut csr_map = self.csr.lock_or_recover();
            if let Some(csr) = csr_map.get_mut(collection) {
                csr.remove_edge(src, label, dst);
            }
        }

        let edge_key = format!("{edge_id}");
        let edge_coll = edge_crdt_collection(collection);
        {
            let mut crdt = self.crdt.lock_or_recover();
            crdt.delete(&edge_coll, &edge_key)
                .map_err(NodeDbError::storage)?;
        }

        // Finalize the history entry if the collection is bitemporal.
        let bitemporal = history::is_bitemporal(self.storage.as_ref(), collection)
            .await
            .unwrap_or(false);
        if bitemporal {
            let system_to_ms = now_millis_i64();
            let _ = history::record_edge_delete(
                self.storage.as_ref(),
                collection,
                &edge_key,
                system_to_ms,
            )
            .await;
        }

        Ok(())
    }

    /// Return edge statistics for `collection`. When `collection` is `Some(name)`,
    /// counts reflect only the edges in that collection. When `collection` is `None`,
    /// all known graph collections are aggregated and a single combined entry is
    /// returned under the key `"*"`.
    ///
    /// `as_of` is not supported on Lite: the backend has no bitemporal store.
    /// Passing `Some(_)` returns an error.
    pub(super) async fn graph_stats_impl(
        &self,
        collection: Option<&str>,
        as_of: Option<i64>,
    ) -> NodeDbResult<Vec<GraphStats>> {
        if as_of.is_some() {
            return Err(NodeDbError::storage(
                "AS OF SYSTEM TIME is not supported on the Lite backend",
            ));
        }

        let crdt = self.crdt.lock_or_recover();

        // Determine which CRDT collections to aggregate.
        let edge_colls: Vec<String> = match collection {
            Some(name) => vec![edge_crdt_collection(name)],
            None => crdt
                .collection_names()
                .into_iter()
                .filter(|c| c.starts_with("__edges__"))
                .collect(),
        };

        let mut node_ids: HashSet<String> = HashSet::new();
        let mut label_counts: HashMap<String, u64> = HashMap::new();
        let mut total_edges: u64 = 0;

        for ec in &edge_colls {
            let edge_ids = crdt.list_ids(ec);
            total_edges += edge_ids.len() as u64;
            for key in &edge_ids {
                if let Some(loro_val) = crdt.read(ec, key) {
                    let doc = loro_value_to_document(key, &loro_val);
                    if let Some(label) = doc.get_str("label") {
                        *label_counts.entry(label.to_string()).or_insert(0) += 1;
                    }
                    if let Some(src) = doc.get_str("src") {
                        node_ids.insert(src.to_string());
                    }
                    if let Some(dst) = doc.get_str("dst") {
                        node_ids.insert(dst.to_string());
                    }
                }
            }
        }

        let mut labels: Vec<(String, u64)> = label_counts.into_iter().collect();
        labels.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        let coll_name = collection.unwrap_or("*").to_string();
        Ok(vec![GraphStats {
            collection: coll_name,
            node_count: node_ids.len() as u64,
            edge_count: total_edges,
            distinct_label_count: labels.len() as u64,
            labels,
        }])
    }

    /// Run PageRank (or Personalized PageRank) on the collection's CSR graph.
    ///
    /// Returns an empty `Vec` when the collection has no edges rather than an
    /// error — an empty graph simply has no ranks to report.
    pub(super) async fn graph_pagerank_impl(
        &self,
        collection: &str,
        personalization: Option<std::collections::HashMap<String, f64>>,
        damping: Option<f64>,
        max_iterations: Option<u32>,
    ) -> NodeDbResult<Vec<(String, f64)>> {
        // Fast path: if the collection isn't in the CSR map it has no edges.
        {
            let csr_map = self.csr.lock_or_recover();
            if !csr_map.contains_key(collection) {
                return Ok(Vec::new());
            }
        }

        let params = AlgoParams {
            collection: collection.to_string(),
            damping,
            max_iterations: max_iterations.map(|v| v as usize),
            personalization_vector: personalization,
            ..Default::default()
        };

        let result = algorithms::run_algo(&self.csr, GraphAlgorithm::PageRank, &params)
            .map_err(|e| NodeDbError::storage(format!("graph_pagerank: {e}")))?;

        // `result.columns` == ["node_id", "rank"]; extract and sort descending.
        let mut pairs: Vec<(String, f64)> = result
            .rows
            .into_iter()
            .filter_map(|mut row| {
                if row.len() < 2 {
                    return None;
                }
                let rank = match row.pop() {
                    Some(Value::Float(f)) => f,
                    _ => return None,
                };
                let node_id = match row.pop() {
                    Some(Value::String(s)) => s,
                    _ => return None,
                };
                Some((node_id, rank))
            })
            .collect();

        pairs.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(pairs)
    }
}
