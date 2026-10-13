// SPDX-License-Identifier: Apache-2.0

//! Graph edges stored as `Namespace::Graph` rows, outside the CRDT engine.
//!
//! `graph_insert_edge` keeps each edge as a CRDT document in
//! `__edges__{collection}`, and every CRDT document stays in memory for the
//! life of the process. A caller that does not sync edges and keeps no edge
//! properties can store them here instead: one key per edge, the same rows
//! `GRAPH INSERT EDGE` writes. The CSR rebuild reads them at open, and
//! traversal reads the CSR, so both kinds of edge answer the same queries.

use nodedb_types::Namespace;
use nodedb_types::error::{NodeDbError, NodeDbResult};
use nodedb_types::id::NodeId;
use nodedb_types::value::Value;

use crate::engine::graph::edge::edge_crdt_collection;
use crate::engine::graph::edge_storage::{edge_store_key, edge_to_value};
use crate::engine::graph::history;
use crate::engine::graph::index::CsrIndex;
use crate::nodedb::LockExt;
use crate::nodedb::NodeDbLite;
use crate::query::graph_ops::edges::edge_delete;
use crate::runtime::now_millis_i64;
use crate::storage::engine::{StorageEngine, WriteOp};

/// Rows per storage batch when moving CRDT edges.
const MOVE_BATCH: usize = 10_000;

impl<S: StorageEngine> NodeDbLite<S> {
    /// Store `edges`, each `(src, label, dst)`, in `collection` as
    /// `Namespace::Graph` rows with no properties, in one storage batch.
    /// Storing an edge that exists is a no-op.
    pub async fn graph_put_stored_edges(
        &self,
        collection: &str,
        edges: &[(&NodeId, &str, &NodeId)],
    ) -> NodeDbResult<()> {
        if self.governor.worst_engine_pressure() == nodedb_mem::PressureLevel::Emergency {
            return Err(NodeDbError::storage(
                crate::error::LiteError::Backpressure {
                    detail: "graph edge insert rejected: memory governor is at Emergency pressure"
                        .into(),
                },
            ));
        }
        let rows: Vec<(&str, &str, &str)> = edges
            .iter()
            .map(|(src, label, dst)| (src.as_str(), *label, dst.as_str()))
            .collect();
        self.put_edge_rows(collection, &rows).await
    }

    /// Remove the stored edge `(src, label, dst)` from `collection`. Removing
    /// an edge that is not stored is a no-op.
    pub async fn graph_delete_stored_edge(
        &self,
        collection: &str,
        src: &NodeId,
        label: &str,
        dst: &NodeId,
    ) -> NodeDbResult<()> {
        edge_delete(
            &self.storage,
            &self.csr,
            collection,
            src.as_str(),
            label,
            dst.as_str(),
        )
        .await
        .map(|_| ())
        .map_err(NodeDbError::storage)
    }

    /// Move every CRDT edge of `collection` to stored rows, then clear its
    /// `__edges__` CRDT collection. Returns the number of edges moved.
    ///
    /// Safe to repeat after a crash: storing an edge again is a no-op, and the
    /// CRDT rows are cleared only after every edge is stored. CRDT edge
    /// properties are not carried over.
    pub async fn graph_move_crdt_edges_to_store(&self, collection: &str) -> NodeDbResult<usize> {
        let rows = self.graph_edge_rows(collection);
        for chunk in rows.chunks(MOVE_BATCH) {
            let chunk: Vec<(&str, &str, &str)> = chunk
                .iter()
                .map(|(src, dst, label)| (src.as_str(), label.as_str(), dst.as_str()))
                .collect();
            self.put_edge_rows(collection, &chunk).await?;
        }
        if !rows.is_empty() {
            self.crdt
                .lock_or_recover()
                .clear_collection(&edge_crdt_collection(collection))
                .map_err(NodeDbError::storage)?;
        }
        Ok(rows.len())
    }

    /// Graph collections that still hold CRDT edges.
    pub fn graph_crdt_edge_collections(&self) -> Vec<String> {
        let crdt = self.crdt.lock_or_recover();
        crdt.collection_names()
            .iter()
            .filter_map(|name| name.strip_prefix("__edges__"))
            .filter(|graph| !crdt.list_ids(&edge_crdt_collection(graph)).is_empty())
            .map(str::to_owned)
            .collect()
    }

    async fn put_edge_rows(
        &self,
        collection: &str,
        rows: &[(&str, &str, &str)],
    ) -> NodeDbResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let memory = self.memory_for(nodedb_mem::EngineId::Graph);
        {
            let mut map = self.csr.lock_or_recover();
            let csr = map.get_or_insert_with(collection, || CsrIndex::new(memory));
            for (src, label, dst) in rows {
                csr.add_edge(src, label, dst)
                    .map_err(|e| NodeDbError::storage(e.to_string()))?;
            }
        }
        let mut ops = Vec::with_capacity(rows.len());
        for (src, label, dst) in rows {
            ops.push(WriteOp::Put {
                ns: Namespace::Graph,
                key: edge_store_key(collection, src, label, dst),
                value: edge_to_value(collection, src, label, dst, &[])
                    .map_err(NodeDbError::storage)?,
            });
        }
        self.storage
            .batch_write(&ops)
            .await
            .map_err(NodeDbError::storage)?;
        if history::is_bitemporal(self.storage.as_ref(), collection)
            .await
            .unwrap_or(false)
        {
            let ts = now_millis_i64();
            for (src, label, dst) in rows {
                let _ = history::record_edge_insert(
                    self.storage.as_ref(),
                    collection,
                    &format!("{src}->{dst}:{label}"),
                    &Value::Null,
                    ts,
                )
                .await;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nodedb_client::NodeDb;
    use nodedb_types::filter::EdgeFilter;
    use nodedb_types::graph::Direction;
    use nodedb_types::id::NodeId;

    use crate::{NodeDbLite, PagedbStorageMem};

    fn node(id: &str) -> NodeId {
        NodeId::from_validated(id.to_string())
    }

    async fn neighbours(db: &NodeDbLite<PagedbStorageMem>, from: &str) -> Vec<String> {
        let mut ids: Vec<String> = db
            .graph_traverse(
                "g",
                &node(from),
                1,
                Direction::Out,
                Some(&EdgeFilter::default()),
            )
            .await
            .expect("traverse")
            .nodes
            .into_iter()
            .map(|n| n.id.as_str().to_string())
            .filter(|id| id != from)
            .collect();
        ids.sort();
        ids
    }

    #[tokio::test]
    async fn crdt_edges_move_to_stored_rows_and_still_traverse() {
        let storage = PagedbStorageMem::open_in_memory().await.expect("storage");
        let db = NodeDbLite::open(storage).await.expect("db");
        let (a, b, c) = (node("a"), node("b"), node("c"));
        db.graph_insert_edge("g", &a, &b, "MENTIONS", None)
            .await
            .expect("crdt edge");
        db.graph_insert_edge("g", &a, &c, "MENTIONS", None)
            .await
            .expect("crdt edge");
        assert_eq!(db.graph_crdt_edge_collections(), vec!["g".to_string()]);

        assert_eq!(
            db.graph_move_crdt_edges_to_store("g").await.expect("move"),
            2
        );
        assert!(db.graph_edge_rows("g").is_empty(), "CRDT edges cleared");
        assert!(db.graph_crdt_edge_collections().is_empty());
        assert_eq!(neighbours(&db, "a").await, vec!["b", "c"]);
        // A second move finds nothing to do.
        assert_eq!(
            db.graph_move_crdt_edges_to_store("g").await.expect("move"),
            0
        );

        let d = node("d");
        db.graph_put_stored_edges("g", &[(&a, "MENTIONS", &d), (&a, "MENTIONS", &d)])
            .await
            .expect("stored edge");
        assert_eq!(neighbours(&db, "a").await, vec!["b", "c", "d"]);

        db.graph_delete_stored_edge("g", &a, "MENTIONS", &b)
            .await
            .expect("delete");
        assert_eq!(neighbours(&db, "a").await, vec!["c", "d"]);
    }
}
