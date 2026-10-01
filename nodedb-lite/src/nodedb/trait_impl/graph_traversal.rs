// SPDX-License-Identifier: Apache-2.0

//! Directed graph traversal and property-aware shortest paths.

use nodedb_types::{
    error::NodeDbResult,
    filter::EdgeFilter,
    graph::Direction,
    id::NodeId,
    result::{SubGraph, SubGraphNode},
};

use crate::{
    engine::graph::{
        properties::{PropertyMap, load_sql_properties},
        traversal::DEFAULT_MAX_VISITED,
    },
    nodedb::{LockExt, NodeDbLite, convert::loro_value_to_document},
    storage::engine::StorageEngine,
};

use super::graph_walk::{WalkParams, walk};

impl<S: StorageEngine> NodeDbLite<S> {
    /// Traverse matching edges in `direction`, preserving their stored orientation.
    ///
    /// SQL property reads reject more than 100,000 records or 64 MiB serialized bytes.
    /// Edge caches reject more than 100,000 physical edges or 64 MiB identity and property bytes.
    pub(super) async fn graph_traverse_impl(
        &self,
        collection: &str,
        start: &NodeId,
        depth: u8,
        direction: Direction,
        edge_filter: Option<&EdgeFilter>,
    ) -> NodeDbResult<SubGraph> {
        let zero_without_edges = {
            let csr_map = self.csr.lock_or_recover();
            let Some(csr) = csr_map.get(collection) else {
                return Ok(SubGraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                });
            };
            let Some(start_id) = csr.node_id_raw(start.as_str()) else {
                return Ok(SubGraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                });
            };
            depth == 0
                && !csr.iter_out_edges_raw(start_id).any(|(label, dst)| {
                    dst == start_id
                        && edge_filter.is_none_or(|filter| {
                            filter.labels.is_empty()
                                || filter
                                    .labels
                                    .iter()
                                    .any(|name| name == csr.label_name(label))
                        })
                })
        };
        if zero_without_edges {
            let crdt = self.crdt.lock_or_recover();
            let properties = crdt
                .read("__nodes", start.as_str())
                .map(|row| loro_value_to_document(start.as_str(), &row).fields)
                .unwrap_or_default();
            return Ok(SubGraph {
                nodes: vec![SubGraphNode {
                    id: start.clone(),
                    depth: 0,
                    properties,
                }],
                edges: Vec::new(),
            });
        }
        let filtered = has_properties(edge_filter);
        let sql = if filtered {
            load_sql_properties(self.storage.as_ref(), collection).await?
        } else {
            PropertyMap::new()
        };
        // CRDT precedes CSR. Neither guard survives an async storage operation.
        let (mut result, nodes) = {
            let crdt = self.crdt.lock_or_recover();
            let csr_map = self.csr.lock_or_recover();
            let Some(csr) = csr_map.get(collection) else {
                return Ok(SubGraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                });
            };
            let result = walk(
                csr,
                &crdt,
                &sql,
                WalkParams {
                    collection,
                    start: start.as_str(),
                    depth,
                    direction,
                    filter: edge_filter,
                    destination: None,
                },
            )?;
            let nodes = result
                .nodes
                .iter()
                .map(|(name, depth)| SubGraphNode {
                    id: NodeId::from_validated(name.clone()),
                    depth: *depth,
                    properties: crdt
                        .read("__nodes", name)
                        .map(|row| loro_value_to_document(name, &row).fields)
                        .unwrap_or_default(),
                })
                .collect();
            (result, nodes)
        };
        if !filtered && !result.missing_properties.is_empty() {
            let sql = load_sql_properties(self.storage.as_ref(), collection).await?;
            for edge in &mut result.edges {
                let key = (
                    edge.from.as_str().to_owned(),
                    edge.label.clone(),
                    edge.to.as_str().to_owned(),
                );
                if result.missing_properties.contains(&key)
                    && let Some(properties) = sql.get(&key)
                {
                    edge.properties.clone_from(properties);
                }
            }
        }
        Ok(SubGraph {
            nodes,
            edges: result.edges,
        })
    }

    /// Find an outgoing path with every requested label and property predicate.
    pub(super) async fn graph_shortest_path_impl(
        &self,
        collection: &str,
        from: &NodeId,
        to: &NodeId,
        max_depth: u8,
        edge_filter: Option<&EdgeFilter>,
    ) -> NodeDbResult<Option<Vec<NodeId>>> {
        if !has_properties(edge_filter) {
            let labels: Vec<&str> = edge_filter
                .map(|filter| filter.labels.iter().map(String::as_str).collect())
                .unwrap_or_default();
            let csr_map = self.csr.lock_or_recover();
            let path = csr_map.get(collection).and_then(|csr| {
                csr.shortest_path(
                    nodedb_graph::ShortestPathParams {
                        src: from.as_str(),
                        dst: to.as_str(),
                        label_filter: &labels,
                        max_depth: max_depth as usize,
                        max_visited: DEFAULT_MAX_VISITED,
                        frontier_bitmap: None,
                    },
                    None,
                )
            });
            return Ok(path.map(|path| path.into_iter().map(NodeId::from_validated).collect()));
        }
        {
            let csr_map = self.csr.lock_or_recover();
            let Some(csr) = csr_map.get(collection) else {
                return Ok(None);
            };
            if !csr.contains_node(from.as_str()) || !csr.contains_node(to.as_str()) {
                return Ok(None);
            }
            if from == to {
                return Ok(Some(vec![from.clone()]));
            }
            if max_depth == 0 {
                return Ok(None);
            }
        }
        let sql = load_sql_properties(self.storage.as_ref(), collection).await?;
        let crdt = self.crdt.lock_or_recover();
        let csr_map = self.csr.lock_or_recover();
        let Some(csr) = csr_map.get(collection) else {
            return Ok(None);
        };
        let result = walk(
            csr,
            &crdt,
            &sql,
            WalkParams {
                collection,
                start: from.as_str(),
                depth: max_depth,
                direction: Direction::Out,
                filter: edge_filter,
                destination: Some(to.as_str()),
            },
        )?;
        if !result.nodes.iter().any(|(node, _)| node == to.as_str()) {
            return Ok(None);
        }
        let mut path = vec![to.as_str().to_owned()];
        let mut current = to.as_str();
        while current != from.as_str() {
            let Some(parent) = result.parents.get(current) else {
                return Ok(None);
            };
            path.push(parent.clone());
            current = parent;
        }
        path.reverse();
        Ok(Some(path.into_iter().map(NodeId::from_validated).collect()))
    }
}

fn has_properties(filter: Option<&EdgeFilter>) -> bool {
    filter.is_some_and(|filter| !filter.property_filters.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PagedbStorageMem;
    use nodedb_client::NodeDb;
    use nodedb_types::{Document, Value, filter::MetadataFilter};
    use std::sync::Arc;

    fn node(id: &str) -> NodeId {
        NodeId::try_new(id).unwrap()
    }
    fn props(score: i64) -> Document {
        let mut doc = Document::new("properties");
        doc.set("score", Value::Integer(score));
        doc.set("active", Value::Bool(true));
        doc
    }
    fn score_filter() -> EdgeFilter {
        EdgeFilter {
            labels: vec!["LINK".into(), "NEXT".into()],
            property_filters: vec![
                MetadataFilter::Gt {
                    field: "score".into(),
                    value: Value::Float(5.0),
                },
                MetadataFilter::and(vec![
                    MetadataFilter::eq("active", true),
                    MetadataFilter::Not(Box::new(MetadataFilter::eq("missing", "present"))),
                ]),
            ],
        }
    }

    async fn open_stored_graph()
    -> Result<Arc<NodeDbLite<PagedbStorageMem>>, Box<dyn std::error::Error>> {
        use crate::{
            engine::graph::CsrIndex,
            query::graph_ops::edges::{EdgePutArgs, edge_put},
            storage::{checksum, engine::StorageEngine},
        };
        use nodedb_types::Namespace;
        use std::collections::HashMap;
        use std::sync::Mutex;

        let storage = Arc::new(PagedbStorageMem::open_in_memory().await?);
        let memory = crate::query::graph_ops::test_memory();
        let csr = Arc::new(Mutex::new(HashMap::<String, CsrIndex>::new()));
        for dst in ["b", "empty"] {
            let mut fields = props(9).fields;
            fields.insert("sql_only".into(), Value::Bool(true));
            let properties = zerompk::to_msgpack_vec(&Value::Object(fields))?;
            edge_put(
                &storage,
                &csr,
                &memory,
                EdgePutArgs {
                    collection: "graph",
                    src_id: "a",
                    label: "LINK",
                    dst_id: dst,
                    properties: &properties,
                },
            )
            .await?;
        }
        let checkpoint = {
            let graph = csr
                .lock()
                .map_err(|_| std::io::Error::other("graph fixture lock is poisoned"))?;
            let index = graph
                .get("graph")
                .ok_or_else(|| std::io::Error::other("graph fixture has no CSR index"))?;
            index.checkpoint_to_bytes()?
        };
        storage
            .put(Namespace::Graph, b"csr:graph", &checksum::wrap(&checkpoint))
            .await?;
        let collections = zerompk::to_msgpack_vec(&vec!["graph".to_owned()])?;
        storage
            .put(Namespace::Meta, b"meta:csr_collections", &collections)
            .await?;
        let storage = Arc::try_unwrap(storage)
            .map_err(|_| std::io::Error::other("graph fixture retains storage owners"))?;
        Ok(NodeDbLite::open(storage).await?)
    }

    #[tokio::test]
    async fn stored_graph_properties_and_crdt_precedence_keep_complete_rows() {
        let db = open_stored_graph().await.unwrap();
        db.graph_insert_edge("graph", &node("a"), &node("empty"), "LINK", None)
            .await
            .unwrap();
        let filter = score_filter();
        let sql = db
            .graph_traverse("graph", &node("a"), 1, Direction::Out, Some(&filter))
            .await
            .unwrap();
        assert_eq!(sql.edges.len(), 1);
        assert_eq!(
            sql.edges[0].properties.get("sql_only"),
            Some(&Value::Bool(true))
        );
        let unfiltered = db
            .graph_traverse("graph", &node("a"), 1, Direction::Out, None)
            .await
            .unwrap();
        assert_eq!(unfiltered.edges[0].properties, sql.edges[0].properties);
        db.graph_insert_edge("graph", &node("a"), &node("b"), "LINK", Some(props(10)))
            .await
            .unwrap();
        let crdt = db
            .graph_traverse("graph", &node("a"), 1, Direction::Out, Some(&filter))
            .await
            .unwrap();
        assert_eq!(
            crdt.edges[0].properties.get("score"),
            Some(&Value::Integer(10))
        );
        assert!(!crdt.edges[0].properties.contains_key("sql_only"));
        let empty = db
            .graph_traverse("graph", &node("a"), 1, Direction::Out, None)
            .await
            .unwrap();
        let empty_edge = empty
            .edges
            .iter()
            .find(|edge| edge.to.as_str() == "empty")
            .unwrap();
        assert!(empty_edge.properties.is_empty());
        let filtered = db
            .graph_traverse("graph", &node("a"), 1, Direction::Out, Some(&filter))
            .await
            .unwrap();
        assert_eq!(filtered.edges.len(), 1);
        assert_eq!(filtered.edges[0].to.as_str(), "b");
    }
}
