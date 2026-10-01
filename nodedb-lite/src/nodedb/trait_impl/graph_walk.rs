// SPDX-License-Identifier: Apache-2.0

//! Synchronous, filtered graph discovery with physical edge identities.

use std::collections::{HashMap, HashSet, VecDeque};

use nodedb_types::{
    error::{NodeDbError, NodeDbResult},
    filter::EdgeFilter,
    graph::Direction,
    id::{EdgeId, NodeId},
    result::SubGraphEdge,
};

use crate::{
    engine::{
        crdt::CrdtEngine,
        graph::{
            edge::edge_crdt_collection,
            index::CsrIndex,
            properties::{EdgeKey, MAX_PROPERTY_BYTES, Properties, PropertyMap},
            traversal::DEFAULT_MAX_VISITED,
        },
    },
    nodedb::convert::loro_value_to_document,
};

pub(super) struct WalkResult {
    pub nodes: Vec<(String, u8)>,
    pub edges: Vec<SubGraphEdge>,
    pub missing_properties: HashSet<EdgeKey>,
    pub parents: HashMap<String, String>,
}

type RawEdgeKey = (u32, u32, u32);

struct CachedEdge {
    edge: SubGraphEdge,
    matches: bool,
    missing_crdt: bool,
}

pub(super) struct WalkParams<'a> {
    pub collection: &'a str,
    pub start: &'a str,
    pub depth: u8,
    pub direction: Direction,
    pub filter: Option<&'a EdgeFilter>,
    pub destination: Option<&'a str>,
}

pub(super) fn walk(
    csr: &CsrIndex,
    crdt: &CrdtEngine,
    sql: &PropertyMap,
    params: WalkParams<'_>,
) -> NodeDbResult<WalkResult> {
    let mut result = WalkResult {
        nodes: Vec::new(),
        edges: Vec::new(),
        missing_properties: HashSet::new(),
        parents: HashMap::new(),
    };
    if !csr.contains_node(params.start) {
        return Ok(result);
    }
    let labels: HashSet<&str> = params
        .filter
        .map(|filter| filter.labels.iter().map(String::as_str).collect())
        .unwrap_or_default();
    let mut seen = HashSet::from([params.start.to_owned()]);
    let mut queue = VecDeque::from([(params.start.to_owned(), 0u8)]);
    let mut cache = HashMap::<RawEdgeKey, CachedEdge>::new();
    let mut budget = EdgeBudget::default();
    let mut candidate_edges = Vec::new();
    let mut edge_ids = HashSet::new();
    while let Some((node, depth)) = queue.pop_front() {
        result.nodes.push((node.clone(), depth));
        if params.destination == Some(node.as_str()) {
            break;
        }
        for (key, neighbor) in oriented_neighbors(
            csr,
            &node,
            &labels,
            &params,
            &cache,
            &mut budget,
            (depth == params.depth).then_some(&seen),
        )? {
            if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
                let (Some(src), Some(dst)) =
                    (csr.node_name_checked(key.0), csr.node_name_checked(key.2))
                else {
                    return Err(NodeDbError::storage(
                        "graph adjacency contains an invalid node: rebuild the graph",
                    ));
                };
                let identity = (
                    src.to_owned(),
                    csr.label_name(key.1).to_owned(),
                    dst.to_owned(),
                );
                let edge = materialize(crdt, sql, &params, &identity, &mut budget)?;
                entry.insert(edge);
            }
            let Some(edge) = cache.get(&key) else {
                continue;
            };
            if !edge.matches {
                continue;
            }
            if edge_ids.insert(key) {
                candidate_edges.push(key);
            }
            if depth < params.depth && !seen.contains(neighbor) {
                if seen.len() == DEFAULT_MAX_VISITED {
                    return Err(NodeDbError::program_limit_exceeded(format!(
                        "graph traversal exceeds {DEFAULT_MAX_VISITED} nodes in '{}': reduce depth",
                        params.collection
                    )));
                }
                seen.insert(neighbor.to_owned());
                result.parents.insert(neighbor.to_owned(), node.clone());
                queue.push_back((neighbor.to_owned(), depth + 1));
            }
        }
    }
    for key in candidate_edges {
        if let Some(edge) = cache.remove(&key)
            && seen.contains(edge.edge.from.as_str())
            && seen.contains(edge.edge.to.as_str())
        {
            if edge.missing_crdt {
                result.missing_properties.insert((
                    edge.edge.from.as_str().to_owned(),
                    edge.edge.label.clone(),
                    edge.edge.to.as_str().to_owned(),
                ));
            }
            result.edges.push(edge.edge);
        }
    }
    Ok(result)
}

fn oriented_neighbors<'a>(
    csr: &'a CsrIndex,
    node: &str,
    labels: &HashSet<&str>,
    params: &WalkParams<'_>,
    cache: &HashMap<RawEdgeKey, CachedEdge>,
    budget: &mut EdgeBudget,
    discovered: Option<&HashSet<String>>,
) -> NodeDbResult<Vec<(RawEdgeKey, &'a str)>> {
    let Some(node_id) = csr.node_id_raw(node) else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    let mut unique = HashSet::new();
    let out = csr
        .iter_out_edges_raw(node_id)
        .filter(|_| params.direction != Direction::In)
        .map(|edge| (Direction::Out, edge));
    let incoming = csr
        .iter_in_edges_raw(node_id)
        .filter(|_| params.direction != Direction::Out)
        .map(|edge| (Direction::In, edge));
    for (orientation, (label_id, neighbor_id)) in out.chain(incoming) {
        let label = csr.label_name(label_id);
        if !labels.is_empty() && !labels.contains(&label) {
            continue;
        }
        let Some(neighbor) = csr.node_name_checked(neighbor_id) else {
            return Err(NodeDbError::storage(
                "graph adjacency contains an invalid node: rebuild the graph",
            ));
        };
        if discovered.is_some_and(|nodes| !nodes.contains(neighbor)) {
            continue;
        }
        let key = if orientation == Direction::Out {
            (node_id, label_id, neighbor_id)
        } else {
            (neighbor_id, label_id, node_id)
        };
        if !unique.insert(key) {
            continue;
        }
        if !cache.contains_key(&key) {
            budget.retain(
                params.collection,
                node.len()
                    .saturating_add(label.len())
                    .saturating_add(neighbor.len()),
            )?;
        }
        result.push((key, neighbor));
    }
    // Borrowed names preserve deterministic ordering without cloning the adjacency list.
    result.sort_unstable_by(|(a, an), (b, bn)| {
        csr.node_name_checked(a.0)
            .cmp(&csr.node_name_checked(b.0))
            .then_with(|| csr.label_name(a.1).cmp(csr.label_name(b.1)))
            .then_with(|| an.cmp(bn))
    });
    Ok(result)
}

fn materialize(
    crdt: &CrdtEngine,
    sql: &PropertyMap,
    params: &WalkParams<'_>,
    key: &EdgeKey,
    budget: &mut EdgeBudget,
) -> NodeDbResult<CachedEdge> {
    let from = NodeId::try_new(key.0.clone()).map_err(NodeDbError::storage)?;
    let to = NodeId::try_new(key.2.clone()).map_err(NodeDbError::storage)?;
    let id =
        EdgeId::try_first(from.clone(), to.clone(), key.1.clone()).map_err(NodeDbError::storage)?;
    let row_id = id.to_string();
    let crdt_row = crdt.read(&edge_crdt_collection(params.collection), &row_id);
    let missing_crdt = crdt_row.is_none();
    let properties: Properties = match crdt_row {
        Some(row) => loro_value_to_document(&row_id, &row)
            .fields
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "src" | "dst" | "label"))
            .collect(),
        None => sql.get(key).cloned().unwrap_or_default(),
    };
    let bytes = zerompk::to_msgpack_vec(&properties)
        .map_err(NodeDbError::storage)?
        .len();
    budget.add_bytes(params.collection, bytes)?;
    let matches = match params.filter {
        Some(filter) if !filter.property_filters.is_empty() => {
            let document = serde_json::to_value(&properties).map_err(NodeDbError::storage)?;
            filter.property_filters.iter().all(|filter| {
                nodedb_query::metadata_filter::matches_metadata_filter(&document, filter)
            })
        }
        _ => true,
    };
    Ok(CachedEdge {
        edge: SubGraphEdge {
            id,
            from,
            to,
            label: key.1.clone(),
            properties,
        },
        matches,
        missing_crdt,
    })
}

struct EdgeBudget {
    max_records: usize,
    max_bytes: usize,
    records: usize,
    bytes: usize,
}

impl Default for EdgeBudget {
    fn default() -> Self {
        Self {
            records: 0,
            bytes: 0,
            max_records: DEFAULT_MAX_VISITED,
            max_bytes: MAX_PROPERTY_BYTES,
        }
    }
}

impl EdgeBudget {
    fn retain(&mut self, collection: &str, bytes: usize) -> NodeDbResult<()> {
        if self.records == self.max_records {
            return Err(NodeDbError::program_limit_exceeded(format!(
                "graph edge cache exceeds {} records in '{collection}': reduce depth",
                self.max_records
            )));
        }
        self.add_bytes(collection, bytes)?;
        self.records += 1;
        Ok(())
    }

    fn add_bytes(&mut self, collection: &str, bytes: usize) -> NodeDbResult<()> {
        let total_bytes = self.bytes.saturating_add(bytes);
        if total_bytes > self.max_bytes {
            return Err(NodeDbError::program_limit_exceeded(format!(
                "graph edge cache exceeds {} identity and property bytes in '{collection}': reduce depth",
                self.max_bytes
            )));
        }
        self.bytes = total_bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk_params(start: &str, direction: Direction) -> WalkParams<'_> {
        WalkParams {
            collection: "graph",
            start,
            depth: 0,
            direction,
            filter: None,
            destination: None,
        }
    }

    #[test]
    fn retained_edges_reject_count_and_byte_overflow() {
        let mut count = EdgeBudget {
            records: DEFAULT_MAX_VISITED - 1,
            ..EdgeBudget::default()
        };
        count.retain("graph", 0).unwrap();
        assert!(
            count
                .retain("graph", 0)
                .unwrap_err()
                .to_string()
                .contains("records")
        );
        let mut bytes = EdgeBudget::default();
        bytes.retain("graph", MAX_PROPERTY_BYTES).unwrap();
        assert!(
            bytes
                .retain("graph", 1)
                .unwrap_err()
                .to_string()
                .contains("identity and property bytes")
        );
        assert_eq!(bytes.records, 1);
        assert_eq!(bytes.bytes, MAX_PROPERTY_BYTES);
    }
    #[test]
    fn adjacency_collection_bounds_dense_buffered_and_mixed_edges() {
        for compact_count in [0, 2, 3] {
            let mut csr = CsrIndex::new(crate::query::graph_ops::test_memory());
            for i in 0..3 {
                if i == compact_count && compact_count != 0 {
                    csr.compact().unwrap();
                }
                csr.add_edge("root", "LINK", &format!("neighbor{i}"))
                    .unwrap();
            }
            if compact_count == 3 {
                csr.compact().unwrap();
            }
            let cache = HashMap::new();
            let mut budget = EdgeBudget {
                max_records: 2,
                ..EdgeBudget::default()
            };
            let error = oriented_neighbors(
                &csr,
                "root",
                &HashSet::new(),
                &walk_params("root", Direction::Out),
                &cache,
                &mut budget,
                None,
            )
            .unwrap_err();
            assert!(error.to_string().contains("records"));
            let mut bytes = EdgeBudget {
                max_bytes: 1,
                ..EdgeBudget::default()
            };
            assert!(
                oriented_neighbors(
                    &csr,
                    "root",
                    &HashSet::new(),
                    &walk_params("root", Direction::Out),
                    &cache,
                    &mut bytes,
                    None,
                )
                .unwrap_err()
                .to_string()
                .contains("identity and property bytes")
            );
        }
    }

    #[test]
    fn both_deduplicates_self_loops_before_charging_identity_budget() {
        let mut csr = CsrIndex::new(crate::query::graph_ops::test_memory());
        for (src, dst) in [("a", "b"), ("b", "a"), ("a", "a")] {
            csr.add_edge(src, "LINK", dst).unwrap();
        }
        let mut budget = EdgeBudget {
            max_records: 3,
            ..EdgeBudget::default()
        };
        let edges = oriented_neighbors(
            &csr,
            "a",
            &HashSet::new(),
            &walk_params("a", Direction::Both),
            &HashMap::new(),
            &mut budget,
            None,
        )
        .unwrap();
        assert_eq!(edges.len(), 3);
        assert_eq!(budget.records, 3);
        assert_eq!(budget.bytes, 18);
    }
    #[test]
    fn boundary_adjacency_skips_undiscovered_nodes_before_cache_charging() {
        let mut csr = CsrIndex::new(crate::query::graph_ops::test_memory());
        for (src, dst) in [("a", "a"), ("a", "outside"), ("incoming", "a"), ("a", "b")] {
            csr.add_edge(src, "LINK", dst).unwrap();
        }
        let cache = HashMap::new();
        let discovered = HashSet::from(["a".to_owned()]);
        let mut zero = EdgeBudget {
            max_records: 1,
            max_bytes: 6,
            ..EdgeBudget::default()
        };
        let edges = oriented_neighbors(
            &csr,
            "a",
            &HashSet::new(),
            &walk_params("a", Direction::Both),
            &cache,
            &mut zero,
            Some(&discovered),
        )
        .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].1, "a");
        assert_eq!(zero.records, 1);
        assert_eq!(zero.bytes, 6);
        let discovered = HashSet::from(["a".to_owned(), "b".to_owned()]);
        let mut boundary = EdgeBudget {
            max_records: 2,
            max_bytes: 12,
            ..EdgeBudget::default()
        };
        let edges = oriented_neighbors(
            &csr,
            "a",
            &HashSet::new(),
            &walk_params("a", Direction::Both),
            &cache,
            &mut boundary,
            Some(&discovered),
        )
        .unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().any(|(_, neighbor)| *neighbor == "b"));
        assert_eq!(boundary.records, 2);
        assert_eq!(boundary.bytes, 12);
    }
}
