// SPDX-License-Identifier: Apache-2.0

//! Metadata-filter evaluation against the CRDT row a vector is attached to.

use std::collections::HashMap;

use nodedb_types::filter::MetadataFilter;
use nodedb_types::value::Value;

use crate::engine::crdt::CrdtEngine;
use crate::engine::vector::IndexIdMap;
use crate::nodedb::convert::loro_value_to_document;

/// The user-visible fields of `doc_id`'s row in `collection`: every field
/// except the vector-internal ones named in `exclude_fields`. `None` when the
/// row does not exist.
pub(super) fn row_fields(
    crdt: &CrdtEngine,
    collection: &str,
    doc_id: &str,
    exclude_fields: &[&str],
) -> Option<HashMap<String, Value>> {
    let loro_val = crdt.read(collection, doc_id)?;
    let doc = loro_value_to_document(doc_id, &loro_val);
    Some(
        doc.fields
            .into_iter()
            .filter(|(k, _)| !exclude_fields.contains(&k.as_str()))
            .collect(),
    )
}

/// Whether `fields` satisfy `filter`.
pub(super) fn fields_match(fields: &HashMap<String, Value>, filter: &MetadataFilter) -> bool {
    let json_doc = serde_json::to_value(fields).unwrap_or_default();
    nodedb_query::metadata_filter::matches_metadata_filter(&json_doc, filter)
}

/// The nodes of one index whose rows satisfy `filter`. Vector-internal
/// fields never take part in the match.
pub(super) fn allowed_nodes(
    ids: Option<&IndexIdMap>,
    crdt: &CrdtEngine,
    collection: &str,
    filter: &MetadataFilter,
    exclude_fields: &[&str],
) -> roaring::RoaringBitmap {
    let mut allowed = roaring::RoaringBitmap::new();
    let Some(ids) = ids else {
        return allowed;
    };
    for (doc_id, node) in ids.iter() {
        if let Some(fields) = row_fields(crdt, collection, doc_id, exclude_fields)
            && fields_match(&fields, filter)
        {
            allowed.insert(node);
        }
    }
    allowed
}
