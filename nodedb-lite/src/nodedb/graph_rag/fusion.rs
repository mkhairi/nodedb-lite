// SPDX-License-Identifier: Apache-2.0

use nodedb_types::result::SearchResult;

/// Build a ranked list from `SearchResult`s for the shared RRF module.
pub(super) fn search_results_to_ranked(
    results: &[SearchResult],
    source: &'static str,
) -> Vec<nodedb_query::fusion::RankedResult> {
    results
        .iter()
        .enumerate()
        .map(|(rank, r)| nodedb_query::fusion::RankedResult {
            document_id: r.id.clone(),
            rank,
            score: r.distance,
            source,
        })
        .collect()
}
