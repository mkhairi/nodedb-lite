// SPDX-License-Identifier: BUSL-1.1

//! Run-compressed index of queued deltas that live only under their `delta:`
//! keys on disk.
//!
//! The pending-delta queue is drained only by an Origin acknowledgement, so a
//! replica with no Origin accumulates every mutation it has ever made. Holding
//! each one's payload in memory costs hundreds of bytes per entry; holding only
//! the fact that it exists costs, for the append-only shape the queue actually
//! has, a single run.
//!
//! Mutation ids are handed out consecutively, so the evicted set is nearly
//! always one contiguous run. Removing an id from the middle (an Origin ack for
//! an entry that was never paged back in) splits a run, which is why this is a
//! run list rather than a single interval — the number of runs is bounded by
//! the number of such acks, not by the size of the queue.

use std::cmp::Ordering;

/// The mutation ids of queue entries that are on disk but not in memory.
#[derive(Debug, Default, Clone)]
pub struct SpillIndex {
    /// Inclusive `(start, end)` runs, ascending and disjoint. Adjacent runs are
    /// merged, so `runs[i].1 + 1 < runs[i + 1].0` always holds.
    runs: Vec<(u64, u64)>,
    /// Number of ids covered by `runs` — the sum of their lengths, maintained
    /// incrementally so the count is O(1) rather than O(runs).
    count: usize,
}

impl SpillIndex {
    /// Number of queue entries held only on disk.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether every queue entry is resident.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Forget every spilled id without touching storage.
    pub fn clear(&mut self) {
        self.runs.clear();
        self.count = 0;
    }

    /// Whether `mutation_id` names an entry that is queued but not resident.
    pub fn contains(&self, mutation_id: u64) -> bool {
        self.run_holding(mutation_id).is_some()
    }

    /// The highest spilled mutation id, if any.
    pub fn max(&self) -> Option<u64> {
        self.runs.last().map(|&(_, end)| end)
    }

    /// Index of the run covering `mutation_id`.
    fn run_holding(&self, mutation_id: u64) -> Option<usize> {
        self.runs
            .binary_search_by(|&(start, end)| {
                if mutation_id < start {
                    Ordering::Greater
                } else if mutation_id > end {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .ok()
    }

    /// Record that `mutation_id`'s entry now lives only on disk.
    ///
    /// Returns `false` when it was already recorded, so the count never
    /// double-counts an id.
    pub fn insert(&mut self, mutation_id: u64) -> bool {
        // Position of the first run starting after `mutation_id`.
        let at = self
            .runs
            .partition_point(|&(start, _)| start <= mutation_id);

        if at > 0 {
            let (_, end) = self.runs[at - 1];
            if mutation_id <= end {
                return false;
            }
        }

        let joins_left = at > 0 && self.runs[at - 1].1 + 1 == mutation_id;
        let joins_right = at < self.runs.len() && mutation_id + 1 == self.runs[at].0;

        match (joins_left, joins_right) {
            (true, true) => {
                self.runs[at - 1].1 = self.runs[at].1;
                self.runs.remove(at);
            }
            (true, false) => self.runs[at - 1].1 = mutation_id,
            (false, true) => self.runs[at].0 = mutation_id,
            (false, false) => self.runs.insert(at, (mutation_id, mutation_id)),
        }
        self.count += 1;
        true
    }

    /// Drop `mutation_id` from the index — the entry has been acknowledged,
    /// rejected, or paged back into memory.
    ///
    /// Returns `false` when it was not spilled.
    pub fn remove(&mut self, mutation_id: u64) -> bool {
        let Some(at) = self.run_holding(mutation_id) else {
            return false;
        };
        let (start, end) = self.runs[at];
        match (mutation_id == start, mutation_id == end) {
            (true, true) => {
                self.runs.remove(at);
            }
            (true, false) => self.runs[at].0 = start + 1,
            (false, true) => self.runs[at].1 = end - 1,
            (false, false) => {
                self.runs[at].1 = mutation_id - 1;
                self.runs.insert(at + 1, (mutation_id + 1, end));
            }
        }
        self.count -= 1;
        true
    }

    /// The `limit` lowest spilled ids, ascending — the next entries to page
    /// back in, oldest first, so the queue is replayed in the order it was
    /// written.
    pub fn lowest(&self, limit: usize) -> Vec<u64> {
        let mut ids = Vec::with_capacity(limit.min(self.count));
        for &(start, end) in &self.runs {
            for id in start..=end {
                if ids.len() == limit {
                    return ids;
                }
                ids.push(id);
            }
        }
        ids
    }

    /// The inclusive runs, for assertions.
    #[cfg(test)]
    pub(crate) fn runs(&self) -> &[(u64, u64)] {
        &self.runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consecutive_inserts_collapse_to_one_run() {
        let mut index = SpillIndex::default();
        for id in 1..=1_000 {
            assert!(index.insert(id));
        }

        assert_eq!(index.len(), 1_000);
        assert_eq!(
            index.runs(),
            [(1, 1_000)],
            "consecutive mutation ids must cost a single run, not one entry each"
        );
    }

    #[test]
    fn insert_is_idempotent() {
        let mut index = SpillIndex::default();
        assert!(index.insert(7));
        assert!(!index.insert(7), "a re-insert must not be counted twice");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn out_of_order_inserts_merge_on_both_sides() {
        let mut index = SpillIndex::default();
        index.insert(1);
        index.insert(3);
        assert_eq!(index.runs(), [(1, 1), (3, 3)]);

        index.insert(2);
        assert_eq!(index.runs(), [(1, 3)], "2 must join the runs either side");
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn removing_from_the_middle_splits_the_run() {
        let mut index = SpillIndex::default();
        for id in 1..=5 {
            index.insert(id);
        }

        assert!(index.remove(3));
        assert_eq!(index.runs(), [(1, 2), (4, 5)]);
        assert_eq!(index.len(), 4);
        assert!(!index.contains(3));
        for id in [1, 2, 4, 5] {
            assert!(index.contains(id), "{id} must still be spilled");
        }
    }

    #[test]
    fn removing_run_edges_shrinks_rather_than_splits() {
        let mut index = SpillIndex::default();
        for id in 1..=3 {
            index.insert(id);
        }

        assert!(index.remove(1));
        assert_eq!(index.runs(), [(2, 3)]);
        assert!(index.remove(3));
        assert_eq!(index.runs(), [(2, 2)]);
        assert!(index.remove(2));
        assert!(index.is_empty());
        assert_eq!(index.runs(), []);
    }

    #[test]
    fn removing_an_absent_id_is_reported() {
        let mut index = SpillIndex::default();
        index.insert(10);
        assert!(!index.remove(11));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn lowest_returns_the_oldest_ids_in_order() {
        let mut index = SpillIndex::default();
        for id in [10, 11, 12, 20, 21] {
            index.insert(id);
        }

        assert_eq!(index.lowest(4), vec![10, 11, 12, 20]);
        assert_eq!(index.lowest(usize::MAX), vec![10, 11, 12, 20, 21]);
        assert_eq!(index.lowest(0), Vec::<u64>::new());
    }

    #[test]
    fn max_reports_the_highest_spilled_id() {
        let mut index = SpillIndex::default();
        assert_eq!(index.max(), None);
        index.insert(5);
        index.insert(9);
        assert_eq!(index.max(), Some(9));
    }
}
