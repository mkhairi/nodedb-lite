// SPDX-License-Identifier: Apache-2.0

//! Dirty tracking for the artifacts `flush` derives from in-memory state.
//!
//! `flush` runs on a timer. Rewriting a derived artifact on every tick costs
//! its full size whether or not anything changed, so an idle store with a
//! large vector index rewrote hundreds of megabytes per tick.
//!
//! Each tracked artifact, keyed by `(FlushArtifact, collection)`, carries two
//! numbers: `current`, bumped on every mutation, and `flushed`, the generation
//! last made durable. The artifact is dirty while they differ.
//!
//! A flush captures `current` under the same lock it serializes the artifact
//! under (`FlushGens::plan`) and records exactly that value once the
//! artifact's last durable write succeeds (`FlushGens::mark_flushed`). A
//! mutation that lands between the two bumps `current` past the captured value,
//! so the artifact stays dirty and the next flush writes it again. A failed
//! write never reaches `mark_flushed`, so the next flush retries it.
//!
//! The lock wrappers below make the bump structural. [`TrackedMap`] and
//! [`TrackedCell`] hand out guards whose mutable access bumps the generation,
//! so a new mutation site cannot forget to mark the artifact dirty.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, LockResult, Mutex, MutexGuard, PoisonError};

use super::lock_ext::LockExt;

/// A derived artifact whose flush is dirty-tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlushArtifact {
    /// A collection's HNSW checkpoint blob, `hnsw:<collection>`. Graph-only
    /// when the storage has vector segments, a full checkpoint otherwise.
    HnswGraph,
    /// The store-wide `hnsw_id_map` blob. Tracked under [`ID_MAP_KEY`].
    HnswIdMap,
    /// A collection's pagedb vector segment. It is built from the durable
    /// vector rows, so it is dirty whenever those rows change.
    VectorSegment,
}

/// Collection key the store-wide [`FlushArtifact::HnswIdMap`] is tracked under.
pub const ID_MAP_KEY: &str = "";

/// One artifact's mutation generation and the generation last made durable.
#[derive(Debug, Clone, Copy)]
struct Generation {
    current: u64,
    flushed: u64,
}

impl Generation {
    /// An artifact nothing has reported on. No evidence says its stored form
    /// matches memory, so it starts dirty and the next flush writes it.
    const UNKNOWN: Self = Self {
        current: 1,
        flushed: 0,
    };
}

/// A write `flush` has planned for one artifact.
///
/// Carries the generation captured when the artifact was serialized. Passing
/// it to `FlushGens::mark_flushed` records that generation, never a later
/// one, so a mutation made after the capture keeps the artifact dirty.
#[derive(Debug, Clone)]
pub(crate) struct ArtifactFlush {
    artifact: FlushArtifact,
    key: String,
    generation: u64,
}

impl ArtifactFlush {
    /// The collection this write belongs to.
    // Read only by the vector segment write, which wasm32 does not have.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn key(&self) -> &str {
        &self.key
    }
}

/// Generation counters, write counters, and last-written meta values for
/// every dirty-tracked flush artifact.
#[derive(Debug, Default)]
pub(crate) struct FlushGens {
    generations: Mutex<HashMap<FlushArtifact, HashMap<String, Generation>>>,
    writes: Mutex<HashMap<FlushArtifact, HashMap<String, u64>>>,
    meta_written: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
}

impl FlushGens {
    /// Record a mutation of `artifact` for `key`.
    pub(crate) fn bump(&self, artifact: FlushArtifact, key: &str) {
        let mut gens = self.generations.lock_or_recover();
        Self::entry(&mut gens, artifact, key).current += 1;
    }

    /// Record a mutation of `artifact` for every key in `keys`.
    pub(crate) fn bump_all<'k>(
        &self,
        artifact: FlushArtifact,
        keys: impl Iterator<Item = &'k str>,
    ) {
        let mut gens = self.generations.lock_or_recover();
        for key in keys {
            Self::entry(&mut gens, artifact, key).current += 1;
        }
    }

    /// Record a mutation of the durable vector rows under `index_key`.
    ///
    /// A segment is built from a prefix scan of `v:<collection>:`, and that
    /// prefix also covers every bucket whose key extends the collection's with
    /// `:<field>`. A row written under `a:b` therefore changes the segment of
    /// `a` as well, so every `:`-delimited ancestor of `index_key` is bumped.
    /// A document id may itself contain `:`, so a row under `a` can land in
    /// the prefix of a tracked bucket `a:<field>`: every tracked descendant
    /// is bumped too.
    pub(crate) fn bump_vector_rows(&self, index_key: &str) {
        let mut gens = self.generations.lock_or_recover();
        let descendant_prefix = format!("{index_key}:");
        for (key, generation) in gens.entry(FlushArtifact::VectorSegment).or_default() {
            if key.starts_with(&descendant_prefix) {
                generation.current += 1;
            }
        }
        let ancestors = index_key
            .char_indices()
            .filter(|&(_, c)| c == ':')
            .map(|(i, _)| &index_key[..i]);
        for key in std::iter::once(index_key).chain(ancestors) {
            Self::entry(&mut gens, FlushArtifact::VectorSegment, key).current += 1;
        }
    }

    /// Record that `artifact` for `key` matches its stored form.
    ///
    /// Used when an artifact is loaded from a stored copy that decoded and
    /// validated, so the first flush does not rewrite what it just read.
    pub(crate) fn mark_clean(&self, artifact: FlushArtifact, key: &str) {
        let mut gens = self.generations.lock_or_recover();
        let entry = Self::entry(&mut gens, artifact, key);
        entry.flushed = entry.current;
    }

    /// Plan a write of `artifact` for `key`.
    ///
    /// Returns `None` when the artifact is clean and `full` is false. Call it
    /// under the lock the artifact is serialized under, so the captured
    /// generation describes exactly the bytes being written.
    pub(crate) fn plan(
        &self,
        artifact: FlushArtifact,
        key: &str,
        full: bool,
    ) -> Option<ArtifactFlush> {
        let mut gens = self.generations.lock_or_recover();
        let entry = *Self::entry(&mut gens, artifact, key);
        if !full && entry.current == entry.flushed {
            return None;
        }
        Some(ArtifactFlush {
            artifact,
            key: key.to_owned(),
            generation: entry.current,
        })
    }

    /// Record that the write planned as `planned` is durable.
    ///
    /// Call it only after the artifact's last durable write succeeded.
    pub(crate) fn mark_flushed(&self, planned: &ArtifactFlush) {
        let mut gens = self.generations.lock_or_recover();
        let entry = Self::entry(&mut gens, planned.artifact, &planned.key);
        entry.flushed = entry.flushed.max(planned.generation);
    }

    /// Count one successful write of the artifact planned as `planned`.
    pub(crate) fn record_write(&self, planned: &ArtifactFlush) {
        let mut writes = self.writes.lock_or_recover();
        *writes
            .entry(planned.artifact)
            .or_default()
            .entry(planned.key.clone())
            .or_default() += 1;
    }

    /// Whether `artifact` for `key` has mutations no flush has made durable.
    pub(crate) fn is_dirty(&self, artifact: FlushArtifact, key: &str) -> bool {
        let gens = self.generations.lock_or_recover();
        let entry = gens
            .get(&artifact)
            .and_then(|by_key| by_key.get(key))
            .copied()
            .unwrap_or(Generation::UNKNOWN);
        entry.current != entry.flushed
    }

    /// Successful writes of `artifact` for `key` since this handle opened.
    pub(crate) fn write_count(&self, artifact: FlushArtifact, key: &str) -> u64 {
        self.writes
            .lock_or_recover()
            .get(&artifact)
            .and_then(|by_key| by_key.get(key))
            .copied()
            .unwrap_or(0)
    }

    /// Whether `value` differs from the value last written under meta `key`
    /// by this handle. A key this handle never wrote counts as changed.
    pub(crate) fn meta_changed(&self, key: &[u8], value: &[u8]) -> bool {
        self.meta_written
            .lock_or_recover()
            .get(key)
            .is_none_or(|written| written.as_slice() != value)
    }

    /// Record that `value` is now durable under meta `key`.
    pub(crate) fn mark_meta_written(&self, key: &[u8], value: Vec<u8>) {
        self.meta_written
            .lock_or_recover()
            .insert(key.to_vec(), value);
    }

    fn entry<'g>(
        gens: &'g mut HashMap<FlushArtifact, HashMap<String, Generation>>,
        artifact: FlushArtifact,
        key: &str,
    ) -> &'g mut Generation {
        gens.entry(artifact)
            .or_default()
            .entry(key.to_owned())
            .or_insert(Generation::UNKNOWN)
    }
}

/// A per-collection map whose mutable access marks the touched collections
/// dirty for `artifact`.
///
/// The only way to reach the map is `TrackedMap::lock_or_recover` or
/// `TrackedMap::lock`, and their guard bumps the generation on every
/// mutable path. Per-key methods bump only that key. Any other mutable use of
/// the whole map bumps every key present when the guard drops, which can
/// over-mark but never under-mark.
///
/// `pub` only because the public `index_row` takes it; the module itself is
/// crate-private.
#[derive(Debug)]
pub struct TrackedMap<V> {
    inner: Mutex<HashMap<String, V>>,
    gens: Arc<FlushGens>,
    artifact: FlushArtifact,
}

impl<V> TrackedMap<V> {
    pub(crate) fn new(
        map: HashMap<String, V>,
        gens: Arc<FlushGens>,
        artifact: FlushArtifact,
    ) -> Self {
        Self {
            inner: Mutex::new(map),
            gens,
            artifact,
        }
    }

    /// Lock the map, recovering the guard from a poisoned mutex.
    pub(crate) fn lock_or_recover(&self) -> TrackedMapGuard<'_, V> {
        self.guard(self.inner.lock_or_recover())
    }

    /// Lock the map, reporting a poisoned mutex to the caller.
    pub(crate) fn lock(&self) -> LockResult<TrackedMapGuard<'_, V>> {
        match self.inner.lock() {
            Ok(guard) => Ok(self.guard(guard)),
            Err(poisoned) => Err(PoisonError::new(self.guard(poisoned.into_inner()))),
        }
    }

    fn guard<'a>(&'a self, guard: MutexGuard<'a, HashMap<String, V>>) -> TrackedMapGuard<'a, V> {
        TrackedMapGuard {
            guard,
            owner: self,
            touched_all: false,
        }
    }
}

/// Guard for a [`TrackedMap`]. Reads go through `Deref`; every mutable path
/// bumps the generation while the lock is still held.
pub(crate) struct TrackedMapGuard<'a, V> {
    guard: MutexGuard<'a, HashMap<String, V>>,
    owner: &'a TrackedMap<V>,
    touched_all: bool,
}

impl<V> TrackedMapGuard<'_, V> {
    /// Mutable access to one entry. Marks `key` dirty when it exists.
    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        let found = self.guard.get_mut(key);
        if found.is_some() {
            self.owner.gens.bump(self.owner.artifact, key);
        }
        found
    }

    /// The entry for `key`, created by `make` when absent. Marks `key` dirty.
    pub(crate) fn get_or_insert_with(&mut self, key: &str, make: impl FnOnce() -> V) -> &mut V {
        self.owner.gens.bump(self.owner.artifact, key);
        self.guard.entry(key.to_owned()).or_insert_with(make)
    }

    /// Insert `value` under `key`. Marks `key` dirty.
    pub(crate) fn insert(&mut self, key: String, value: V) -> Option<V> {
        self.owner.gens.bump(self.owner.artifact, &key);
        self.guard.insert(key, value)
    }

    /// Insert `value` under `key` and mark `key` clean.
    ///
    /// Only for a value decoded from the artifact's stored form, which then
    /// already matches what a flush would write.
    pub(crate) fn insert_clean(&mut self, key: String, value: V) -> Option<V> {
        self.owner.gens.mark_clean(self.owner.artifact, &key);
        self.guard.insert(key, value)
    }

    /// Remove the entry for `key`. Marks `key` dirty.
    pub(crate) fn remove(&mut self, key: &str) -> Option<V> {
        self.owner.gens.bump(self.owner.artifact, key);
        self.guard.remove(key)
    }
}

impl<V> Deref for TrackedMapGuard<'_, V> {
    type Target = HashMap<String, V>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<V> DerefMut for TrackedMapGuard<'_, V> {
    /// Whole-map mutable access. Which keys change is unknown, so every key
    /// present when the guard drops is marked dirty.
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.touched_all = true;
        &mut self.guard
    }
}

impl<V> Drop for TrackedMapGuard<'_, V> {
    fn drop(&mut self) {
        // Runs before the `MutexGuard` field drops, so the bump is still
        // under the lock the mutation was made under.
        if self.touched_all {
            self.owner
                .gens
                .bump_all(self.owner.artifact, self.guard.keys().map(String::as_str));
        }
    }
}

/// A single tracked artifact whose mutable access marks it dirty.
#[derive(Debug)]
pub(crate) struct TrackedCell<T> {
    inner: Mutex<T>,
    gens: Arc<FlushGens>,
    artifact: FlushArtifact,
    key: &'static str,
}

impl<T> TrackedCell<T> {
    pub(crate) fn new(
        value: T,
        gens: Arc<FlushGens>,
        artifact: FlushArtifact,
        key: &'static str,
    ) -> Self {
        Self {
            inner: Mutex::new(value),
            gens,
            artifact,
            key,
        }
    }

    /// Lock the value, recovering the guard from a poisoned mutex.
    pub(crate) fn lock_or_recover(&self) -> TrackedCellGuard<'_, T> {
        TrackedCellGuard {
            guard: self.inner.lock_or_recover(),
            owner: self,
            touched: false,
        }
    }
}

/// Guard for a [`TrackedCell`]. Any mutable access marks the artifact dirty.
pub(crate) struct TrackedCellGuard<'a, T> {
    guard: MutexGuard<'a, T>,
    owner: &'a TrackedCell<T>,
    touched: bool,
}

impl<T> Deref for TrackedCellGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for TrackedCellGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.touched = true;
        &mut self.guard
    }
}

impl<T> Drop for TrackedCellGuard<'_, T> {
    fn drop(&mut self) {
        // Still under the lock: the `MutexGuard` field drops after this.
        if self.touched {
            self.owner.gens.bump(self.owner.artifact, self.owner.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> TrackedMap<u32> {
        let gens = Arc::new(FlushGens::default());
        let mut initial = HashMap::new();
        initial.insert("a".to_string(), 1);
        initial.insert("b".to_string(), 2);
        let map = TrackedMap::new(initial, Arc::clone(&gens), FlushArtifact::HnswGraph);
        gens.mark_clean(FlushArtifact::HnswGraph, "a");
        gens.mark_clean(FlushArtifact::HnswGraph, "b");
        map
    }

    #[test]
    fn an_unknown_artifact_starts_dirty() {
        let gens = FlushGens::default();
        assert!(gens.is_dirty(FlushArtifact::HnswGraph, "never_seen"));
    }

    #[test]
    fn reads_leave_every_key_clean() {
        let map = map();
        let guard = map.lock_or_recover();
        assert_eq!(guard.get("a"), Some(&1));
        drop(guard);
        assert!(!map.gens.is_dirty(FlushArtifact::HnswGraph, "a"));
        assert!(!map.gens.is_dirty(FlushArtifact::HnswGraph, "b"));
    }

    #[test]
    fn get_mut_marks_only_that_key() {
        let map = map();
        if let Some(v) = map.lock_or_recover().get_mut("a") {
            *v += 1;
        }
        assert!(map.gens.is_dirty(FlushArtifact::HnswGraph, "a"));
        assert!(!map.gens.is_dirty(FlushArtifact::HnswGraph, "b"));
    }

    #[test]
    fn whole_map_mutation_marks_every_key() {
        let map = map();
        map.lock_or_recover().values_mut().for_each(|v| *v += 1);
        assert!(map.gens.is_dirty(FlushArtifact::HnswGraph, "a"));
        assert!(map.gens.is_dirty(FlushArtifact::HnswGraph, "b"));
    }

    #[test]
    fn a_mutation_after_the_plan_keeps_the_artifact_dirty() {
        let map = map();
        map.lock_or_recover().insert("a".to_string(), 5);
        let planned = map
            .gens
            .plan(FlushArtifact::HnswGraph, "a", false)
            .expect("dirty artifact is planned");
        map.lock_or_recover().insert("a".to_string(), 6);
        map.gens.mark_flushed(&planned);
        assert!(map.gens.is_dirty(FlushArtifact::HnswGraph, "a"));
    }

    #[test]
    fn a_clean_artifact_is_planned_only_when_full() {
        let map = map();
        assert!(
            map.gens
                .plan(FlushArtifact::HnswGraph, "a", false)
                .is_none()
        );
        assert!(map.gens.plan(FlushArtifact::HnswGraph, "a", true).is_some());
    }

    #[test]
    fn vector_rows_mark_the_bucket_and_every_ancestor() {
        let gens = FlushGens::default();
        for key in ["docs", "docs:emb", "other"] {
            gens.mark_clean(FlushArtifact::VectorSegment, key);
        }
        gens.bump_vector_rows("docs:emb");
        assert!(gens.is_dirty(FlushArtifact::VectorSegment, "docs:emb"));
        assert!(gens.is_dirty(FlushArtifact::VectorSegment, "docs"));
        assert!(!gens.is_dirty(FlushArtifact::VectorSegment, "other"));
    }

    #[test]
    fn vector_rows_mark_every_tracked_descendant_bucket() {
        let gens = FlushGens::default();
        for key in ["docs", "docs:emb", "docs2"] {
            gens.mark_clean(FlushArtifact::VectorSegment, key);
        }
        gens.bump_vector_rows("docs");
        assert!(gens.is_dirty(FlushArtifact::VectorSegment, "docs"));
        assert!(gens.is_dirty(FlushArtifact::VectorSegment, "docs:emb"));
        assert!(
            !gens.is_dirty(FlushArtifact::VectorSegment, "docs2"),
            "a name that only shares the prefix is not a descendant"
        );
    }

    #[test]
    fn a_cell_is_dirty_only_after_mutable_access() {
        let gens = Arc::new(FlushGens::default());
        gens.mark_clean(FlushArtifact::HnswIdMap, ID_MAP_KEY);
        let cell = TrackedCell::new(
            0u32,
            Arc::clone(&gens),
            FlushArtifact::HnswIdMap,
            ID_MAP_KEY,
        );
        assert_eq!(*cell.lock_or_recover(), 0);
        assert!(!gens.is_dirty(FlushArtifact::HnswIdMap, ID_MAP_KEY));
        *cell.lock_or_recover() = 1;
        assert!(gens.is_dirty(FlushArtifact::HnswIdMap, ID_MAP_KEY));
    }
}
