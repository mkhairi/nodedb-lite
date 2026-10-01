// SPDX-License-Identifier: Apache-2.0

//! Durable checkpoint completeness and declaration compatibility.

use crate::error::LiteError;
use crate::storage::engine::StorageEngine;
use nodedb_types::Namespace;
use std::collections::BTreeMap;

use super::format::{LAYOUT_KEY, LAYOUT_PER_FIELD};

const KEY: &[u8] = b"fts_checkpoint_state";
const FORMAT: u32 = 1;
const SEMANTICS: u32 = 1;

#[derive(zerompk::ToMessagePack, zerompk::FromMessagePack)]
struct CheckpointState {
    format_version: u32,
    semantics_version: u32,
    complete: bool,
    revisions: BTreeMap<String, u64>,
}

async fn persist<S: StorageEngine>(storage: &S, state: &CheckpointState) -> Result<(), LiteError> {
    let bytes = zerompk::to_msgpack_vec(state).map_err(|error| LiteError::Serialization {
        detail: format!("FTS checkpoint state: {error}"),
    })?;
    storage.put(Namespace::Meta, KEY, &bytes).await
}

pub(crate) async fn persist_checkpoint_incomplete<S: StorageEngine>(
    storage: &S,
) -> Result<(), LiteError> {
    persist(
        storage,
        &CheckpointState {
            format_version: FORMAT,
            semantics_version: SEMANTICS,
            complete: false,
            revisions: BTreeMap::new(),
        },
    )
    .await
}

pub(crate) async fn persist_checkpoint_complete<S: StorageEngine>(
    storage: &S,
    revisions: &BTreeMap<String, u64>,
) -> Result<(), LiteError> {
    persist(
        storage,
        &CheckpointState {
            format_version: FORMAT,
            semantics_version: SEMANTICS,
            complete: true,
            revisions: revisions.clone(),
        },
    )
    .await
}

pub(crate) async fn checkpoint_compatible<S: StorageEngine>(
    storage: &S,
    revisions: &BTreeMap<String, u64>,
) -> Result<bool, LiteError> {
    let Some(bytes) = storage.get(Namespace::Meta, KEY).await? else {
        return Ok(false);
    };
    let Ok(state) = zerompk::from_msgpack::<CheckpointState>(&bytes) else {
        return Ok(false);
    };
    if state.format_version != FORMAT
        || state.semantics_version != SEMANTICS
        || !state.complete
        || state.revisions != *revisions
    {
        return Ok(false);
    }
    Ok(storage.get(Namespace::Fts, LAYOUT_KEY).await?.as_deref() == Some(&[LAYOUT_PER_FIELD]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::pagedb_storage::PagedbStorageMem;

    #[tokio::test]
    async fn checkpoint_requires_complete_current_semantics_and_matching_revisions() {
        let storage = PagedbStorageMem::open_in_memory().await.unwrap();
        let revisions = BTreeMap::from([("empty".into(), 2)]);
        assert!(!checkpoint_compatible(&storage, &revisions).await.unwrap());
        storage
            .put(Namespace::Fts, LAYOUT_KEY, &[LAYOUT_PER_FIELD])
            .await
            .unwrap();
        persist_checkpoint_complete(&storage, &revisions)
            .await
            .unwrap();
        assert!(checkpoint_compatible(&storage, &revisions).await.unwrap());
        assert!(
            !checkpoint_compatible(&storage, &BTreeMap::new())
                .await
                .unwrap()
        );
        persist_checkpoint_incomplete(&storage).await.unwrap();
        assert!(!checkpoint_compatible(&storage, &revisions).await.unwrap());
        persist(
            &storage,
            &CheckpointState {
                format_version: FORMAT,
                semantics_version: SEMANTICS + 1,
                complete: true,
                revisions: revisions.clone(),
            },
        )
        .await
        .unwrap();
        assert!(!checkpoint_compatible(&storage, &revisions).await.unwrap());
    }
}
