// SPDX-License-Identifier: Apache-2.0

//! Durable per-collection search declarations and revision tombstones.

mod storage;
mod types;

pub(crate) use storage::{load_declarations, persist_declaration, persist_declaration_tombstone};
pub(crate) use types::{DECLARATION_FORMAT, SearchDeclaration, SearchDeclarationRecord};
