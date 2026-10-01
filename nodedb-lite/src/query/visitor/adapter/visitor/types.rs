// SPDX-License-Identifier: Apache-2.0

//! SQL visitor admission context and boxed result future.

use crate::error::LiteError;
use crate::query::engine::LiteQueryEngine;
use crate::storage::engine::StorageEngine;
use nodedb_types::result::QueryResult;
use std::future::Future;
use std::pin::Pin;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type LiteFut<'a> =
    Pin<Box<dyn Future<Output = Result<QueryResult, LiteError>> + Send + 'a>>;

#[cfg(target_arch = "wasm32")]
pub(crate) type LiteFut<'a> = Pin<Box<dyn Future<Output = Result<QueryResult, LiteError>> + 'a>>;

pub(crate) struct LiteVisitor<'a, S: StorageEngine> {
    pub(crate) engine: &'a LiteQueryEngine<S>,
    pub(crate) permit: Option<&'a crate::engine::fts::coordinator::TextMutationPermit>,
}

impl<'a, S: StorageEngine> LiteVisitor<'a, S> {
    pub(super) fn mutation_permit(
        &self,
    ) -> Result<&'a crate::engine::fts::coordinator::TextMutationPermit, LiteError> {
        self.permit.ok_or_else(|| LiteError::Unsupported {
            detail: "source mutation lacks admission: execute through the SQL dispatcher".into(),
        })
    }
}
