// SPDX-License-Identifier: Apache-2.0
//! SQL-visitor lowering for vector-primary writes.

mod identity;
mod insert;
mod mutation;

pub(super) use insert::lower_vector_primary_insert;
pub(super) use mutation::{
    lower_vector_primary_delete, lower_vector_primary_truncate, lower_vector_primary_update,
};
