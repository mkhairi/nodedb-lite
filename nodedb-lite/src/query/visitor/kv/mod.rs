// SPDX-License-Identifier: Apache-2.0
//! SQL-visitor lowering for KV SqlPlan variants: KvInsert.

mod encoding;
mod insert;

pub(super) use insert::lower_kv_insert;
