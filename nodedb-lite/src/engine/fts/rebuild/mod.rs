// SPDX-License-Identifier: Apache-2.0

//! Authoritative bounded source pages for private collection replacements.

mod build;
mod ordinary;
mod row_budget;

pub(crate) use build::build_collection_replacement;
pub(crate) use row_budget::{PAGE_BYTES, PAGE_RECORDS, check_row_budget};
