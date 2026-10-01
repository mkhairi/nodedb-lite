// SPDX-License-Identifier: Apache-2.0
//! Conversion between document, strict, and columnar storage.

mod schema;
mod source;
mod targets;

pub(in crate::query) use schema::default_convert_schema;
