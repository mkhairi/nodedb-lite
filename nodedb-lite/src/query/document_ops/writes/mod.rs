// SPDX-License-Identifier: Apache-2.0
//! Document writes with shared text mutation admission.
mod batch;
mod bulk_delete;
mod bulk_update;
mod delete;
mod insert;
mod truncate;
mod update;
mod upsert;
pub use batch::batch_insert;
pub(crate) use batch::{batch_insert_admitted, batch_insert_coordinated};
pub use bulk_delete::bulk_delete;
pub use bulk_update::bulk_update;
pub(crate) use bulk_update::bulk_update_coordinated;
pub use delete::point_delete;
pub(crate) use delete::{point_delete_admitted, point_delete_coordinated};
pub use insert::{point_insert, point_put};
pub(crate) use insert::{point_insert_admitted, point_insert_coordinated, point_put_coordinated};
pub(super) use nodedb_physical::physical_plan::document::types::UpdateValue;
pub use truncate::truncate;
pub(crate) use truncate::truncate_coordinated;
pub use update::point_update;
pub(crate) use update::{point_update_admitted, point_update_coordinated};
pub use upsert::upsert;
pub(crate) use upsert::upsert_coordinated;
