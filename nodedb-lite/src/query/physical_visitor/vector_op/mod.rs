// SPDX-License-Identifier: Apache-2.0
//! Dispatch for vector operations on the Lite executor.

mod direct;
mod dispatch;
mod sparse;

pub(super) use dispatch::{execute_vector_op, execute_vector_op_admitted};
