// SPDX-License-Identifier: Apache-2.0

//! Physical execution of `TextOp` variants for the Lite data plane, split by
//! responsibility: `dispatch` (variant routing), `search` (BM25, score scan,
//! phrase), `hybrid` (vector + text fusion), `sync` (Origin index frames).

mod dispatch;
mod hybrid;
mod search;
mod sync;

pub(crate) use dispatch::{execute_text_op, execute_text_op_admitted, execute_text_op_on_field};
