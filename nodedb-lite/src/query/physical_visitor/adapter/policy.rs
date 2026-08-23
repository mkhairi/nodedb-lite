// SPDX-License-Identifier: Apache-2.0
//! Plan-policy slots that Lite does not execute.
//!
//! Origin attaches row-level-security programs and RETURNING projections to
//! write ops. Lite enforces neither. Executing the write while dropping the
//! slot would apply an unpoliced write, so every dispatch arm that can carry
//! one rejects it here instead.

use nodedb_physical::physical_plan::ReturningSpec;

use crate::error::LiteError;

/// Reject a plan carrying a policy slot Lite cannot honour.
///
/// `rls` holds every RLS program on the op: read filters and write checks
/// alike. `op` names the variant, so the message points at the statement that
/// produced it.
pub(super) fn deny_policy(
    op: &str,
    returning: Option<&ReturningSpec>,
    rls: &[&[u8]],
) -> Result<(), LiteError> {
    if !rls.iter().all(|program| program.is_empty()) {
        return Err(LiteError::Unsupported {
            detail: format!(
                "{op}: row-level security is enforced by the Origin data plane \
                 and has no equivalent on the single-node Lite engine"
            ),
        });
    }
    if returning.is_some() {
        return Err(LiteError::Unsupported {
            detail: format!("{op}: RETURNING is unsupported on the Lite engine"),
        });
    }
    Ok(())
}
