// SPDX-License-Identifier: Apache-2.0
//! Plan-policy slots that Lite does not execute.
//!
//! Origin attaches row-level-security programs and RETURNING projections to
//! write ops. Lite enforces neither. Executing the write while dropping the
//! slot would apply an unpoliced write, so every dispatch arm that can carry
//! one rejects it here instead.

use nodedb_physical::physical_plan::ReturningSpec;
use nodedb_types::{RlsWriteCheck, WriteGateDecision};

use crate::error::LiteError;

/// Reject a plan carrying a policy slot Lite cannot honour.
///
/// `rls` holds the RLS read filters on the op. Write checks are a separate
/// type; see [`deny_write_check`]. `op` names the variant, so the message
/// points at the statement that produced it.
pub(crate) fn deny_policy(
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

/// Reject a write whose RLS check is anything but "admit unconditionally".
///
/// `AdmitAll` is the only decision Lite can honour by executing the write:
/// there is no policy to evaluate and none was dropped. `Evaluate` carries a
/// predicate Lite cannot run, and `DenyNotInjected` means the plan never went
/// through injection — both must refuse rather than write unpoliced rows.
pub(crate) fn deny_write_check(op: &str, checks: &[&RlsWriteCheck]) -> Result<(), LiteError> {
    if checks
        .iter()
        .any(|check| !matches!(check.decision(), WriteGateDecision::AdmitAll))
    {
        return Err(LiteError::Unsupported {
            detail: format!(
                "{op}: row-level security is enforced by the Origin data plane \
                 and has no equivalent on the single-node Lite engine"
            ),
        });
    }
    Ok(())
}
