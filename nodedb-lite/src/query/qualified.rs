// SPDX-License-Identifier: Apache-2.0

//! Collection qualification for the single-database Lite engine.
//!
//! Origin qualifies every collection by database id, so plan ops carry a
//! [`QualifiedCollection`] rather than a bare name. Lite serves exactly one
//! database, `DatabaseId::DEFAULT`, for which qualification is the identity —
//! but the plan types still demand the qualified form, and building it here
//! keeps the assumption in one place instead of at every plan-construction
//! site.

use nodedb_types::id::{DatabaseId, QualifiedCollection};

/// Qualify `collection` for the only database Lite has.
pub(crate) fn qualify(collection: impl AsRef<str>) -> QualifiedCollection {
    QualifiedCollection::new(DatabaseId::DEFAULT, collection.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_database_qualifies_to_the_bare_name() {
        assert_eq!(qualify("users").as_str(), "users");
    }
}
