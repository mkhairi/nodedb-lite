// SPDX-License-Identifier: Apache-2.0

//! Predicates a scan can answer from a secondary index.
//!
//! The shared planner rewrites equality onto document and strict indexes
//! only. A scan whose WHERE clause bounds an indexed field (`=`, `>`, `>=`,
//! `<`, `<=`, `BETWEEN`) can read the index's candidates for that range
//! instead of every row: a range on any indexed engine, an equality on a
//! key-value one. The scan still applies every one of its filters to the
//! candidates, so the bound only narrows which rows are read, never which
//! rows match.

use std::sync::Arc;

use nodedb_sql::types::filter::{CompareOp, Filter, FilterExpr};
use nodedb_sql::types_expr::{BinaryOp, SqlExpr, SqlValue};
use nodedb_types::value::Value;

use crate::index::{IndexCatalog, IndexDef, IndexEngine, canonical_field, field_spec};
use crate::query::filter_convert::sql_value_to_value;

/// A range over one indexed field.
pub(super) struct IndexRange {
    pub def: Arc<IndexDef>,
    pub lower: Option<Value>,
    pub upper: Option<Value>,
}

#[derive(Clone, Copy)]
enum Side {
    Lower,
    Upper,
    /// An equality: both a lower and an upper bound.
    Both,
}

impl Side {
    fn lower(self) -> bool {
        matches!(self, Side::Lower | Side::Both)
    }

    fn upper(self) -> bool {
        matches!(self, Side::Upper | Side::Both)
    }
}

/// One bound on one field.
struct Bound {
    field: String,
    side: Side,
    value: SqlValue,
}

fn compare_side(op: CompareOp) -> Option<Side> {
    match op {
        CompareOp::Gt | CompareOp::Ge => Some(Side::Lower),
        CompareOp::Lt | CompareOp::Le => Some(Side::Upper),
        CompareOp::Eq => Some(Side::Both),
        CompareOp::Ne => None,
    }
}

fn binary_side(op: BinaryOp) -> Option<Side> {
    match op {
        BinaryOp::Gt | BinaryOp::Ge => Some(Side::Lower),
        BinaryOp::Lt | BinaryOp::Le => Some(Side::Upper),
        BinaryOp::Eq => Some(Side::Both),
        _ => None,
    }
}

fn flip(side: Side) -> Side {
    match side {
        Side::Lower => Side::Upper,
        Side::Upper => Side::Lower,
        Side::Both => Side::Both,
    }
}

/// Bounds in a conjunct of an expression filter.
fn expr_bounds(expr: &SqlExpr, out: &mut Vec<Bound>) {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOp::And,
            right,
        } => {
            expr_bounds(left, out);
            expr_bounds(right, out);
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let Some(side) = binary_side(*op) else {
                return;
            };
            match (left.as_ref(), right.as_ref()) {
                (SqlExpr::Column { name, .. }, SqlExpr::Literal(value)) => out.push(Bound {
                    field: name.clone(),
                    side,
                    value: value.clone(),
                }),
                (SqlExpr::Literal(value), SqlExpr::Column { name, .. }) => out.push(Bound {
                    field: name.clone(),
                    side: flip(side),
                    value: value.clone(),
                }),
                _ => {}
            }
        }
        SqlExpr::Between {
            expr,
            low,
            high,
            negated: false,
        } => {
            if let (SqlExpr::Column { name, .. }, SqlExpr::Literal(lo), SqlExpr::Literal(hi)) =
                (expr.as_ref(), low.as_ref(), high.as_ref())
            {
                out.push(Bound {
                    field: name.clone(),
                    side: Side::Lower,
                    value: lo.clone(),
                });
                out.push(Bound {
                    field: name.clone(),
                    side: Side::Upper,
                    value: hi.clone(),
                });
            }
        }
        _ => {}
    }
}

/// Bounds in the top-level conjuncts of `filters`.
fn filter_bounds(filters: &[Filter], out: &mut Vec<Bound>) {
    for filter in filters {
        match &filter.expr {
            FilterExpr::Comparison { field, op, value } => {
                if let Some(side) = compare_side(*op) {
                    out.push(Bound {
                        field: field.clone(),
                        side,
                        value: value.clone(),
                    });
                }
            }
            FilterExpr::Between { field, low, high } => {
                out.push(Bound {
                    field: field.clone(),
                    side: Side::Lower,
                    value: low.clone(),
                });
                out.push(Bound {
                    field: field.clone(),
                    side: Side::Upper,
                    value: high.clone(),
                });
            }
            FilterExpr::And(inner) => filter_bounds(inner, out),
            FilterExpr::Expr(expr) => expr_bounds(expr, out),
            _ => {}
        }
    }
}

/// The index range the scan filters of `collection` imply, when a bounded
/// field has a full, case-sensitive, scalar index over `engine` rows.
pub(super) fn index_range(
    indexes: &IndexCatalog,
    collection: &str,
    engine: IndexEngine,
    filters: &[Filter],
) -> Option<IndexRange> {
    let mut bounds = Vec::new();
    filter_bounds(filters, &mut bounds);
    for bound in &bounds {
        let (path, is_array) = canonical_field(&bound.field);
        if is_array {
            continue;
        }
        let Some(def) = indexes.def_on_field(collection, &field_spec(&path, false)) else {
            continue;
        };
        // A partial index lacks rows outside its predicate, and a
        // case-insensitive one orders folded strings: neither can list every
        // row a range matches.
        if def.engine != engine || def.predicate.is_some() || def.case_insensitive {
            continue;
        }
        let value_of = |side: fn(Side) -> bool| {
            bounds
                .iter()
                .filter(|b| b.field == bound.field && side(b.side))
                .find_map(|b| match sql_value_to_value(&b.value) {
                    Ok(Value::Null) | Err(_) => None,
                    Ok(v) => Some(v),
                })
        };
        let lower = value_of(Side::lower);
        let upper = value_of(Side::upper);
        if lower.is_none() && upper.is_none() {
            continue;
        }
        return Some(IndexRange { def, lower, upper });
    }
    None
}
