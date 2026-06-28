//! List decoding.
//!
//! A JSON array maps to a `List`; each element is coerced to the inner type. A
//! failed element becomes a null element. A non-array value (or a null/absent
//! row) yields a null list.

use polars_core::chunked_array::builder::AnonymousOwnedListBuilder;
use polars_core::prelude::*;
use simd_json::BorrowedValue;

use super::{build_field_series, build_field_series_with_diagnostics};
use crate::diagnostics::{is_json_null, DiagRowValue, DiagRowValues, DiagnosticsCollector};
use crate::dtype::ir_to_polars;
use crate::ir::SchemaType;

/// Build a `List` `Series` whose elements match `inner`.
///
/// `rows` holds one optional JSON value per output row. Non-array values (and
/// `None` rows) become null lists.
pub fn build_list_series(
    name: PlSmallStr,
    rows: super::scalar::RowValues,
    inner: &SchemaType,
    coerce_flag: bool,
) -> PolarsResult<Series> {
    let inner_dtype = ir_to_polars(inner)?;
    let list_dtype = DataType::List(Box::new(inner_dtype.clone()));

    // Build each row's inner Series first (owned), so the builder can borrow them.
    let mut per_row: Vec<Option<Series>> = Vec::with_capacity(rows.len());
    for opt in rows {
        match opt {
            Some(BorrowedValue::Array(arr)) => {
                let elems: Vec<Option<&BorrowedValue>> = arr.iter().map(Some).collect();
                let inner_series = build_field_series("".into(), &elems, inner, coerce_flag)?;
                per_row.push(Some(inner_series));
            }
            // Non-array, JSON null, or absent row -> null list.
            _ => per_row.push(None),
        }
    }

    let mut builder = AnonymousOwnedListBuilder::new(name, rows.len(), Some(inner_dtype));
    for opt_s in &per_row {
        match opt_s {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    let list = builder.finish().into_series();

    // Guarantee the declared inner dtype even for empty/all-null lists.
    if list.dtype() == &list_dtype {
        Ok(list)
    } else {
        list.cast(&list_dtype)
    }
}

/// Diagnostic variant of [`build_list_series`].
pub(crate) fn build_list_series_with_diagnostics<'a, 'v>(
    name: PlSmallStr,
    rows: DiagRowValues<'a, 'v>,
    inner: &SchemaType,
    coerce_flag: bool,
    path: &str,
    diagnostics: &mut DiagnosticsCollector,
) -> PolarsResult<Series> {
    let inner_dtype = ir_to_polars(inner)?;
    let list_dtype = DataType::List(Box::new(inner_dtype.clone()));
    let child_path = format!("{path}[]");

    let mut per_row: Vec<Option<Series>> = Vec::with_capacity(rows.len());
    for row in rows {
        match row.value {
            Some(BorrowedValue::Array(arr)) => {
                let elems: Vec<DiagRowValue<'_, '_>> = arr
                    .iter()
                    .map(|value| DiagRowValue {
                        row_idx: row.row_idx,
                        value: Some(value),
                    })
                    .collect();
                let inner_series = build_field_series_with_diagnostics(
                    "".into(),
                    &elems,
                    inner,
                    coerce_flag,
                    &child_path,
                    diagnostics,
                )?;
                per_row.push(Some(inner_series));
            }
            Some(value) if !is_json_null(value) => {
                diagnostics.record_value_mismatch(row.row_idx, path, "array", value);
                per_row.push(None);
            }
            _ => per_row.push(None),
        }
    }

    let mut builder = AnonymousOwnedListBuilder::new(name, rows.len(), Some(inner_dtype));
    for opt_s in &per_row {
        match opt_s {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    let list = builder.finish().into_series();

    if list.dtype() == &list_dtype {
        Ok(list)
    } else {
        list.cast(&list_dtype)
    }
}
