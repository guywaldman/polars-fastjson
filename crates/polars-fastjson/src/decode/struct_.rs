//! Struct decoding.
//!
//! A JSON object maps to a `Struct`; fields are projected recursively. Fields
//! present in the JSON but absent from the schema are dropped (`extra="ignore"`).
//! Missing fields and JSON `null` fields become null. A non-object value (or a
//! null/absent row) yields a null struct (the whole row).
//!
//! Strategy: for each schema field, navigate every row's object to that field
//! (producing one `Option<&BorrowedValue>` per row) and recurse via the shared
//! field dispatcher ([`super::build_field_series`]). The per-field child Series
//! are assembled with [`StructChunked::from_series`]. Rows whose value is not a
//! JSON object are marked null at the struct (outer) level.

use polars::prelude::*;
use simd_json::{BorrowedValue, StaticNode};

use super::{build_field_series, build_field_series_with_diagnostics, scalar::RowValues};
use crate::diagnostics::{is_json_null, DiagRowValue, DiagRowValues, DiagnosticsCollector};
use crate::ir::FieldIR;

/// Build a `Struct` `Series` described by `fields`.
///
/// `rows` holds one optional JSON value per output row. A row whose value is not
/// a JSON object (or is `None`) becomes a null struct.
pub fn build_struct_series(
    name: PlSmallStr,
    rows: RowValues,
    fields: &[FieldIR],
    coerce_flag: bool,
) -> PolarsResult<Series> {
    let len = rows.len();

    // Per-row outer validity: true when the row is a JSON object.
    let row_valid: Vec<bool> = rows
        .iter()
        .map(|opt| matches!(opt, Some(BorrowedValue::Object(_))))
        .collect();

    let mut children: Vec<Series> = Vec::with_capacity(fields.len());
    // Reusable per-field navigation buffer.
    let mut field_rows: Vec<Option<&BorrowedValue>> = Vec::with_capacity(len);

    for f in fields {
        field_rows.clear();
        // The JSON key to read from; defaults to the output field name.
        let key = f.json_key.as_deref().unwrap_or(f.name.as_str());
        for opt in rows {
            let navigated = match opt {
                Some(BorrowedValue::Object(map)) => {
                    // Missing field -> None, JSON null value -> None (null field).
                    match map.get(key) {
                        Some(BorrowedValue::Static(StaticNode::Null)) => None,
                        other => other,
                    }
                }
                // Non-object row: child is null here (the whole struct is nulled
                // via outer validity anyway).
                _ => None,
            };
            field_rows.push(navigated);
        }
        let child = build_field_series(f.name.as_str().into(), &field_rows, &f.dtype, coerce_flag)?;
        children.push(child);
    }

    // `StructChunked::from_series` rejects an empty field set, so a zero-field
    // struct is constructed as a full-null series of the empty struct dtype.
    let st = if children.is_empty() {
        let dtype = DataType::Struct(vec![]);
        return Ok(Series::full_null(name, len, &dtype));
    } else {
        StructChunked::from_series(name, len, children.iter())?
    };

    let mut series = st.into_series();

    // Apply outer validity: any non-object row becomes a null struct.
    if !row_valid.iter().all(|&v| v) {
        let mask = BooleanChunked::from_slice("__mask".into(), &row_valid);
        let null = Series::full_null(series.name().clone(), len, series.dtype());
        // `Series::zip_with(mask, other)` keeps `self` where mask is true,
        // `other` (null) where false.
        series = series.zip_with(&mask, &null)?;
    }

    Ok(series)
}

/// Diagnostic variant of [`build_struct_series`].
pub(crate) fn build_struct_series_with_diagnostics<'a, 'v>(
    name: PlSmallStr,
    rows: DiagRowValues<'a, 'v>,
    fields: &[FieldIR],
    coerce_flag: bool,
    path: &str,
    diagnostics: &mut DiagnosticsCollector,
) -> PolarsResult<Series> {
    let len = rows.len();

    let row_valid: Vec<bool> = rows
        .iter()
        .map(|row| matches!(row.value, Some(BorrowedValue::Object(_))))
        .collect();

    for row in rows {
        if let Some(value) = row.value {
            if !matches!(value, BorrowedValue::Object(_)) && !is_json_null(value) {
                diagnostics.record_value_mismatch(row.row_idx, path, "object", value);
            }
        }
    }

    let mut children: Vec<Series> = Vec::with_capacity(fields.len());
    let mut field_rows: Vec<DiagRowValue<'_, '_>> = Vec::with_capacity(len);

    for f in fields {
        field_rows.clear();
        let key = f.json_key.as_deref().unwrap_or(f.name.as_str());
        for row in rows {
            let navigated = match row.value {
                Some(BorrowedValue::Object(map)) => match map.get(key) {
                    Some(BorrowedValue::Static(StaticNode::Null)) => None,
                    other => other,
                },
                _ => None,
            };
            field_rows.push(DiagRowValue {
                row_idx: row.row_idx,
                value: navigated,
            });
        }
        let child_path = format!("{path}.{}", f.name);
        let child = build_field_series_with_diagnostics(
            f.name.as_str().into(),
            &field_rows,
            &f.dtype,
            coerce_flag,
            &child_path,
            diagnostics,
        )?;
        children.push(child);
    }

    let st = if children.is_empty() {
        let dtype = DataType::Struct(vec![]);
        return Ok(Series::full_null(name, len, &dtype));
    } else {
        StructChunked::from_series(name, len, children.iter())?
    };

    let mut series = st.into_series();
    if !row_valid.iter().all(|&v| v) {
        let mask = BooleanChunked::from_slice("__mask".into(), &row_valid);
        let null = Series::full_null(series.name().clone(), len, series.dtype());
        series = series.zip_with(&mask, &null)?;
    }

    Ok(series)
}
