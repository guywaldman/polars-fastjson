//! Decode orchestrator.

pub mod coerce;
pub mod list;
pub mod required;
pub mod scalar;
pub mod struct_;

use polars::prelude::*;
use simd_json::{BorrowedValue, StaticNode};

use crate::diagnostics::{
    DecodeDiagnostics, DiagRowValue, DiagRowValues, DiagnosticsCollector, DiagnosticsOptions,
};
use crate::error::ErrorMode;
use crate::ir::SchemaType;
use crate::DecodeOptions;

/// Dispatch to the correct builder for `ir`, navigating `rows`.
///
/// This is the single recursion point shared by [`list`] and [`struct_`].
pub(crate) fn build_field_series(
    name: PlSmallStr,
    rows: scalar::RowValues,
    ir: &SchemaType,
    coerce_flag: bool,
) -> PolarsResult<Series> {
    match ir {
        SchemaType::List { inner } => list::build_list_series(name, rows, inner, coerce_flag),
        SchemaType::Struct { fields } => {
            struct_::build_struct_series(name, rows, fields, coerce_flag)
        }
        _ => scalar::build_scalar_series(name, rows, ir, coerce_flag),
    }
}

/// Diagnostic variant of [`build_field_series`].
///
/// This is intentionally separate from the hot-path dispatcher so diagnostics
/// do not add path/row bookkeeping to normal decodes.
pub(crate) fn build_field_series_with_diagnostics<'a, 'v>(
    name: PlSmallStr,
    rows: DiagRowValues<'a, 'v>,
    ir: &SchemaType,
    coerce_flag: bool,
    path: &str,
    diagnostics: &mut DiagnosticsCollector,
) -> PolarsResult<Series> {
    match ir {
        SchemaType::List { inner } => list::build_list_series_with_diagnostics(
            name,
            rows,
            inner,
            coerce_flag,
            path,
            diagnostics,
        ),
        SchemaType::Struct { fields } => struct_::build_struct_series_with_diagnostics(
            name,
            rows,
            fields,
            coerce_flag,
            path,
            diagnostics,
        ),
        _ => scalar::build_scalar_series_with_diagnostics(
            name,
            rows,
            ir,
            coerce_flag,
            path,
            diagnostics,
        ),
    }
}

/// Decode a JSON string column into a `Series` matching `ir`.
///
/// 1. Parse every non-null row into its own buffer (borrowed tape).
/// 2. Parse failures: null the row, or raise under [`ErrorMode::Error`].
/// 3. Dispatch to the struct/list/scalar builder over the parsed rows.
///
/// The output `Series` carries the same name as `values` and the dtype produced
/// by [`crate::dtype::ir_to_polars`].
pub fn decode_rows(
    values: &StringChunked,
    ir: &SchemaType,
    coerce_flag: bool,
    on_error: ErrorMode,
    strict_required_fields: bool,
) -> PolarsResult<Series> {
    let len = values.len();

    // Per-row owned byte buffers, one per input row that carries a raw string.
    // Each buffer is a distinct allocation so the borrowed values parsed from
    // them can all coexist while the column is assembled.
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(len);
    // For each output row: Some(index into buffers) if it had a raw string, or
    // None (null input row).
    let mut buffer_index: Vec<Option<usize>> = Vec::with_capacity(len);
    // Map a buffer index back to its originating output row index (for error
    // messages and for nulling a row on parse failure).
    let mut buffer_row: Vec<usize> = Vec::with_capacity(len);

    for (row_idx, opt_raw) in values.into_iter().enumerate() {
        match opt_raw {
            None => buffer_index.push(None),
            Some(raw) => {
                let mut buf = Vec::with_capacity(raw.len());
                buf.extend_from_slice(raw.as_bytes());
                buffer_index.push(Some(buffers.len()));
                buffer_row.push(row_idx);
                buffers.push(buf);
            }
        }
    }

    // Parse each buffer once. A parse failure either nulls the row (default) or
    // raises (ErrorMode::Error) with the originating row index.
    let mut parsed: Vec<Option<BorrowedValue>> = Vec::with_capacity(buffers.len());
    for (i, b) in buffers.iter_mut().enumerate() {
        match crate::parse::parse_borrowed_buf(b) {
            Ok(v) => parsed.push(Some(v)),
            Err(e) => match on_error {
                ErrorMode::Error => {
                    let row_idx = buffer_row[i];
                    return Err(PolarsError::ComputeError(
                        format!("fastjson_decode: invalid JSON at row {row_idx}: {e}").into(),
                    ));
                }
                ErrorMode::Null => parsed.push(None),
            },
        }
    }

    // Map each output row to its parsed value (or None for null input / parse
    // failure under ErrorMode::Null).
    let rows: Vec<Option<&BorrowedValue>> = buffer_index
        .iter()
        .map(|opt| opt.and_then(|i| parsed[i].as_ref()))
        .collect();

    // Top-level structural mismatch: when the schema root is a Struct and a row
    // parsed to a valid JSON value that is neither an object nor JSON `null`,
    // that row is a row-level mismatch.
    // Under ErrorMode::Error this raises (consistent with how invalid JSON is treated).
    // Under ErrorMode::Null the struct builder nulls the row (its outer-validity pass).
    if let SchemaType::Struct { .. } = ir {
        if on_error == ErrorMode::Error {
            for (row_idx, opt) in rows.iter().enumerate() {
                match opt {
                    Some(BorrowedValue::Object(_)) | None => {}
                    Some(BorrowedValue::Static(StaticNode::Null)) => {}
                    Some(other) => {
                        let found = json_kind(other);
                        return Err(PolarsError::ComputeError(
                            format!(
                                "fastjson_decode: top-level JSON is {found} at row {row_idx}, \
                                 expected an object for the struct schema"
                            )
                            .into(),
                        ));
                    }
                }
            }
        }
    }

    let effective_rows = if strict_required_fields {
        Some(apply_required_field_policy(
            &rows,
            ir,
            coerce_flag,
            on_error,
        )?)
    } else {
        None
    };
    let rows = effective_rows.as_deref().unwrap_or(&rows);

    let mut series = build_field_series(values.name().clone(), rows, ir, coerce_flag)?;
    series.rename(values.name().clone());
    Ok(series)
}

/// Decode a JSON string column into a `Series` matching `ir`, using
/// [`DecodeOptions`].
pub fn decode_series(
    values: &StringChunked,
    ir: &SchemaType,
    opts: &DecodeOptions,
) -> PolarsResult<Series> {
    decode_rows(
        values,
        ir,
        opts.coerce,
        opts.on_error,
        opts.strict_required_fields,
    )
}

/// Decode with opt-in clustered diagnostics.
///
/// This mirrors [`decode_rows`] but threads row identity and path information
/// through a separate set of builders.
pub fn decode_rows_with_diagnostics(
    values: &StringChunked,
    ir: &SchemaType,
    coerce_flag: bool,
    on_error: ErrorMode,
    strict_required_fields: bool,
    diagnostics_options: DiagnosticsOptions,
) -> PolarsResult<DecodeDiagnostics> {
    let len = values.len();
    let mut diagnostics = DiagnosticsCollector::new(values.name().to_string(), diagnostics_options);

    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(len);
    let mut buffer_index: Vec<Option<usize>> = Vec::with_capacity(len);
    let mut buffer_row: Vec<usize> = Vec::with_capacity(len);

    for (row_idx, opt_raw) in values.into_iter().enumerate() {
        match opt_raw {
            None => buffer_index.push(None),
            Some(raw) => {
                let mut buf = Vec::with_capacity(raw.len());
                buf.extend_from_slice(raw.as_bytes());
                buffer_index.push(Some(buffers.len()));
                buffer_row.push(row_idx);
                buffers.push(buf);
            }
        }
    }

    let mut parsed: Vec<Option<BorrowedValue>> = Vec::with_capacity(buffers.len());
    for (i, b) in buffers.iter_mut().enumerate() {
        match crate::parse::parse_borrowed_buf(b) {
            Ok(v) => parsed.push(Some(v)),
            Err(e) => {
                let row_idx = buffer_row[i];
                let raw = values.get(row_idx).unwrap_or("");
                diagnostics.record_parse_error(row_idx, raw, &e);
                match on_error {
                    ErrorMode::Error => {
                        return Err(PolarsError::ComputeError(
                            format!("fastjson_decode: invalid JSON at row {row_idx}: {e}").into(),
                        ));
                    }
                    ErrorMode::Null => parsed.push(None),
                }
            }
        }
    }

    let rows: Vec<DiagRowValue<'_, '_>> = buffer_index
        .iter()
        .enumerate()
        .map(|(row_idx, opt)| DiagRowValue {
            row_idx,
            value: opt.and_then(|i| parsed[i].as_ref()),
        })
        .collect();

    if let SchemaType::Struct { .. } = ir {
        if on_error == ErrorMode::Error {
            for row in &rows {
                match row.value {
                    Some(BorrowedValue::Object(_)) | None => {}
                    Some(BorrowedValue::Static(StaticNode::Null)) => {}
                    Some(other) => {
                        diagnostics.record_value_mismatch(row.row_idx, "$", "object", other);
                        let found = json_kind(other);
                        return Err(PolarsError::ComputeError(
                            format!(
                                "fastjson_decode: top-level JSON is {found} at row {}, \
                                 expected an object for the struct schema",
                                row.row_idx
                            )
                            .into(),
                        ));
                    }
                }
            }
        }
    }

    let effective_rows = if strict_required_fields {
        Some(apply_required_field_policy_with_diagnostics(
            &rows,
            ir,
            coerce_flag,
            on_error,
            &mut diagnostics,
        )?)
    } else {
        None
    };
    let rows = effective_rows.as_deref().unwrap_or(&rows);

    let mut series = build_field_series_with_diagnostics(
        values.name().clone(),
        rows,
        ir,
        coerce_flag,
        "$",
        &mut diagnostics,
    )?;
    series.rename(values.name().clone());
    let summary = diagnostics.finish();
    Ok(DecodeDiagnostics { series, summary })
}

/// Decode with opt-in diagnostics, using [`DecodeOptions`].
pub fn decode_series_with_diagnostics(
    values: &StringChunked,
    ir: &SchemaType,
    opts: &DecodeOptions,
    diagnostics_options: DiagnosticsOptions,
) -> PolarsResult<DecodeDiagnostics> {
    decode_rows_with_diagnostics(
        values,
        ir,
        opts.coerce,
        opts.on_error,
        opts.strict_required_fields,
        diagnostics_options,
    )
}

fn apply_required_field_policy<'a, 'v>(
    rows: &[Option<&'a BorrowedValue<'v>>],
    ir: &SchemaType,
    coerce_flag: bool,
    on_error: ErrorMode,
) -> PolarsResult<Vec<Option<&'a BorrowedValue<'v>>>> {
    let mut out = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.iter().enumerate() {
        let Some(value) = row else {
            out.push(None);
            continue;
        };
        if let Some(failure) = required::first_required_failure(value, ir, coerce_flag, "$") {
            match on_error {
                ErrorMode::Error => {
                    return Err(PolarsError::ComputeError(
                        format!(
                            "fastjson_decode: required field {} failed at row {row_idx}: {}",
                            failure.path,
                            failure.render()
                        )
                        .into(),
                    ));
                }
                ErrorMode::Null => out.push(None),
            }
        } else {
            out.push(Some(*value));
        }
    }
    Ok(out)
}

fn apply_required_field_policy_with_diagnostics<'a, 'v>(
    rows: &[DiagRowValue<'a, 'v>],
    ir: &SchemaType,
    coerce_flag: bool,
    on_error: ErrorMode,
    diagnostics: &mut DiagnosticsCollector,
) -> PolarsResult<Vec<DiagRowValue<'a, 'v>>> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(value) = row.value else {
            out.push(*row);
            continue;
        };
        if let Some(failure) = required::first_required_failure(value, ir, coerce_flag, "$") {
            diagnostics.record_required_field_failure(
                row.row_idx,
                &failure.path,
                failure.expected,
                failure.found,
                failure.reason,
            );
            match on_error {
                ErrorMode::Error => {
                    return Err(PolarsError::ComputeError(
                        format!(
                            "fastjson_decode: required field {} failed at row {}: {}",
                            failure.path,
                            row.row_idx,
                            failure.render()
                        )
                        .into(),
                    ));
                }
                ErrorMode::Null => out.push(DiagRowValue {
                    row_idx: row.row_idx,
                    value: None,
                }),
            }
        } else {
            out.push(*row);
        }
    }
    Ok(out)
}

/// Human-readable kind name for a parsed JSON value, used in error messages.
fn json_kind(v: &BorrowedValue) -> &'static str {
    match v {
        BorrowedValue::Static(StaticNode::Null) => "null",
        BorrowedValue::Static(StaticNode::Bool(_)) => "a boolean",
        BorrowedValue::Static(StaticNode::I64(_)) | BorrowedValue::Static(StaticNode::U64(_)) => {
            "an integer"
        }
        BorrowedValue::Static(StaticNode::F64(_)) => "a float",
        BorrowedValue::String(_) => "a string",
        BorrowedValue::Array(_) => "an array",
        BorrowedValue::Object(_) => "an object",
    }
}
