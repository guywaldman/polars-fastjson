//! Scalar leaf decoding.
//!
//! Given the per-row JSON values already navigated to a leaf field, produce a
//! typed Polars `Series` of the correct dtype and length, applying the coercion
//! rules from [`super::coerce`].
//!
//! Temporal leaves build their physical integer column and then cast to the
//! logical dtype, guaranteeing agreement with [`crate::dtype::ir_to_polars`].

use polars_core::prelude::*;

use super::coerce;
use crate::diagnostics::{expected_type, DiagRowValue, DiagRowValues, DiagnosticsCollector};
use crate::ir::{SchemaType, TimeUnit as IrTimeUnit};

/// One navigated JSON value per output row.
///
/// `None` means "this row has no value here" (null input row, JSON `null`,
/// missing field, or a parent that was itself null/mismatched). Such rows always
/// produce a null leaf regardless of coercion.
pub type RowValues<'a, 'v> = &'a [Option<&'a simd_json::BorrowedValue<'v>>];

/// Map an IR [`IrTimeUnit`] to a Polars [`TimeUnit`].
fn pl_tu(tu: IrTimeUnit) -> TimeUnit {
    match tu {
        IrTimeUnit::Us => TimeUnit::Microseconds,
        IrTimeUnit::Ms => TimeUnit::Milliseconds,
        IrTimeUnit::Ns => TimeUnit::Nanoseconds,
    }
}

/// Build an integer `Series` of native type `T` from a per-row coercion closure.
fn build_int<T>(name: PlSmallStr, rows: RowValues, coerce_flag: bool) -> Series
where
    T: PolarsNumericType,
    T::Native: TryFrom<i128>,
{
    let it = rows
        .iter()
        .map(|opt| opt.and_then(|v| coerce::coerce_int::<T::Native>(v, coerce_flag)));
    ChunkedArray::<T>::from_iter_options(name, it).into_series()
}

fn build_int_with_diagnostics<T>(
    name: PlSmallStr,
    rows: DiagRowValues,
    ir: &SchemaType,
    coerce_flag: bool,
    path: &str,
    diagnostics: &mut DiagnosticsCollector,
) -> Series
where
    T: PolarsNumericType,
    T::Native: TryFrom<i128>,
{
    let expected = expected_type(ir);
    let it = rows.iter().map(|row| match row.value {
        Some(value) => match coerce::coerce_int::<T::Native>(value, coerce_flag) {
            Some(v) => Some(v),
            None => {
                diagnostics.record_value_mismatch(row.row_idx, path, expected, value);
                None
            }
        },
        None => None,
    });
    ChunkedArray::<T>::from_iter_options(name, it).into_series()
}

fn record_scalar_failure(
    row: &DiagRowValue<'_, '_>,
    path: &str,
    ir: &SchemaType,
    diagnostics: &mut DiagnosticsCollector,
) {
    if let Some(value) = row.value {
        diagnostics.record_value_mismatch(row.row_idx, path, expected_type(ir), value);
    }
}

/// Build a `Series` for the given scalar `ir`, navigating each row's value.
///
/// `name` is the field/output name; `rows` holds one optional JSON value per
/// output row. The returned `Series` has the dtype mandated by `ir` and length
/// `rows.len()`.
pub fn build_scalar_series(
    name: PlSmallStr,
    rows: RowValues,
    ir: &SchemaType,
    coerce_flag: bool,
) -> PolarsResult<Series> {
    let len = rows.len();
    let s = match ir {
        SchemaType::Null => Series::full_null(name, len, &DataType::Null),

        SchemaType::Bool => {
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_bool(v, coerce_flag)));
            BooleanChunked::from_iter_options(name, it).into_series()
        }

        SchemaType::I8 => build_int::<Int8Type>(name, rows, coerce_flag),
        SchemaType::I16 => build_int::<Int16Type>(name, rows, coerce_flag),
        SchemaType::I32 => build_int::<Int32Type>(name, rows, coerce_flag),
        SchemaType::I64 => build_int::<Int64Type>(name, rows, coerce_flag),
        SchemaType::U8 => build_int::<UInt8Type>(name, rows, coerce_flag),
        SchemaType::U16 => build_int::<UInt16Type>(name, rows, coerce_flag),
        SchemaType::U32 => build_int::<UInt32Type>(name, rows, coerce_flag),
        SchemaType::U64 => build_int::<UInt64Type>(name, rows, coerce_flag),

        SchemaType::F32 => {
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_f32(v, coerce_flag)));
            Float32Chunked::from_iter_options(name, it).into_series()
        }
        SchemaType::F64 => {
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_f64(v, coerce_flag)));
            Float64Chunked::from_iter_options(name, it).into_series()
        }

        SchemaType::Str => {
            let strings: Vec<Option<String>> = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_str(v, coerce_flag)))
                .collect();
            StringChunked::from_iter_options(name, strings.into_iter()).into_series()
        }

        SchemaType::Binary => {
            let bytes: Vec<Option<Vec<u8>>> = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_binary(v, coerce_flag)))
                .collect();
            BinaryChunked::from_iter_options(name, bytes.into_iter()).into_series()
        }

        SchemaType::Date => {
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_date(v, coerce_flag)));
            let phys = Int32Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Date)?
        }

        SchemaType::Time => {
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_time(v, coerce_flag)));
            let phys = Int64Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Time)?
        }

        SchemaType::Datetime {
            time_unit,
            time_zone,
        } => {
            let tu = *time_unit;
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_datetime(v, tu, coerce_flag)));
            let phys = Int64Chunked::from_iter_options(name, it).into_series();
            let tz = TimeZone::opt_try_new(time_zone.as_deref())?;
            phys.cast(&DataType::Datetime(pl_tu(tu), tz))?
        }

        SchemaType::Duration { time_unit } => {
            let tu = *time_unit;
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_duration(v, tu, coerce_flag)));
            let phys = Int64Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Duration(pl_tu(tu)))?
        }

        SchemaType::Decimal { precision, scale } => {
            // v1: decimals are sourced from a JSON float/int/string and cast to
            // the declared Decimal dtype. Values that cannot be coerced to f64
            // become null.
            let it = rows
                .iter()
                .map(|opt| opt.and_then(|v| coerce::coerce_f64(v, coerce_flag)));
            let phys = Float64Chunked::from_iter_options(name, it).into_series();
            // `None` precision defaults to 38 (polars max); see dtype.rs.
            phys.cast(&DataType::Decimal(precision.unwrap_or(38), *scale))?
        }

        SchemaType::List { .. } | SchemaType::Struct { .. } => {
            unreachable!("list/struct are handled by their dedicated decoders")
        }
    };
    Ok(s)
}

/// Diagnostic variant of [`build_scalar_series`].
pub(crate) fn build_scalar_series_with_diagnostics(
    name: PlSmallStr,
    rows: DiagRowValues,
    ir: &SchemaType,
    coerce_flag: bool,
    path: &str,
    diagnostics: &mut DiagnosticsCollector,
) -> PolarsResult<Series> {
    let len = rows.len();
    let s = match ir {
        SchemaType::Null => {
            for row in rows {
                record_scalar_failure(row, path, ir, diagnostics);
            }
            Series::full_null(name, len, &DataType::Null)
        }

        SchemaType::Bool => {
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_bool(value, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            BooleanChunked::from_iter_options(name, it).into_series()
        }

        SchemaType::I8 => {
            build_int_with_diagnostics::<Int8Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::I16 => {
            build_int_with_diagnostics::<Int16Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::I32 => {
            build_int_with_diagnostics::<Int32Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::I64 => {
            build_int_with_diagnostics::<Int64Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::U8 => {
            build_int_with_diagnostics::<UInt8Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::U16 => {
            build_int_with_diagnostics::<UInt16Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::U32 => {
            build_int_with_diagnostics::<UInt32Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }
        SchemaType::U64 => {
            build_int_with_diagnostics::<UInt64Type>(name, rows, ir, coerce_flag, path, diagnostics)
        }

        SchemaType::F32 => {
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_f32(value, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            Float32Chunked::from_iter_options(name, it).into_series()
        }
        SchemaType::F64 => {
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_f64(value, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            Float64Chunked::from_iter_options(name, it).into_series()
        }

        SchemaType::Str => {
            let strings: Vec<Option<String>> = rows
                .iter()
                .map(|row| match row.value {
                    Some(value) => match coerce::coerce_str(value, coerce_flag) {
                        Some(v) => Some(v),
                        None => {
                            record_scalar_failure(row, path, ir, diagnostics);
                            None
                        }
                    },
                    None => None,
                })
                .collect();
            StringChunked::from_iter_options(name, strings.into_iter()).into_series()
        }

        SchemaType::Binary => {
            let bytes: Vec<Option<Vec<u8>>> = rows
                .iter()
                .map(|row| match row.value {
                    Some(value) => match coerce::coerce_binary(value, coerce_flag) {
                        Some(v) => Some(v),
                        None => {
                            record_scalar_failure(row, path, ir, diagnostics);
                            None
                        }
                    },
                    None => None,
                })
                .collect();
            BinaryChunked::from_iter_options(name, bytes.into_iter()).into_series()
        }

        SchemaType::Date => {
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_date(value, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            let phys = Int32Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Date)?
        }

        SchemaType::Time => {
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_time(value, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            let phys = Int64Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Time)?
        }

        SchemaType::Datetime {
            time_unit,
            time_zone,
        } => {
            let tu = *time_unit;
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_datetime(value, tu, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            let phys = Int64Chunked::from_iter_options(name, it).into_series();
            let tz = TimeZone::opt_try_new(time_zone.as_deref())?;
            phys.cast(&DataType::Datetime(pl_tu(tu), tz))?
        }

        SchemaType::Duration { time_unit } => {
            let tu = *time_unit;
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_duration(value, tu, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            let phys = Int64Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Duration(pl_tu(tu)))?
        }

        SchemaType::Decimal { precision, scale } => {
            let it = rows.iter().map(|row| match row.value {
                Some(value) => match coerce::coerce_f64(value, coerce_flag) {
                    Some(v) => Some(v),
                    None => {
                        record_scalar_failure(row, path, ir, diagnostics);
                        None
                    }
                },
                None => None,
            });
            let phys = Float64Chunked::from_iter_options(name, it).into_series();
            phys.cast(&DataType::Decimal(precision.unwrap_or(38), *scale))?
        }

        SchemaType::List { .. } | SchemaType::Struct { .. } => {
            unreachable!("list/struct are handled by their dedicated decoders")
        }
    };
    Ok(s)
}
