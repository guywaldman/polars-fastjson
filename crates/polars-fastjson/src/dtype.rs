//! IR <> Polars/Arrow dtype mapping.
//!
//! The same `SchemaType` drives both the declared output dtype (plan-time callback)
//! and the runtime builders, guaranteeing they are always in-sync.

use polars::prelude::{DataType, Field, PolarsResult, TimeUnit as PlTimeUnit, TimeZone};

use crate::ir::{FieldIR, SchemaType, TimeUnit};

fn map_time_unit(tu: TimeUnit) -> PlTimeUnit {
    match tu {
        TimeUnit::Us => PlTimeUnit::Microseconds,
        TimeUnit::Ms => PlTimeUnit::Milliseconds,
        TimeUnit::Ns => PlTimeUnit::Nanoseconds,
    }
}

/// Map a `SchemaType` to its corresponding Polars `DataType`.
pub fn ir_to_polars(ir: &SchemaType) -> PolarsResult<DataType> {
    let dt = match ir {
        SchemaType::Null => DataType::Null,
        SchemaType::Bool => DataType::Boolean,
        SchemaType::I8 => DataType::Int8,
        SchemaType::I16 => DataType::Int16,
        SchemaType::I32 => DataType::Int32,
        SchemaType::I64 => DataType::Int64,
        SchemaType::U8 => DataType::UInt8,
        SchemaType::U16 => DataType::UInt16,
        SchemaType::U32 => DataType::UInt32,
        SchemaType::U64 => DataType::UInt64,
        SchemaType::F32 => DataType::Float32,
        SchemaType::F64 => DataType::Float64,
        SchemaType::Str => DataType::String,
        SchemaType::Binary => DataType::Binary,
        SchemaType::Date => DataType::Date,
        SchemaType::Time => DataType::Time,
        SchemaType::Datetime {
            time_unit,
            time_zone,
        } => {
            let tz = TimeZone::opt_try_new(time_zone.as_deref())?;
            DataType::Datetime(map_time_unit(*time_unit), tz)
        }
        SchemaType::Duration { time_unit } => DataType::Duration(map_time_unit(*time_unit)),
        // polars 0.52's `Decimal(usize, usize)` requires a concrete precision
        // (invariant: 1 <= precision <= 38).
        // The IR keeps `precision: Option<usize>` because JSON may omit it.
        // A `None` precision defaults to 38 (polars' maximum).
        SchemaType::Decimal { precision, scale } => {
            DataType::Decimal(precision.unwrap_or(38), *scale)
        }
        SchemaType::List { inner } => DataType::List(Box::new(ir_to_polars(inner)?)),
        SchemaType::Struct { fields } => {
            let mut pl_fields: Vec<Field> = Vec::with_capacity(fields.len());
            for FieldIR { name, dtype, .. } in fields {
                pl_fields.push(Field::new(name.as_str().into(), ir_to_polars(dtype)?));
            }
            DataType::Struct(pl_fields)
        }
    };
    Ok(dt)
}
