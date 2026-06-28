//! Thin Polars expression plugin (cdylib) wiring `polars-fastjson` core into the
//! Polars runtime via the pyo3-polars plugin ABI.
//!
//! The plugin imports NO Python schema libraries: Python normalizes every schema
//! source into a JSON-serializable `SchemaIR`, passed across the boundary as the
//! `schema_ir` kwarg.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use polars_fastjson::{decode_series, ir_to_polars, DecodeOptions, ErrorMode, SchemaType};

/// Kwargs forwarded from the Python `register_plugin_function` call.
///
/// `schema_ir` is the serde-tagged `SchemaType` produced by the Python adapters.
#[derive(Deserialize)]
struct FastjsonKwargs {
    schema_ir: SchemaType,
    on_error: String,
    coerce: bool,
    // Reserved for `extra="ignore"|"capture"`; only "ignore" is honored in v1.
    #[allow(dead_code)]
    extra: String,
}

fn parse_on_error(s: &str) -> PolarsResult<ErrorMode> {
    match s {
        "null" => Ok(ErrorMode::Null),
        "error" => Ok(ErrorMode::Error),
        other => Err(PolarsError::ComputeError(
            format!("invalid on_error: {other:?} (expected \"null\" or \"error\")").into(),
        )),
    }
}

/// Plan-time output dtype: the declared `Struct` (or other) dtype derived from
/// the SchemaIR, carrying the input column's name.
//
// verify: exact `output_type_func_with_kwargs` callback signature on first
// cargo fetch against the installed pyo3-polars version.
fn fastjson_decode_output(input_fields: &[Field], kwargs: FastjsonKwargs) -> PolarsResult<Field> {
    let name = input_fields
        .first()
        .map(|f| f.name().clone())
        .unwrap_or_else(|| "fastjson".into());
    let dtype = ir_to_polars(&kwargs.schema_ir)?;
    Ok(Field::new(name, dtype))
}

/// Lenient, schema-aware JSON -> Struct projection expression.
//
// verify: exact `#[polars_expr]` macro attribute spelling on first cargo fetch.
#[polars_expr(output_type_func_with_kwargs = fastjson_decode_output)]
fn fastjson_decode(inputs: &[Series], kwargs: FastjsonKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let values = s.str()?;

    let opts = DecodeOptions {
        on_error: parse_on_error(&kwargs.on_error)?,
        coerce: kwargs.coerce,
    };

    decode_series(values, &kwargs.schema_ir, &opts)
}
