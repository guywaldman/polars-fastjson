//! Thin Polars expression plugin (cdylib) wiring `polars-fastjson` core into the
//! Polars runtime via the pyo3-polars plugin ABI.
//!
//! The plugin imports NO Python schema libraries: Python normalizes every schema
//! source into a JSON-serializable `SchemaIR`, passed across the boundary as the
//! `schema_ir` kwarg.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use polars_fastjson::{
    decode_series, decode_series_with_diagnostics, ir_to_polars, DecodeOptions, DiagnosticsMode,
    DiagnosticsOptions, DiagnosticsSummary, ErrorMode, SchemaType,
};

/// Kwargs forwarded from the Python `register_plugin_function` call.
///
/// `schema_ir` is the serde-tagged `SchemaType` produced by the Python adapters.
#[derive(Deserialize)]
struct FastjsonKwargs {
    schema_ir: SchemaType,
    on_error: String,
    coerce: bool,
    #[serde(default = "default_diagnostics")]
    diagnostics: String,
    // Reserved for `extra="ignore"|"capture"`; only "ignore" is honored in v1.
    #[allow(dead_code)]
    extra: String,
}

fn default_diagnostics() -> String {
    "off".to_string()
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

fn parse_diagnostics(s: &str) -> PolarsResult<DiagnosticsMode> {
    match s {
        "off" => Ok(DiagnosticsMode::Off),
        "summary" => Ok(DiagnosticsMode::Summary),
        other => Err(PolarsError::ComputeError(
            format!("invalid diagnostics: {other:?} (expected \"off\" or \"summary\")").into(),
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

    match parse_diagnostics(&kwargs.diagnostics)? {
        DiagnosticsMode::Off => decode_series(values, &kwargs.schema_ir, &opts),
        DiagnosticsMode::Summary if !diagnostics_logger_enabled() => {
            decode_series(values, &kwargs.schema_ir, &opts)
        }
        DiagnosticsMode::Summary => {
            let ids = match inputs.get(1) {
                Some(id_series) => Some(series_to_id_strings(id_series)?),
                None => None,
            };
            let decoded = decode_series_with_diagnostics(
                values,
                &kwargs.schema_ir,
                &opts,
                DiagnosticsOptions { ids },
            )?;
            if !decoded.summary.is_empty() {
                emit_diagnostics(&decoded.summary)?;
            }
            Ok(decoded.series)
        }
    }
}

fn series_to_id_strings(series: &Series) -> PolarsResult<Vec<Option<String>>> {
    let mut ids = Vec::with_capacity(series.len());
    for idx in 0..series.len() {
        let value = series.get(idx)?;
        let id = match value {
            AnyValue::Null => None,
            AnyValue::String(value) => Some(value.to_string()),
            AnyValue::StringOwned(value) => Some(value.to_string()),
            other => Some(other.to_string()),
        };
        ids.push(id);
    }
    Ok(ids)
}

fn diagnostics_logger_enabled() -> bool {
    Python::attach(|py| -> PyResult<bool> {
        let logging = py.import("logging")?;
        let logger = logging.call_method1("getLogger", ("polars_fastjson.diagnostics",))?;
        let warning = logging.getattr("WARNING")?;
        logger.call_method1("isEnabledFor", (warning,))?.extract()
    })
    .unwrap_or(false)
}

fn emit_diagnostics(summary: &DiagnosticsSummary) -> PolarsResult<()> {
    Python::attach(|py| -> PyResult<()> {
        let logging = py.import("logging")?;
        let logger = logging.call_method1("getLogger", ("polars_fastjson.diagnostics",))?;
        let kwargs = PyDict::new(py);
        let extra = PyDict::new(py);
        extra.set_item("fastjson_diagnostics", summary_to_py(py, summary)?)?;
        kwargs.set_item("extra", extra)?;
        logger.call_method("warning", (summary.render(),), Some(&kwargs))?;
        Ok(())
    })
    .map_err(|err| {
        PolarsError::ComputeError(
            format!("fastjson_decode: failed to emit diagnostics log: {err}").into(),
        )
    })
}

fn summary_to_py<'py>(
    py: Python<'py>,
    summary: &DiagnosticsSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("column", &summary.column)?;
    payload.set_item("issues", summary.issues)?;

    let clusters = PyList::empty(py);
    for cluster in &summary.clusters {
        let item = PyDict::new(py);
        item.set_item("kind", &cluster.kind)?;
        item.set_item("path", &cluster.path)?;
        item.set_item("expected", &cluster.expected)?;
        item.set_item("found", &cluster.found)?;
        item.set_item("reason", &cluster.reason)?;
        item.set_item("count", cluster.count)?;
        item.set_item("samples", PyList::new(py, &cluster.samples)?)?;
        item.set_item("omitted_samples", cluster.omitted_samples)?;
        item.set_item("ids", PyList::new(py, &cluster.ids)?)?;
        item.set_item("omitted_ids", cluster.omitted_ids)?;
        clusters.append(item)?;
    }
    payload.set_item("clusters", clusters)?;

    Ok(payload)
}
