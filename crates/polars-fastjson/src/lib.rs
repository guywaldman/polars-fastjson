//! # polars-fastjson (core)
//!
//! Performant and safe JSON to `Struct` projection for [Polars](https://pola.rs).
//!
//! Given a JSON string column and a [`SchemaType`], each row is projected into a
//! `Struct` value.
//! Bad JSON, missing fields, wrong leaf types, and structural mismatches produce nulls
//! (or coerced values) rather than aborting the whole column,
//! unless leniency is explicitly disabled via [`ErrorMode::Error`].
//!
//! This crate is the pure-Rust core (IR, dtype mapping, parse boundary, decode, coercion).
//! The Python plugin lives in `polars-fastjson-plugin`.

pub mod decode;
pub mod dtype;
pub mod error;
pub mod ir;
pub mod parse;

pub use dtype::ir_to_polars;
pub use error::ErrorMode;
pub use ir::{FieldIR, SchemaType, TimeUnit};

use polars::prelude::{PolarsResult, Series, StringChunked};

/// Runtime options for a decode pass.
#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    /// How parse failures / top-level structural mismatches are handled.
    pub on_error: ErrorMode,
    /// When `true`, leaf values of a compatible-but-different JSON kind are
    /// converted to the target type (e.g. `"123"` coerces to int).
    /// When `false`, only the exactly-matching JSON kind is accepted, and other kinds become null fields.
    pub coerce: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            on_error: ErrorMode::Null,
            coerce: true,
        }
    }
}

/// Decode a string column of JSON into a `Series` whose dtype is derived from
/// `ir` (a top-level [`SchemaType::Struct`] in the common case).
///
/// This is the core entry point used by the Polars plugin.
pub fn decode_series(
    values: &StringChunked,
    ir: &SchemaType,
    opts: &DecodeOptions,
) -> PolarsResult<Series> {
    decode::decode_series(values, ir, opts)
}
