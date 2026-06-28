//! The `SchemaIR` grammar shared verbatim between Python (producer) and Rust
//! (consumer).
//!
//! It is an internally tagged serde enum: `{"type": "<tag>", ...}`.

use serde::{Deserialize, Serialize};

/// Temporal resolution. Serializes to `"us"` / `"ms"` / `"ns"`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimeUnit {
    Us,
    Ms,
    Ns,
}

/// The schema intermediate representation.
///
/// Internally tagged on `type`, with `snake_case` tags.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SchemaType {
    Null,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Str,
    Binary,
    Date,
    Time,
    Datetime {
        time_unit: TimeUnit,
        #[serde(default)]
        time_zone: Option<String>,
    },
    Duration {
        time_unit: TimeUnit,
    },
    Decimal {
        #[serde(default)]
        precision: Option<usize>,
        scale: usize,
    },
    List {
        inner: Box<SchemaType>,
    },
    Struct {
        fields: Vec<FieldIR>,
    },
}

/// A single named field within a struct.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FieldIR {
    pub name: String,
    pub dtype: SchemaType,
    /// JSON key to read this field from. Defaults to `name` when absent. The
    /// output struct field is always named `name`; `json_key` only affects which
    /// key is read from the input JSON object (e.g. a pydantic validation alias).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_key: Option<String>,
    /// Metadata for future diagnostics.
    #[serde(default)]
    pub required: bool,
}
