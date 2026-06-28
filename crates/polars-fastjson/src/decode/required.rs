//! Strict required-field validation.
//!
//! This is only used when the caller opts into row-level required-field
//! failures. The default lenient decode path does not walk this validator.

use simd_json::{BorrowedValue, StaticNode};

use super::coerce;
use crate::diagnostics::{expected_type, json_kind};
use crate::ir::{FieldIR, SchemaType};

#[derive(Debug, Clone)]
pub(crate) struct RequiredFailure {
    pub path: String,
    pub expected: &'static str,
    pub found: Option<&'static str>,
    pub reason: &'static str,
}

impl RequiredFailure {
    fn missing(path: String, expected: &'static str) -> Self {
        Self {
            path,
            expected,
            found: None,
            reason: "missing required field",
        }
    }

    fn null(path: String, expected: &'static str) -> Self {
        Self {
            path,
            expected,
            found: Some("null"),
            reason: "required field is null",
        }
    }

    fn mismatch(path: String, expected: &'static str, found: &'static str) -> Self {
        Self {
            path,
            expected,
            found: Some(found),
            reason: "required field failed to decode",
        }
    }

    pub fn render(&self) -> String {
        match self.found {
            Some(found) => format!(
                "{} (expected {}, found {})",
                self.reason, self.expected, found
            ),
            None => format!("{} (expected {})", self.reason, self.expected),
        }
    }
}

pub(crate) fn first_required_failure(
    value: &BorrowedValue,
    ir: &SchemaType,
    coerce_flag: bool,
    path: &str,
) -> Option<RequiredFailure> {
    match ir {
        SchemaType::Struct { fields } => required_struct_failure(value, fields, coerce_flag, path),
        _ => required_value_failure(Some(value), ir, coerce_flag, path),
    }
}

fn required_struct_failure(
    value: &BorrowedValue,
    fields: &[FieldIR],
    coerce_flag: bool,
    path: &str,
) -> Option<RequiredFailure> {
    let map = match value {
        BorrowedValue::Object(map) => map,
        BorrowedValue::Static(StaticNode::Null) => {
            return Some(RequiredFailure::null(
                path.to_string(),
                expected_type(&SchemaType::Struct { fields: vec![] }),
            ));
        }
        other => {
            return Some(RequiredFailure::mismatch(
                path.to_string(),
                expected_type(&SchemaType::Struct { fields: vec![] }),
                json_kind(other),
            ));
        }
    };

    for field in fields {
        if !field.required {
            continue;
        }

        let key = field.json_key.as_deref().unwrap_or(field.name.as_str());
        let field_path = format!("{path}.{}", field.name);
        let Some(child) = map.get(key) else {
            return Some(RequiredFailure::missing(
                field_path,
                expected_type(&field.dtype),
            ));
        };
        if let Some(failure) =
            required_value_failure(Some(child), &field.dtype, coerce_flag, &field_path)
        {
            return Some(failure);
        }
    }

    None
}

fn required_value_failure(
    value: Option<&BorrowedValue>,
    ir: &SchemaType,
    coerce_flag: bool,
    path: &str,
) -> Option<RequiredFailure> {
    let expected = expected_type(ir);
    let Some(value) = value else {
        return Some(RequiredFailure::missing(path.to_string(), expected));
    };

    if matches!(value, BorrowedValue::Static(StaticNode::Null)) {
        return match ir {
            SchemaType::Null => None,
            _ => Some(RequiredFailure::null(path.to_string(), expected)),
        };
    }

    match ir {
        SchemaType::Struct { fields } => required_struct_failure(value, fields, coerce_flag, path),
        SchemaType::List { .. } => match value {
            BorrowedValue::Array(_) => None,
            other => Some(RequiredFailure::mismatch(
                path.to_string(),
                expected,
                json_kind(other),
            )),
        },
        _ if scalar_decodes(value, ir, coerce_flag) => None,
        _ => Some(RequiredFailure::mismatch(
            path.to_string(),
            expected,
            json_kind(value),
        )),
    }
}

fn scalar_decodes(value: &BorrowedValue, ir: &SchemaType, coerce_flag: bool) -> bool {
    match ir {
        SchemaType::Null => matches!(value, BorrowedValue::Static(StaticNode::Null)),
        SchemaType::Bool => coerce::coerce_bool(value, coerce_flag).is_some(),
        SchemaType::I8 => coerce::coerce_int::<i8>(value, coerce_flag).is_some(),
        SchemaType::I16 => coerce::coerce_int::<i16>(value, coerce_flag).is_some(),
        SchemaType::I32 => coerce::coerce_int::<i32>(value, coerce_flag).is_some(),
        SchemaType::I64 => coerce::coerce_int::<i64>(value, coerce_flag).is_some(),
        SchemaType::U8 => coerce::coerce_int::<u8>(value, coerce_flag).is_some(),
        SchemaType::U16 => coerce::coerce_int::<u16>(value, coerce_flag).is_some(),
        SchemaType::U32 => coerce::coerce_int::<u32>(value, coerce_flag).is_some(),
        SchemaType::U64 => coerce::coerce_int::<u64>(value, coerce_flag).is_some(),
        SchemaType::F32 => coerce::coerce_f32(value, coerce_flag).is_some(),
        SchemaType::F64 => coerce::coerce_f64(value, coerce_flag).is_some(),
        SchemaType::Str => coerce::coerce_str(value, coerce_flag).is_some(),
        SchemaType::Binary => coerce::coerce_binary(value, coerce_flag).is_some(),
        SchemaType::Date => coerce::coerce_date(value, coerce_flag).is_some(),
        SchemaType::Time => coerce::coerce_time(value, coerce_flag).is_some(),
        SchemaType::Datetime { time_unit, .. } => {
            coerce::coerce_datetime(value, *time_unit, coerce_flag).is_some()
        }
        SchemaType::Duration { time_unit } => {
            coerce::coerce_duration(value, *time_unit, coerce_flag).is_some()
        }
        SchemaType::Decimal { .. } => coerce::coerce_f64(value, coerce_flag).is_some(),
        SchemaType::List { .. } | SchemaType::Struct { .. } => unreachable!(),
    }
}
