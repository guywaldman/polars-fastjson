//! Decode tests covering the lenient semantics matrix and the coercion table.

use polars::prelude::*;
use polars_fastjson::ir::{FieldIR, SchemaType};
use polars_fastjson::{decode_series, DecodeOptions, ErrorMode};

fn input(name: &str, rows: &[Option<&str>]) -> StringChunked {
    StringChunked::from_iter_options(name.into(), rows.iter().map(|o| o.map(|s| s.to_string())))
}

fn field(name: &str, dtype: SchemaType) -> FieldIR {
    FieldIR {
        name: name.to_string(),
        dtype,
        json_key: None,
        required: false,
    }
}

fn field_keyed(name: &str, json_key: &str, dtype: SchemaType) -> FieldIR {
    FieldIR {
        name: name.to_string(),
        dtype,
        json_key: Some(json_key.to_string()),
        required: false,
    }
}

fn struct_of(fields: Vec<FieldIR>) -> SchemaType {
    SchemaType::Struct { fields }
}

fn child(series: &Series, name: &str) -> Series {
    series.struct_().unwrap().field_by_name(name).unwrap()
}

fn opts(on_error: ErrorMode, coerce: bool) -> DecodeOptions {
    DecodeOptions { on_error, coerce }
}

#[test]
fn flat_struct_valid_row() {
    let schema = struct_of(vec![
        field("id", SchemaType::I64),
        field("name", SchemaType::Str),
        field("score", SchemaType::F64),
        field("active", SchemaType::Bool),
    ]);
    let col = input(
        "raw",
        &[Some(
            r#"{"id": 7, "name": "alice", "score": 9.5, "active": true}"#,
        )],
    );

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(out.name().as_str(), "raw");
    assert_eq!(out.len(), 1);

    assert_eq!(child(&out, "id").i64().unwrap().get(0), Some(7));
    assert_eq!(child(&out, "name").str().unwrap().get(0), Some("alice"));
    assert_eq!(child(&out, "score").f64().unwrap().get(0), Some(9.5));
    assert_eq!(child(&out, "active").bool().unwrap().get(0), Some(true));
}

#[test]
fn json_key_reads_alias_outputs_field_name() {
    let schema = struct_of(vec![field_keyed(
        "user_id_internal",
        "user_id",
        SchemaType::Str,
    )]);
    let col = input("raw", &[Some(r#"{"user_id": "u1"}"#)]);

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();

    let fields = out.struct_().unwrap().fields_as_series();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].name().as_str(), "user_id_internal");
    assert_eq!(
        child(&out, "user_id_internal").str().unwrap().get(0),
        Some("u1")
    );
}

#[test]
fn json_key_absent_reads_by_field_name() {
    let schema = struct_of(vec![field("user_id", SchemaType::Str)]);
    let col = input("raw", &[Some(r#"{"user_id": "u2"}"#)]);

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(child(&out, "user_id").str().unwrap().get(0), Some("u2"));
}

#[test]
fn json_key_does_not_read_by_field_name() {
    // With a json_key set, the field name must NOT be used as a fallback key:
    // a JSON object carrying only the field name (and not the alias) -> null.
    let schema = struct_of(vec![field_keyed(
        "user_id_internal",
        "user_id",
        SchemaType::Str,
    )]);
    let col = input("raw", &[Some(r#"{"user_id_internal": "u3"}"#)]);

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(child(&out, "user_id_internal").str().unwrap().get(0), None);
}

#[test]
fn invalid_json_nulls_row_in_null_mode() {
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input(
        "raw",
        &[
            Some(r#"{"id": 1}"#),
            Some("{not json"),
            Some(r#"{"id": 3}"#),
        ],
    );

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(out.len(), 3);
    assert!(!out.is_null().get(0).unwrap());
    assert!(out.is_null().get(1).unwrap());
    assert!(!out.is_null().get(2).unwrap());

    let ids = child(&out, "id");
    let ids = ids.i64().unwrap();
    assert_eq!(ids.get(0), Some(1));
    assert_eq!(ids.get(1), None);
    assert_eq!(ids.get(2), Some(3));
}

#[test]
fn invalid_json_raises_in_error_mode() {
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input("raw", &[Some(r#"{"id": 1}"#), Some("{not json")]);

    let res = decode_series(&col, &schema, &opts(ErrorMode::Error, true));
    let err = res.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("row 1"), "error should name the row: {msg}");
}

#[test]
fn missing_field_is_null() {
    let schema = struct_of(vec![
        field("id", SchemaType::I64),
        field("name", SchemaType::Str),
    ]);
    let col = input("raw", &[Some(r#"{"id": 1}"#)]);

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert!(!out.is_null().get(0).unwrap());
    assert_eq!(child(&out, "id").i64().unwrap().get(0), Some(1));
    assert_eq!(child(&out, "name").str().unwrap().get(0), None);
}

#[test]
fn coerce_string_to_int() {
    let schema = struct_of(vec![field("n", SchemaType::I64)]);
    let col = input("raw", &[Some(r#"{"n": "123"}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(child(&out, "n").i64().unwrap().get(0), Some(123));
}

#[test]
fn coerce_int_to_string() {
    let schema = struct_of(vec![field("s", SchemaType::Str)]);
    let col = input("raw", &[Some(r#"{"s": 42}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(child(&out, "s").str().unwrap().get(0), Some("42"));
}

#[test]
fn coerce_string_to_bool() {
    let schema = struct_of(vec![field("b", SchemaType::Bool)]);
    let col = input(
        "raw",
        &[
            Some(r#"{"b": "true"}"#),
            Some(r#"{"b": "FALSE"}"#),
            Some(r#"{"b": "nope"}"#),
        ],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let b = child(&out, "b");
    let b = b.bool().unwrap();
    assert_eq!(b.get(0), Some(true));
    assert_eq!(b.get(1), Some(false));
    assert_eq!(b.get(2), None);
}

#[test]
fn coerce_float_with_fraction_to_int_is_null() {
    let schema = struct_of(vec![field("n", SchemaType::I64)]);
    let col = input("raw", &[Some(r#"{"n": 1.5}"#), Some(r#"{"n": 2.0}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let n = child(&out, "n");
    let n = n.i64().unwrap();
    assert_eq!(n.get(0), None);
    assert_eq!(n.get(1), Some(2));
}

#[test]
fn coerce_int_overflow_is_null() {
    let schema = struct_of(vec![field("n", SchemaType::I8)]);
    let col = input("raw", &[Some(r#"{"n": 5}"#), Some(r#"{"n": 9999}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let n = child(&out, "n");
    let n = n.i8().unwrap();
    assert_eq!(n.get(0), Some(5));
    assert_eq!(n.get(1), None);
}

#[test]
fn coerce_bool_to_int_is_rejected() {
    let schema = struct_of(vec![field("n", SchemaType::I64)]);
    let col = input("raw", &[Some(r#"{"n": true}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert_eq!(child(&out, "n").i64().unwrap().get(0), None);
}

#[test]
fn no_coerce_wrong_type_is_null() {
    let schema = struct_of(vec![
        field("n", SchemaType::I64),
        field("s", SchemaType::Str),
    ]);
    let col = input("raw", &[Some(r#"{"n": "123", "s": 42}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, false)).unwrap();
    assert_eq!(child(&out, "n").i64().unwrap().get(0), None);
    assert_eq!(child(&out, "s").str().unwrap().get(0), None);
}

#[test]
fn no_coerce_exact_match_ok() {
    let schema = struct_of(vec![
        field("n", SchemaType::I64),
        field("s", SchemaType::Str),
    ]);
    let col = input("raw", &[Some(r#"{"n": 123, "s": "hi"}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, false)).unwrap();
    assert_eq!(child(&out, "n").i64().unwrap().get(0), Some(123));
    assert_eq!(child(&out, "s").str().unwrap().get(0), Some("hi"));
}

#[test]
fn extra_field_ignored() {
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input(
        "raw",
        &[Some(r#"{"id": 1, "extra": "dropped", "more": [1,2,3]}"#)],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let fields = out.struct_().unwrap().fields_as_series();
    assert_eq!(fields.len(), 1);
    assert_eq!(child(&out, "id").i64().unwrap().get(0), Some(1));
}

#[test]
fn top_level_non_object_is_null_row() {
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input(
        "raw",
        &[Some("[1, 2, 3]"), Some("42"), Some(r#"{"id": 9}"#)],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert!(out.is_null().get(0).unwrap());
    assert!(out.is_null().get(1).unwrap());
    assert!(!out.is_null().get(2).unwrap());
    assert_eq!(child(&out, "id").i64().unwrap().get(2), Some(9));
}

#[test]
fn top_level_non_object_raises_in_error_mode() {
    // A top-level structural mismatch (valid JSON that is not an object and not
    // JSON null, while the schema root is a struct) RAISES under
    // on_error="error", consistent with how invalid JSON is treated.
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    for raw in [
        "[1, 2, 3]",
        "42",
        "3.14", // float
        r#""foo""#,
        "true",
    ] {
        let col = input("raw", &[Some(raw)]);
        let err = decode_series(&col, &schema, &opts(ErrorMode::Error, true)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("top-level JSON") && msg.contains("row 0"),
            "expected a top-level-mismatch error for {raw:?}, got: {msg}"
        );
    }
}

#[test]
fn top_level_non_object_nulls_row_in_null_mode() {
    // Under on_error="null", a top-level structural mismatch nulls the row (and
    // well-formed object rows still decode).
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input(
        "raw",
        &[
            Some("[1, 2, 3]"),
            Some("42"),
            Some(r#""foo""#),
            Some("true"),
            Some(r#"{"id": 9}"#),
        ],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert!(out.is_null().get(0).unwrap());
    assert!(out.is_null().get(1).unwrap());
    assert!(out.is_null().get(2).unwrap());
    assert!(out.is_null().get(3).unwrap());
    assert!(!out.is_null().get(4).unwrap());
    assert_eq!(child(&out, "id").i64().unwrap().get(4), Some(9));
}

#[test]
fn top_level_json_null_nulls_row_in_both_modes() {
    // A top-level JSON `null` literal is absence, not a mismatch, so it nulls the
    // row in BOTH modes and never raises.
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input("raw", &[Some("null"), Some(r#"{"id": 7}"#)]);

    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert!(out.is_null().get(0).unwrap());
    assert!(!out.is_null().get(1).unwrap());

    let out = decode_series(&col, &schema, &opts(ErrorMode::Error, true)).unwrap();
    assert!(out.is_null().get(0).unwrap());
    assert!(!out.is_null().get(1).unwrap());
    assert_eq!(child(&out, "id").i64().unwrap().get(1), Some(7));
}

#[test]
fn json_null_literal_is_null_field() {
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input("raw", &[Some(r#"{"id": null}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert!(!out.is_null().get(0).unwrap());
    assert_eq!(child(&out, "id").i64().unwrap().get(0), None);
}

#[test]
fn null_input_row_is_null_row() {
    let schema = struct_of(vec![field("id", SchemaType::I64)]);
    let col = input("raw", &[None, Some(r#"{"id": 5}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    assert!(out.is_null().get(0).unwrap());
    assert!(!out.is_null().get(1).unwrap());
}

#[test]
fn nested_struct_and_lists() {
    let schema = struct_of(vec![
        field(
            "user",
            struct_of(vec![
                field("id", SchemaType::I64),
                field("name", SchemaType::Str),
            ]),
        ),
        field(
            "tags",
            SchemaType::List {
                inner: Box::new(SchemaType::Str),
            },
        ),
        field(
            "nums",
            SchemaType::List {
                inner: Box::new(SchemaType::I64),
            },
        ),
    ]);

    let col = input(
        "raw",
        &[Some(
            r#"{"user": {"id": 1, "name": "bob"}, "tags": ["a", "b"], "nums": [1, "x", 3]}"#,
        )],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();

    let user = child(&out, "user");
    assert_eq!(child(&user, "id").i64().unwrap().get(0), Some(1));
    assert_eq!(child(&user, "name").str().unwrap().get(0), Some("bob"));

    let tags = child(&out, "tags");
    let tags0 = tags.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(tags0.str().unwrap().get(0), Some("a"));
    assert_eq!(tags0.str().unwrap().get(1), Some("b"));

    // list<i64> with one bad element ("x") -> null element.
    let nums = child(&out, "nums");
    let nums0 = nums.list().unwrap().get_as_series(0).unwrap();
    let nums0 = nums0.i64().unwrap();
    assert_eq!(nums0.len(), 3);
    assert_eq!(nums0.get(0), Some(1));
    assert_eq!(nums0.get(1), None); // bad element -> null element
    assert_eq!(nums0.get(2), Some(3));
}

#[test]
fn non_array_list_field_is_null() {
    let schema = struct_of(vec![field(
        "tags",
        SchemaType::List {
            inner: Box::new(SchemaType::Str),
        },
    )]);
    let col = input("raw", &[Some(r#"{"tags": "notalist"}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let tags = child(&out, "tags");
    assert!(tags.is_null().get(0).unwrap());
    assert_eq!(tags.dtype(), &DataType::List(Box::new(DataType::String)));
}

#[test]
fn nested_non_object_struct_field_is_null() {
    let schema = struct_of(vec![field(
        "user",
        struct_of(vec![field("id", SchemaType::I64)]),
    )]);
    let col = input("raw", &[Some(r#"{"user": 42}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let user = child(&out, "user");
    assert!(user.is_null().get(0).unwrap());
}

#[test]
fn temporal_date_parse() {
    let schema = struct_of(vec![field("d", SchemaType::Date)]);
    let col = input(
        "raw",
        &[Some(r#"{"d": "1970-01-02"}"#), Some(r#"{"d": "bad"}"#)],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let d = child(&out, "d");
    assert_eq!(d.dtype(), &DataType::Date);
    let phys = d.to_physical_repr();
    let phys = phys.i32().unwrap();
    assert_eq!(phys.get(0), Some(1));
    assert_eq!(phys.get(1), None);
}

#[test]
fn temporal_datetime_parse_utc() {
    use polars_fastjson::ir::TimeUnit;
    let schema = struct_of(vec![field(
        "ts",
        SchemaType::Datetime {
            time_unit: TimeUnit::Us,
            time_zone: None,
        },
    )]);
    let col = input("raw", &[Some(r#"{"ts": "1970-01-01T00:00:01Z"}"#)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let ts = child(&out, "ts");
    assert_eq!(
        ts.dtype(),
        &DataType::Datetime(polars::prelude::TimeUnit::Microseconds, None)
    );
    let phys = ts.to_physical_repr();
    let phys = phys.i64().unwrap();
    assert_eq!(phys.get(0), Some(1_000_000));
}

#[test]
fn temporal_datetime_fractional_seconds() {
    use polars_fastjson::ir::TimeUnit;
    let schema = struct_of(vec![field(
        "ts",
        SchemaType::Datetime {
            time_unit: TimeUnit::Us,
            time_zone: None,
        },
    )]);
    // Fractional seconds are padded to nanosecond resolution, then scaled to us.
    // ".5" -> 500_000_000 ns; ".123456789" truncates beyond 9 digits.
    let col = input(
        "raw",
        &[
            Some(r#"{"ts": "1970-01-01T00:00:01.5Z"}"#),
            Some(r#"{"ts": "1970-01-01T00:00:00.123456789Z"}"#),
        ],
    );
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();
    let ts = child(&out, "ts");
    let phys = ts.to_physical_repr();
    let phys = phys.i64().unwrap();
    assert_eq!(phys.get(0), Some(1_500_000)); // 1.5s in us
    assert_eq!(phys.get(1), Some(123_456)); // 0.123456789s -> 123456 us
}
