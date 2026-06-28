//! Adversarial nested-leniency tests.
//!
//! Verifies that list-of-struct and other nested cases degrade *per-element*
//! and *per-field* rather than nulling siblings or aborting the row/list. Each
//! test asserts the EXACT expected output.
//!
//! Key questions probed:
//!   - `list[Struct]` with `[{}, valid, valid, valid]`: do the valid ones parse?
//!   - non-object elements in a struct-list -> null element vs struct-of-nulls?
//!   - deeply nested struct->list[struct], list[list[i64]], coerce interactions.

use polars_core::prelude::*;
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

fn struct_of(fields: Vec<FieldIR>) -> SchemaType {
    SchemaType::Struct { fields }
}

fn list_of(inner: SchemaType) -> SchemaType {
    SchemaType::List {
        inner: Box::new(inner),
    }
}

fn child(series: &Series, name: &str) -> Series {
    series.struct_().unwrap().field_by_name(name).unwrap()
}

fn opts(on_error: ErrorMode, coerce: bool) -> DecodeOptions {
    DecodeOptions {
        on_error,
        coerce,
        strict_required_fields: false,
    }
}

/// The schema field `s` wrapping a single top-level struct with one field
/// `lst: list[<inner>]`. Returns the decoded inner-list Series for row 0.
fn decode_single_list_field(json: &str, inner: SchemaType, coerce: bool) -> Series {
    let schema = struct_of(vec![field("lst", list_of(inner))]);
    let col = input("raw", &[Some(json)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, coerce)).unwrap();
    assert!(
        !out.is_null().get(0).unwrap(),
        "top struct row should be non-null"
    );
    child(&out, "lst")
}

#[test]
fn case1_empty_struct_then_valid_siblings_parse() {
    let inner = struct_of(vec![
        field("a", SchemaType::I64),
        field("b", SchemaType::Str),
    ]);
    let json = r#"{"lst": [ {}, {"a":1,"b":"x"}, {"a":2,"b":"y"}, {"a":3,"b":"z"} ]}"#;
    let lst = decode_single_list_field(json, inner, true);

    assert!(
        !lst.is_null().get(0).unwrap(),
        "list itself must not be null"
    );

    let elems = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 4, "list length must be 4");

    let a = child(&elems, "a");
    let a = a.i64().unwrap();
    let b = child(&elems, "b");
    let b = b.str().unwrap();

    assert!(!elems.is_null().get(0).unwrap(), "elem 0 struct present");
    assert_eq!(a.get(0), None);
    assert_eq!(b.get(0), None);

    assert_eq!(a.get(1), Some(1));
    assert_eq!(b.get(1), Some("x"));
    assert_eq!(a.get(2), Some(2));
    assert_eq!(b.get(2), Some("y"));
    assert_eq!(a.get(3), Some(3));
    assert_eq!(b.get(3), Some("z"));
}

#[test]
fn case2_heterogeneous_elements_degrade_per_element() {
    let inner = struct_of(vec![
        field("a", SchemaType::I64),
        field("b", SchemaType::Str),
    ]);
    let json = r#"{"lst": [ {"a":1,"b":"x"}, 42, "str", null, [1,2], {"a":5,"b":"q"} ]}"#;
    let lst = decode_single_list_field(json, inner, true);

    assert!(!lst.is_null().get(0).unwrap(), "list must not be null");
    let elems = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 6, "list length must be 6");

    let a = child(&elems, "a");
    let a = a.i64().unwrap();
    let b = child(&elems, "b");
    let b = b.str().unwrap();

    assert!(!elems.is_null().get(0).unwrap());
    assert_eq!(a.get(0), Some(1));
    assert_eq!(b.get(0), Some("x"));

    assert!(
        elems.is_null().get(1).unwrap(),
        "number elem -> null struct"
    );
    assert!(
        elems.is_null().get(2).unwrap(),
        "string elem -> null struct"
    );
    assert!(
        elems.is_null().get(3).unwrap(),
        "json null elem -> null struct"
    );
    assert!(elems.is_null().get(4).unwrap(), "array elem -> null struct");

    assert!(!elems.is_null().get(5).unwrap());
    assert_eq!(a.get(5), Some(5));
    assert_eq!(b.get(5), Some("q"));
}

#[test]
fn case3_partial_structs_missing_fields_null() {
    let inner = struct_of(vec![
        field("a", SchemaType::I64),
        field("b", SchemaType::Str),
    ]);
    let json = r#"{"lst": [ {"a":1}, {"b":"only_b"}, {"a":3,"b":"z"} ]}"#;
    let lst = decode_single_list_field(json, inner, true);

    let elems = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 3);

    let a = child(&elems, "a");
    let a = a.i64().unwrap();
    let b = child(&elems, "b");
    let b = b.str().unwrap();

    assert!(!elems.is_null().get(0).unwrap());
    assert!(!elems.is_null().get(1).unwrap());
    assert!(!elems.is_null().get(2).unwrap());

    assert_eq!(a.get(0), Some(1));
    assert_eq!(b.get(0), None);
    assert_eq!(a.get(1), None);
    assert_eq!(b.get(1), Some("only_b"));
    assert_eq!(a.get(2), Some(3));
    assert_eq!(b.get(2), Some("z"));
}

#[test]
fn case4_struct_with_list_of_structs() {
    let item = struct_of(vec![
        field("id", SchemaType::Str),
        field("qty", SchemaType::I64),
    ]);
    let schema = struct_of(vec![field("items", list_of(item))]);
    let json = r#"{"items":[{"id":"x","qty":2}, {}, {"id":"z"}, "garbage"]}"#;
    let col = input("raw", &[Some(json)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();

    assert!(!out.is_null().get(0).unwrap(), "top struct present");
    let items = child(&out, "items");
    assert!(!items.is_null().get(0).unwrap(), "items list present");

    let elems = items.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 4, "items list length 4");

    let id = child(&elems, "id");
    let id = id.str().unwrap();
    let qty = child(&elems, "qty");
    let qty = qty.i64().unwrap();

    assert!(!elems.is_null().get(0).unwrap());
    assert_eq!(id.get(0), Some("x"));
    assert_eq!(qty.get(0), Some(2));
    assert!(!elems.is_null().get(1).unwrap());
    assert_eq!(id.get(1), None);
    assert_eq!(qty.get(1), None);
    assert!(!elems.is_null().get(2).unwrap());
    assert_eq!(id.get(2), Some("z"));
    assert_eq!(qty.get(2), None);
    assert!(
        elems.is_null().get(3).unwrap(),
        "string elem -> null struct"
    );
}

#[test]
fn case5_list_of_list_i64() {
    let inner = list_of(SchemaType::I64);
    let json = r#"{"lst": [[1,2],["x",3],null,[4]]}"#;
    let lst = decode_single_list_field(json, inner, true);

    assert!(!lst.is_null().get(0).unwrap(), "outer list not null");
    let outer = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(outer.len(), 4, "outer list len 4");

    let outer_list = outer.list().unwrap();

    assert!(!outer.is_null().get(0).unwrap());
    let l0 = outer_list.get_as_series(0).unwrap();
    let l0 = l0.i64().unwrap();
    assert_eq!(l0.len(), 2);
    assert_eq!(l0.get(0), Some(1));
    assert_eq!(l0.get(1), Some(2));

    // [1] = [null, 3]  (bad inner element "x" -> null)
    assert!(!outer.is_null().get(1).unwrap());
    let l1 = outer_list.get_as_series(1).unwrap();
    let l1 = l1.i64().unwrap();
    assert_eq!(l1.len(), 2);
    assert_eq!(l1.get(0), None);
    assert_eq!(l1.get(1), Some(3));

    assert!(
        outer.is_null().get(2).unwrap(),
        "json null inner list -> null"
    );

    assert!(!outer.is_null().get(3).unwrap());
    let l3 = outer_list.get_as_series(3).unwrap();
    let l3 = l3.i64().unwrap();
    assert_eq!(l3.len(), 1);
    assert_eq!(l3.get(0), Some(4));
}

#[test]
fn case6_coerce_true_inside_list_struct() {
    let inner = struct_of(vec![
        field("a", SchemaType::I64),
        field("b", SchemaType::Str),
    ]);
    let json = r#"{"lst": [{"a":"123","b":5}]}"#;
    let lst = decode_single_list_field(json, inner, true);

    let elems = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 1);
    assert!(!elems.is_null().get(0).unwrap());
    assert_eq!(child(&elems, "a").i64().unwrap().get(0), Some(123));
    assert_eq!(child(&elems, "b").str().unwrap().get(0), Some("5"));
}

#[test]
fn case6_coerce_false_inside_list_struct() {
    let inner = struct_of(vec![
        field("a", SchemaType::I64),
        field("b", SchemaType::Str),
    ]);
    let json = r#"{"lst": [{"a":"123","b":5}]}"#;
    let lst = decode_single_list_field(json, inner, false);

    let elems = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 1);
    assert!(!elems.is_null().get(0).unwrap());
    assert_eq!(child(&elems, "a").i64().unwrap().get(0), None);
    assert_eq!(child(&elems, "b").str().unwrap().get(0), None);
}

#[test]
fn case7_whole_field_null_when_not_array() {
    let schema = struct_of(vec![
        field("id", SchemaType::I64),
        field("tags", list_of(SchemaType::Str)),
        field("name", SchemaType::Str),
    ]);
    let json = r#"{"id": 7, "tags": 42, "name": "alice"}"#;
    let col = input("raw", &[Some(json)]);
    let out = decode_series(&col, &schema, &opts(ErrorMode::Null, true)).unwrap();

    assert!(!out.is_null().get(0).unwrap(), "struct row present");

    let tags = child(&out, "tags");
    assert!(
        tags.is_null().get(0).unwrap(),
        "tags (non-array) -> null list"
    );
    assert_eq!(tags.dtype(), &DataType::List(Box::new(DataType::String)));

    assert_eq!(child(&out, "id").i64().unwrap().get(0), Some(7));
    assert_eq!(child(&out, "name").str().unwrap().get(0), Some("alice"));
}

#[test]
fn case8_empty_array_is_empty_list_not_null() {
    let inner = struct_of(vec![
        field("a", SchemaType::I64),
        field("b", SchemaType::Str),
    ]);
    let json = r#"{"lst": []}"#;
    let lst = decode_single_list_field(json, inner.clone(), true);

    assert!(
        !lst.is_null().get(0).unwrap(),
        "empty array -> non-null list"
    );
    let elems = lst.list().unwrap().get_as_series(0).unwrap();
    assert_eq!(elems.len(), 0, "empty list len 0");

    let expected_inner = polars_fastjson::dtype::ir_to_polars(&inner).unwrap();
    assert_eq!(
        lst.dtype(),
        &DataType::List(Box::new(expected_inner)),
        "inner dtype preserved on empty list"
    );
}
