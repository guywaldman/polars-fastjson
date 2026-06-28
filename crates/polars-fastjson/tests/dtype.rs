//! Tests for the IR -> Polars dtype mapping.

use polars_core::prelude::{DataType, Field, TimeUnit as PlTimeUnit};
use polars_fastjson::ir::{FieldIR, SchemaType, TimeUnit};
use polars_fastjson::ir_to_polars;

#[test]
fn scalar_i64_maps_to_int64() {
    assert_eq!(ir_to_polars(&SchemaType::I64).unwrap(), DataType::Int64);
}

#[test]
fn datetime_us_no_tz() {
    let ir = SchemaType::Datetime {
        time_unit: TimeUnit::Us,
        time_zone: None,
    };
    assert_eq!(
        ir_to_polars(&ir).unwrap(),
        DataType::Datetime(PlTimeUnit::Microseconds, None)
    );
}

#[test]
fn struct_with_list_field() {
    let ir = SchemaType::Struct {
        fields: vec![
            FieldIR {
                name: "id".to_string(),
                dtype: SchemaType::Str,
                json_key: None,
                required: false,
            },
            FieldIR {
                name: "score".to_string(),
                dtype: SchemaType::F64,
                json_key: None,
                required: false,
            },
            FieldIR {
                name: "tags".to_string(),
                dtype: SchemaType::List {
                    inner: Box::new(SchemaType::Str),
                },
                json_key: None,
                required: false,
            },
        ],
    };

    let expected = DataType::Struct(vec![
        Field::new("id".into(), DataType::String),
        Field::new("score".into(), DataType::Float64),
        Field::new("tags".into(), DataType::List(Box::new(DataType::String))),
    ]);

    assert_eq!(ir_to_polars(&ir).unwrap(), expected);
}
