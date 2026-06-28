from __future__ import annotations

import polars as pl

from polars_fastjson.schema import normalize


def test_normalize_dict_schema():
    schema = {"id": pl.String, "score": pl.Float64, "tags": pl.List(pl.String)}
    ir = normalize(schema)

    assert ir == {
        "type": "struct",
        "fields": [
            {"name": "id", "dtype": {"type": "str"}, "required": False},
            {"name": "score", "dtype": {"type": "f64"}, "required": False},
            {
                "name": "tags",
                "dtype": {"type": "list", "inner": {"type": "str"}},
                "required": False,
            },
        ],
    }


def test_normalize_scalar_types():
    ir = normalize({"a": pl.Int64, "b": pl.Boolean, "c": pl.UInt32})
    tags = {f["name"]: f["dtype"]["type"] for f in ir["fields"]}
    assert tags == {"a": "i64", "b": "bool", "c": "u32"}


def test_normalize_datetime_us_no_tz():
    ir = normalize({"ts": pl.Datetime("us")})
    (field,) = ir["fields"]
    assert field["dtype"] == {"type": "datetime", "time_unit": "us", "time_zone": None}


def test_normalize_struct_dtype():
    dt = pl.Struct({"x": pl.Int32, "y": pl.String})
    ir = normalize(dt)
    assert ir["type"] == "struct"
    tags = {f["name"]: f["dtype"]["type"] for f in ir["fields"]}
    assert tags == {"x": "i32", "y": "str"}


def test_normalize_nested_struct_and_list():
    dt = pl.Struct(
        {
            "id": pl.String,
            "items": pl.List(pl.Int64),
            "meta": pl.Struct({"a": pl.Boolean, "b": pl.Float32}),
        }
    )
    ir = normalize(dt)
    assert ir == {
        "type": "struct",
        "fields": [
            {"name": "id", "dtype": {"type": "str"}, "required": False},
            {
                "name": "items",
                "dtype": {"type": "list", "inner": {"type": "i64"}},
                "required": False,
            },
            {
                "name": "meta",
                "dtype": {
                    "type": "struct",
                    "fields": [
                        {"name": "a", "dtype": {"type": "bool"}, "required": False},
                        {"name": "b", "dtype": {"type": "f32"}, "required": False},
                    ],
                },
                "required": False,
            },
        ],
    }


def test_normalize_list_of_struct():
    dt = pl.List(pl.Struct({"x": pl.Int32}))
    ir = normalize(dt)
    assert ir == {
        "type": "list",
        "inner": {
            "type": "struct",
            "fields": [{"name": "x", "dtype": {"type": "i32"}, "required": False}],
        },
    }


def test_normalize_temporal_dtypes():
    ir = normalize(
        {
            "d": pl.Date,
            "t": pl.Time,
            "ts_ms": pl.Datetime("ms"),
            "ts_tz": pl.Datetime("ns", "UTC"),
            "dur": pl.Duration("us"),
        }
    )
    by_name = {f["name"]: f["dtype"] for f in ir["fields"]}
    assert by_name["d"] == {"type": "date"}
    assert by_name["t"] == {"type": "time"}
    assert by_name["ts_ms"] == {
        "type": "datetime",
        "time_unit": "ms",
        "time_zone": None,
    }
    assert by_name["ts_tz"] == {
        "type": "datetime",
        "time_unit": "ns",
        "time_zone": "UTC",
    }
    assert by_name["dur"] == {"type": "duration", "time_unit": "us"}


def test_normalize_unknown_raises():
    import pytest

    with pytest.raises(TypeError):
        normalize(object())
