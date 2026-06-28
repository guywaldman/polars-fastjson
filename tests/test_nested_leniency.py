"""End-to-end nested-leniency tests against the compiled Rust plugin."""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_fastjson import fastjson_decode


def _decode(payload: str, schema: Any, *, coerce: bool = True) -> Any:
    """Decode a single-row payload and return the parsed value (row 0)."""
    df = pl.DataFrame({"payload": [payload]})
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=schema, coerce=coerce).alias("p")
    )
    return out["p"].to_list()[0]


def test_case1_empty_struct_then_valid_siblings_parse():
    schema = {"lst": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))}
    payload = '{"lst": [ {}, {"a":1,"b":"x"}, {"a":2,"b":"y"}, {"a":3,"b":"z"} ]}'
    row = _decode(payload, schema)

    assert row is not None
    lst = row["lst"]
    assert lst == [
        {"a": None, "b": None},  # empty object -> struct with null fields
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
        {"a": 3, "b": "z"},
    ]


def test_case2_heterogeneous_elements_degrade_per_element():
    schema = {"lst": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))}
    payload = '{"lst": [ {"a":1,"b":"x"}, 42, "str", null, [1,2], {"a":5,"b":"q"} ]}'
    row = _decode(payload, schema)

    assert row is not None
    lst = row["lst"]
    assert lst == [
        {"a": 1, "b": "x"},
        None,  # number -> null struct element
        None,  # string -> null struct element
        None,  # json null -> null struct element
        None,  # array -> null struct element
        {"a": 5, "b": "q"},
    ]


def test_case3_partial_structs_missing_fields_null():
    schema = {"lst": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))}
    payload = '{"lst": [ {"a":1}, {"b":"only_b"}, {"a":3,"b":"z"} ]}'
    row = _decode(payload, schema)

    assert row["lst"] == [
        {"a": 1, "b": None},
        {"a": None, "b": "only_b"},
        {"a": 3, "b": "z"},
    ]


def test_case4_struct_with_list_of_structs():
    schema = {"items": pl.List(pl.Struct({"id": pl.String, "qty": pl.Int64}))}
    payload = '{"items":[{"id":"x","qty":2}, {}, {"id":"z"}, "garbage"]}'
    row = _decode(payload, schema)

    assert row is not None
    assert row["items"] == [
        {"id": "x", "qty": 2},
        {"id": None, "qty": None},  # {} -> struct, null fields
        {"id": "z", "qty": None},  # missing qty -> null
        None,  # "garbage" string -> null struct element
    ]


def test_case5_list_of_list_i64():
    schema = {"lst": pl.List(pl.List(pl.Int64))}
    payload = '{"lst": [[1,2],["x",3],null,[4]]}'
    row = _decode(payload, schema)

    assert row["lst"] == [
        [1, 2],
        [None, 3],  # bad inner element "x" -> null
        None,  # json null inner list -> null
        [4],
    ]


def test_case6_coerce_true_inside_list_struct():
    schema = {"lst": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))}
    payload = '{"lst": [{"a":"123","b":5}]}'
    row = _decode(payload, schema, coerce=True)
    assert row["lst"] == [{"a": 123, "b": "5"}]


def test_case6_coerce_false_inside_list_struct():
    schema = {"lst": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))}
    payload = '{"lst": [{"a":"123","b":5}]}'
    row = _decode(payload, schema, coerce=False)
    # Struct present (it IS an object); fields null under coerce=False.
    assert row["lst"] == [{"a": None, "b": None}]


def test_case7_whole_field_null_when_not_array():
    schema = {"id": pl.Int64, "tags": pl.List(pl.String), "name": pl.String}
    payload = '{"id": 7, "tags": 42, "name": "alice"}'
    row = _decode(payload, schema)

    assert row is not None
    assert row["id"] == 7
    assert row["tags"] is None  # non-array -> whole list null
    assert row["name"] == "alice"  # siblings intact


def test_case8_empty_array_is_empty_list_not_null():
    inner = pl.Struct({"a": pl.Int64, "b": pl.String})
    schema = {"lst": pl.List(inner)}
    payload = '{"lst": []}'
    df = pl.DataFrame({"payload": [payload]})
    out = df.with_columns(fastjson_decode(pl.col("payload"), schema=schema).alias("p"))
    row = out["p"].to_list()[0]

    assert row["lst"] == []  # empty list, NOT null
    # Dtype preserved: List(Struct(...)).
    assert out["p"].dtype == pl.Struct({"lst": pl.List(inner)})
