from __future__ import annotations

from typing import Union

import polars as pl
import pytest
from typing_extensions import NotRequired, TypedDict

from polars_fastjson import fastjson_decode
from polars_fastjson.schema import normalize


class InnerTD(TypedDict):
    x: int
    y: str


class TotalTD(TypedDict):  # total=True (default): all keys required
    name: str
    age: int
    score: float
    active: bool
    tags: list[str]
    inner: InnerTD


class OptionalTD(TypedDict, total=False):  # all keys optional
    a: str
    b: int


class MixedTD(TypedDict):
    a: str  # required
    b: NotRequired[int]  # explicitly optional


def test_typeddict_total_required_keys():
    ir = normalize(TotalTD)
    req = {f["name"]: f["required"] for f in ir["fields"]}
    assert all(req.values())
    types = {f["name"]: f["dtype"]["type"] for f in ir["fields"]}
    assert types == {
        "name": "str",
        "age": "i64",
        "score": "f64",
        "active": "bool",
        "tags": "list",
        "inner": "struct",
    }
    inner = next(f for f in ir["fields"] if f["name"] == "inner")
    assert inner["dtype"] == {
        "type": "struct",
        "fields": [
            {"name": "x", "dtype": {"type": "i64"}, "required": True},
            {"name": "y", "dtype": {"type": "str"}, "required": True},
        ],
    }


def test_typeddict_total_false_all_optional():
    ir = normalize(OptionalTD)
    req = {f["name"]: f["required"] for f in ir["fields"]}
    assert req == {"a": False, "b": False}


def test_typeddict_not_required_mixed():
    ir = normalize(MixedTD)
    req = {f["name"]: f["required"] for f in ir["fields"]}
    assert req == {"a": True, "b": False}
    # NotRequired[int] must still map to a plain i64 leaf.
    b = next(f for f in ir["fields"] if f["name"] == "b")
    assert b["dtype"] == {"type": "i64"}


EXPECTED_DTYPE = pl.Struct(
    {
        "name": pl.String,
        "age": pl.Int64,
        "score": pl.Float64,
        "active": pl.Boolean,
        "tags": pl.List(pl.String),
        "inner": pl.Struct({"x": pl.Int64, "y": pl.String}),
    }
)

SAMPLE = (
    '{"name": "n", "age": 5, "score": 2.5, "active": true, '
    '"tags": ["t1"], "inner": {"x": 9, "y": "yy"}}'
)
EXPECTED_VALUE = {
    "name": "n",
    "age": 5,
    "score": 2.5,
    "active": True,
    "tags": ["t1"],
    "inner": {"x": 9, "y": "yy"},
}


def test_typeddict_decode_eager():
    df = pl.DataFrame({"payload": [SAMPLE]})
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=TotalTD).alias("parsed")
    )
    assert out["parsed"].dtype == EXPECTED_DTYPE
    assert out["parsed"].to_list() == [EXPECTED_VALUE]


def test_typeddict_decode_lazy_matches_eager():
    df = pl.DataFrame({"payload": [SAMPLE, "not json"]})
    eager = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=TotalTD).alias("parsed")
    )
    lazy = (
        df.lazy()
        .with_columns(
            fastjson_decode(pl.col("payload"), schema=TotalTD).alias("parsed")
        )
        .collect()
    )
    assert lazy["parsed"].dtype == EXPECTED_DTYPE
    assert lazy.equals(eager)
    assert lazy["parsed"].to_list() == [EXPECTED_VALUE, None]


def test_typing_module_typeddict_also_supported():
    # The adapter must accept ``typing.TypedDict`` too, not only
    # ``typing_extensions.TypedDict``.
    import typing

    class PlainTD(typing.TypedDict):
        a: str
        b: int

    ir = normalize(PlainTD)
    assert ir["type"] == "struct"
    assert {f["name"]: f["dtype"]["type"] for f in ir["fields"]} == {
        "a": "str",
        "b": "i64",
    }


def test_typeddict_missing_field_is_null():
    df = pl.DataFrame(
        {"payload": ['{"name": "n", "age": 1, "score": 0.0, "active": false}']}
    )
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=TotalTD).alias("parsed")
    )
    (val,) = out["parsed"].to_list()
    assert val["tags"] is None
    assert val["inner"] is None


def test_typeddict_scalar_union_raises_clear_message():
    class UnionTD(TypedDict):
        a: Union[int, str]

    with pytest.raises(NotImplementedError, match="unions are not supported"):
        normalize(UnionTD)


# Guard: bare object is not a valid schema source.
def test_non_typeddict_rejected():
    with pytest.raises(TypeError):
        normalize(object())
