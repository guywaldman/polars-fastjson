from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import polars as pl
import pytest

from polars_fastjson import fastjson_decode
from polars_fastjson.schema import normalize


@dataclass
class Inner:
    x: int
    y: str


@dataclass
class Person:
    name: str
    age: int
    score: float
    active: bool
    nickname: Optional[int]  # Optional[X] -> X (nullable implicit)
    tags: list[str]
    inner: Inner  # nested dataclass -> struct


EXPECTED_IR = {
    "type": "struct",
    "fields": [
        {"name": "name", "dtype": {"type": "str"}, "required": False},
        {"name": "age", "dtype": {"type": "i64"}, "required": False},
        {"name": "score", "dtype": {"type": "f64"}, "required": False},
        {"name": "active", "dtype": {"type": "bool"}, "required": False},
        {"name": "nickname", "dtype": {"type": "i64"}, "required": False},
        {
            "name": "tags",
            "dtype": {"type": "list", "inner": {"type": "str"}},
            "required": False,
        },
        {
            "name": "inner",
            "dtype": {
                "type": "struct",
                "fields": [
                    {"name": "x", "dtype": {"type": "i64"}, "required": False},
                    {"name": "y", "dtype": {"type": "str"}, "required": False},
                ],
            },
            "required": False,
        },
    ],
}

EXPECTED_DTYPE = pl.Struct(
    {
        "name": pl.String,
        "age": pl.Int64,
        "score": pl.Float64,
        "active": pl.Boolean,
        "nickname": pl.Int64,
        "tags": pl.List(pl.String),
        "inner": pl.Struct({"x": pl.Int64, "y": pl.String}),
    }
)

SAMPLE = (
    '{"name": "ada", "age": 36, "score": 9.5, "active": true, '
    '"nickname": 7, "tags": ["a", "b"], "inner": {"x": 1, "y": "z"}}'
)
EXPECTED_VALUE = {
    "name": "ada",
    "age": 36,
    "score": 9.5,
    "active": True,
    "nickname": 7,
    "tags": ["a", "b"],
    "inner": {"x": 1, "y": "z"},
}


def test_dataclass_normalize_ir():
    assert normalize(Person) == EXPECTED_IR


def test_dataclass_decode_eager():
    df = pl.DataFrame({"payload": [SAMPLE]})
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=Person).alias("parsed")
    )
    assert out["parsed"].dtype == EXPECTED_DTYPE
    assert out["parsed"].to_list() == [EXPECTED_VALUE]


def test_dataclass_decode_optional_missing_is_null():
    df = pl.DataFrame(
        {"payload": ['{"name": "x", "age": 1, "score": 0.0, "active": false}']}
    )
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=Person).alias("parsed")
    )
    (val,) = out["parsed"].to_list()
    assert val["name"] == "x"
    assert val["nickname"] is None
    assert val["tags"] is None
    assert val["inner"] is None


def test_dataclass_scalar_union_raises_clear_message():
    @dataclass
    class UnionDC:
        a: Union[int, str]

    with pytest.raises(NotImplementedError, match="unions are not supported"):
        normalize(UnionDC)


def test_dataclass_pipe_union_raises_clear_message():
    @dataclass
    class PipeUnionDC:
        a: "int | str"

    with pytest.raises(NotImplementedError, match="unions are not supported"):
        normalize(PipeUnionDC)


def test_dataclass_decode_lazy_matches_eager():
    df = pl.DataFrame({"payload": [SAMPLE, "not json"]})
    eager = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=Person).alias("parsed")
    )
    lazy = (
        df.lazy()
        .with_columns(fastjson_decode(pl.col("payload"), schema=Person).alias("parsed"))
        .collect()
    )
    assert lazy["parsed"].dtype == EXPECTED_DTYPE
    assert lazy.equals(eager)
    assert lazy["parsed"].to_list() == [EXPECTED_VALUE, None]
