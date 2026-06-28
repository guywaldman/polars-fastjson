"""End-to-end decode tests against the compiled Rust plugin."""

from __future__ import annotations

import polars as pl
import pytest

from polars_fastjson import fastjson_decode

SCHEMA = {"id": pl.String, "score": pl.Float64, "tags": pl.List(pl.String)}
EXPECTED_DTYPE = pl.Struct(
    {"id": pl.String, "score": pl.Float64, "tags": pl.List(pl.String)}
)


def test_eager_basic_decode():
    df = pl.DataFrame({"payload": ['{"id": "a", "score": 1.5, "tags": ["x", "y"]}']})

    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=SCHEMA).alias("parsed")
    )

    assert out["parsed"].dtype == EXPECTED_DTYPE
    assert out["parsed"].to_list() == [{"id": "a", "score": 1.5, "tags": ["x", "y"]}]


def test_lenient_invalid_json_becomes_null_row():
    df = pl.DataFrame(
        {
            "payload": [
                '{"id": "a", "score": 1.5, "tags": ["x"]}',
                "not json",
                None,
            ]
        }
    )

    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=SCHEMA).alias("parsed")
    )

    values = out["parsed"].to_list()
    assert values[0] == {"id": "a", "score": 1.5, "tags": ["x"]}
    assert values[1] is None
    assert values[2] is None


def test_missing_field_becomes_null_field():
    df = pl.DataFrame({"payload": ['{"id": "a"}']})

    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=SCHEMA).alias("parsed")
    )

    assert out["parsed"].to_list() == [{"id": "a", "score": None, "tags": None}]


def test_wrong_type_coerce_true():
    # score arrives as a JSON string; coerce=True parses it to a float.
    df = pl.DataFrame({"payload": ['{"id": "a", "score": "2.5"}']})

    out = df.with_columns(
        fastjson_decode(
            pl.col("payload"),
            schema={"id": pl.String, "score": pl.Float64},
            coerce=True,
        ).alias("parsed")
    )

    assert out["parsed"].to_list() == [{"id": "a", "score": 2.5}]


def test_wrong_type_coerce_false_becomes_null_field():
    df = pl.DataFrame({"payload": ['{"id": "a", "score": "2.5"}']})

    out = df.with_columns(
        fastjson_decode(
            pl.col("payload"),
            schema={"id": pl.String, "score": pl.Float64},
            coerce=False,
        ).alias("parsed")
    )

    # coerce=False: a string at a float leaf -> null field.
    assert out["parsed"].to_list() == [{"id": "a", "score": None}]


def test_lazy_matches_eager():
    df = pl.DataFrame(
        {
            "payload": [
                '{"id": "a", "score": 1.5, "tags": ["x", "y"]}',
                "not json",
                '{"id": "b"}',
            ]
        }
    )

    eager = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=SCHEMA).alias("parsed")
    )
    lazy = (
        df.lazy()
        .with_columns(fastjson_decode(pl.col("payload"), schema=SCHEMA).alias("parsed"))
        .collect()
    )

    # Plan-time declared dtype (callback) must agree with the built Series, so
    # the lazy collect produces an identical frame.
    assert lazy["parsed"].dtype == EXPECTED_DTYPE
    assert eager["parsed"].dtype == lazy["parsed"].dtype
    assert lazy.equals(eager)


def test_on_error_error_raises_on_invalid_json():
    df = pl.DataFrame({"payload": ['{"id": "a"}', "not json"]})

    with pytest.raises(Exception):
        df.with_columns(
            fastjson_decode(pl.col("payload"), schema=SCHEMA, on_error="error").alias(
                "parsed"
            )
        )


def test_on_error_error_raises_under_lazy_collect():
    df = pl.DataFrame({"payload": ["not json"]})

    lf = df.lazy().with_columns(
        fastjson_decode(pl.col("payload"), schema=SCHEMA, on_error="error").alias(
            "parsed"
        )
    )

    with pytest.raises(Exception):
        lf.collect()
