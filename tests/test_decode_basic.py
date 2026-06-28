"""End-to-end decode tests against the compiled Rust plugin."""

from __future__ import annotations

import logging
from typing import Any, cast

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


def _diagnostic_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [rec for rec in caplog.records if rec.name == "polars_fastjson.diagnostics"]


def _diagnostic_payload(record: logging.LogRecord) -> dict[str, Any]:
    return cast(dict[str, Any], getattr(record, "fastjson_diagnostics"))


def test_summary_diagnostics_logs_under_null_mode(caplog):
    df = pl.DataFrame(
        {
            "payload": [
                '{"id": "a", "score": "bad", "tags": "oops"}',
                "not json",
            ]
        }
    )

    with caplog.at_level(logging.WARNING, logger="polars_fastjson.diagnostics"):
        out = df.with_columns(
            fastjson_decode(
                pl.col("payload"),
                schema=SCHEMA,
                on_error="null",
                diagnostics="summary",
            ).alias("parsed")
        )

    assert out["parsed"].to_list() == [
        {"id": "a", "score": None, "tags": None},
        None,
    ]

    records = _diagnostic_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "fastjson decode produced 3 issues in column" in message
    assert "$.score type_mismatch" in message
    assert "expected: f64" in message
    assert "$.tags type_mismatch" in message
    assert "expected: array" in message
    assert "invalid_json" in message

    payload = _diagnostic_payload(records[0])
    assert payload["column"] == "payload"
    assert payload["issues"] == 3
    clusters = {(c["path"], c["kind"]): c for c in payload["clusters"]}
    assert clusters[("$.score", "type_mismatch")]["samples"] == ['"bad"']
    assert clusters[("$.tags", "type_mismatch")]["found"] == "string"
    assert clusters[("$", "invalid_json")]["expected"] == "json"


def test_diagnostics_off_logs_nothing(caplog):
    df = pl.DataFrame({"payload": ['{"id": "a", "score": "bad"}']})

    with caplog.at_level(logging.WARNING, logger="polars_fastjson.diagnostics"):
        df.with_columns(
            fastjson_decode(pl.col("payload"), schema=SCHEMA).alias("parsed")
        )

    assert _diagnostic_records(caplog) == []


def test_summary_diagnostics_id_accepts_column_name_and_caps_ids(caplog):
    df = pl.DataFrame(
        {
            "event_id": [f"evt_{i}" for i in range(25)],
            "payload": ['{"score": "bad"}' for _ in range(25)],
        }
    )

    with caplog.at_level(logging.WARNING, logger="polars_fastjson.diagnostics"):
        df.with_columns(
            fastjson_decode(
                pl.col("payload"),
                schema={"score": pl.Float64},
                diagnostics="summary",
                diagnostics_id="event_id",
            ).alias("parsed")
        )

    payload = _diagnostic_payload(_diagnostic_records(caplog)[0])
    cluster = payload["clusters"][0]
    assert cluster["path"] == "$.score"
    assert cluster["count"] == 25
    assert cluster["ids"] == [f"evt_{i}" for i in range(20)]
    assert cluster["omitted_ids"] == 5


def test_summary_diagnostics_id_accepts_expression(caplog):
    df = pl.DataFrame(
        {
            "event_id": ["a", "b"],
            "payload": ['{"score": "bad"}', '{"score": "worse"}'],
        }
    )

    with caplog.at_level(logging.WARNING, logger="polars_fastjson.diagnostics"):
        df.with_columns(
            fastjson_decode(
                pl.col("payload"),
                schema={"score": pl.Float64},
                diagnostics="summary",
                diagnostics_id=pl.concat_str([pl.lit("id-"), pl.col("event_id")]),
            ).alias("parsed")
        )

    payload = _diagnostic_payload(_diagnostic_records(caplog)[0])
    assert payload["clusters"][0]["ids"] == ["id-a", "id-b"]


def test_invalid_diagnostics_mode_raises():
    with pytest.raises(ValueError, match="diagnostics"):
        fastjson_decode(
            pl.col("payload"),
            schema=SCHEMA,
            diagnostics=cast(Any, "verbose"),
        )
