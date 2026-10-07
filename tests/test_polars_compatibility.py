from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from polars_fastjson import fastjson_decode


@pytest.mark.parametrize("lazy", [False, True])
def test_plugin_preserves_logical_dtypes(lazy: bool):
    schema = {
        "date": pl.Date,
        "time": pl.Time,
        "datetime": pl.Datetime("us", "UTC"),
        "duration": pl.Duration("us"),
        "decimal": pl.Decimal(precision=8, scale=2),
        "binary": pl.Binary,
    }
    df = pl.DataFrame(
        {
            "payload": [
                '{"date":"1970-01-02","time":"00:00:01",'
                '"datetime":"1970-01-01T00:00:01Z",'
                '"duration":1000000,"decimal":12.5,"binary":"hello"}',
                None,
            ]
        }
    )
    expr = fastjson_decode("payload", schema=schema).alias("parsed")
    out = df.lazy().select(expr).collect() if lazy else df.select(expr)

    assert out["parsed"].dtype == pl.Struct(schema)
    assert out["parsed"].to_list() == [
        {
            "date": date(1970, 1, 2),
            "time": time(0, 0, 1),
            "datetime": datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            "duration": timedelta(seconds=1),
            "decimal": Decimal("12.50"),
            "binary": b"hello",
        },
        None,
    ]


def test_plugin_decode_from_lazy_scan(tmp_path: Path):
    schema = {"items": pl.List(pl.Struct({"id": pl.Int64}))}
    df = pl.DataFrame(
        {
            "payload": [
                '{"items":[{"id":1},{}]}',
                "not json",
                '{"items":[]}',
                None,
            ]
        }
    )
    path = tmp_path / "payloads.csv"
    df.write_csv(path)
    query = pl.scan_csv(path).select(
        fastjson_decode("payload", schema=schema).alias("parsed")
    )

    assert query.collect_schema()["parsed"] == pl.Struct(schema)
    assert query.collect()["parsed"].to_list() == [
        {"items": [{"id": 1}, {"id": None}]},
        None,
        {"items": []},
        None,
    ]
