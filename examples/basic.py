"""Minimal usage example for polars-fastjson.

Requires the compiled plugin (`just build-maturin`).
"""

from __future__ import annotations

import polars as pl

from polars_fastjson import fastjson_decode


def main() -> None:
    df = pl.DataFrame(
        {
            "raw": [
                '{"id": "a", "score": 1.5, "tags": ["x", "y"]}',
                "not json",
                '{"id": "b", "score": 2.0, "tags": []}',
            ]
        }
    )

    schema = {"id": pl.String, "score": pl.Float64, "tags": pl.List(pl.String)}

    out = df.with_columns(fastjson_decode(pl.col("raw"), schema=schema).alias("parsed"))
    print(out)


if __name__ == "__main__":
    main()
