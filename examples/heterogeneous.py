"""Heterogeneous rows: one Struct column per type, plus the strict error mode.

Rows are discriminated by a "type" column ("user" / "org") and carry a "raw"
JSON string. Each type is decoded into its own column via when/then, because
Polars requires both branches of a single when/then to share a dtype. The
second block shows on_error="error" raising on a malformed row.

Requires the compiled plugin (`just build-maturin`).
"""

from __future__ import annotations

import polars as pl

from polars_fastjson import fastjson_decode

USER_SCHEMA = {"id": pl.String, "score": pl.Float64}
ORG_SCHEMA = {"name": pl.String, "members": pl.Int64}


def main() -> None:
    df = pl.DataFrame(
        {
            "type": ["user", "org", "user"],
            "raw": [
                '{"id": "a", "score": 1.5}',
                '{"name": "acme", "members": 12}',
                '{"id": "b", "score": 2.0}',
            ],
        }
    )

    out = df.with_columns(
        pl.when(pl.col("type") == "user")
        .then(fastjson_decode(pl.col("raw"), schema=USER_SCHEMA))
        .alias("user"),
        pl.when(pl.col("type") == "org")
        .then(fastjson_decode(pl.col("raw"), schema=ORG_SCHEMA))
        .alias("org"),
    )
    print(out)

    # Strict mode: a malformed row raises instead of nulling.
    bad = pl.DataFrame({"raw": ['{"id": "a", "score": 1.5}', "not json"]})
    try:
        bad.with_columns(
            fastjson_decode(pl.col("raw"), schema=USER_SCHEMA, on_error="error").alias(
                "parsed"
            )
        )
    except Exception as err:  # noqa: BLE001 - demonstrating the raised error
        print(f"on_error='error' raised: {err!r}")


if __name__ == "__main__":
    main()
