"""
Tests for `fastjson_decode` composed with Polars ``when``/``then``/``otherwise``.

When rows carry different shapes (discriminated by a ``type`` column), the
recommended pattern is **one column per type**, each gated by ``when`` and
producing its **own** ``Struct`` dtype. Non-matching rows become ``null``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_fastjson import fastjson_decode

USER_SCHEMA = {"id": pl.String, "email": pl.String}
ORG_SCHEMA = {"org_id": pl.String, "seats": pl.Int64}

USER_DTYPE = pl.Struct({"id": pl.String, "email": pl.String})
ORG_DTYPE = pl.Struct({"org_id": pl.String, "seats": pl.Int64})


def _hetero_frame() -> pl.DataFrame:
    """A frame with USER, ORG, and an unrelated ("OTHER") row type."""
    return pl.DataFrame(
        {
            "type": ["USER", "ORG", "OTHER"],
            "raw_json": [
                '{"id": "u1", "email": "u@example.com"}',
                '{"org_id": "o1", "seats": 5}',
                '{"something": "else"}',
            ],
        }
    )


def _recommended_expr() -> list[pl.Expr]:
    return [
        pl.when(pl.col("type") == "USER")
        .then(fastjson_decode(pl.col("raw_json"), schema=USER_SCHEMA))
        .alias("user"),
        pl.when(pl.col("type") == "ORG")
        .then(fastjson_decode(pl.col("raw_json"), schema=ORG_SCHEMA))
        .alias("org"),
    ]


def _assert_recommended(out: pl.DataFrame) -> None:
    assert out["user"].dtype == USER_DTYPE
    assert out["org"].dtype == ORG_DTYPE

    assert out["user"].to_list() == [
        {"id": "u1", "email": "u@example.com"},
        None,
        None,
    ]
    assert out["org"].to_list() == [
        None,
        {"org_id": "o1", "seats": 5},
        None,
    ]


def test_recommended_per_type_columns_eager():
    out = _hetero_frame().with_columns(_recommended_expr())
    _assert_recommended(out)


def test_recommended_per_type_columns_lazy():
    out = _hetero_frame().lazy().with_columns(_recommended_expr()).collect()
    _assert_recommended(out)


def test_recommended_per_type_columns_eager_matches_lazy():
    df = _hetero_frame()
    eager = df.with_columns(_recommended_expr())
    lazy = df.lazy().with_columns(_recommended_expr()).collect()
    assert eager["user"].dtype == lazy["user"].dtype
    assert eager["org"].dtype == lazy["org"].dtype
    assert eager.equals(lazy)


SUPERSET_SCHEMA = {
    "id": pl.String,
    "email": pl.String,
    "org_id": pl.String,
    "seats": pl.Int64,
}
SUPERSET_DTYPE = pl.Struct(
    {
        "id": pl.String,
        "email": pl.String,
        "org_id": pl.String,
        "seats": pl.Int64,
    }
)


def _superset_expr() -> pl.Expr:
    return (
        pl.when(pl.col("type") == "USER")
        .then(fastjson_decode(pl.col("raw_json"), schema=SUPERSET_SCHEMA))
        .otherwise(fastjson_decode(pl.col("raw_json"), schema=SUPERSET_SCHEMA))
        .alias("entity")
    )


def _assert_superset(out: pl.DataFrame) -> None:
    assert out["entity"].dtype == SUPERSET_DTYPE
    assert out["entity"].to_list() == [
        {"id": "u1", "email": "u@example.com", "org_id": None, "seats": None},
        {"id": None, "email": None, "org_id": "o1", "seats": 5},
        {"id": None, "email": None, "org_id": None, "seats": None},
    ]


def test_single_column_superset_eager():
    out = _hetero_frame().with_columns(_superset_expr())
    _assert_superset(out)


def test_single_column_superset_lazy():
    out = _hetero_frame().lazy().with_columns(_superset_expr()).collect()
    _assert_superset(out)


def test_disjoint_struct_branches_are_merged_into_superset():
    """Disjoint field names DON'T raise on Polars 1.42: they supertype-merge.

    ``then(schema={"id": str})`` and ``otherwise(schema={"org_id": str,
    "seats": int})`` resolve to a single wide ``Struct`` containing the union
    of fields. This implicit superset is a semantic loss (it conflates "not
    applicable" with "missing"), which is why per-type columns (section 1) are
    recommended over a single merged column.
    """
    df = pl.DataFrame(
        {
            "type": ["USER", "ORG"],
            "raw_json": ['{"id": "u1"}', '{"org_id": "o1", "seats": 5}'],
        }
    )
    out = df.with_columns(
        pl.when(pl.col("type") == "USER")
        .then(fastjson_decode(pl.col("raw_json"), schema={"id": pl.String}))
        .otherwise(
            fastjson_decode(
                pl.col("raw_json"),
                schema={"org_id": pl.String, "seats": pl.Int64},
            )
        )
        .alias("entity")
    )
    dtype = out["entity"].dtype
    assert isinstance(dtype, pl.Struct)
    fields = dict(dtype.to_schema())
    assert fields == {"id": pl.String, "org_id": pl.String, "seats": pl.Int64}
    rows = out["entity"].to_list()
    assert rows[0] == {"id": "u1", "org_id": None, "seats": None}
    assert rows[1] == {"id": None, "org_id": "o1", "seats": 5}


def _incompatible_branches_expr() -> pl.Expr:
    # Shared field "a": Struct in the `then` branch, List in `otherwise`.
    # No supertype exists -> Polars raises SchemaError.
    return (
        pl.when(pl.col("type") == "USER")
        .then(
            fastjson_decode(
                pl.col("raw_json"), schema={"a": pl.Struct({"k": pl.String})}
            )
        )
        .otherwise(
            fastjson_decode(pl.col("raw_json"), schema={"a": pl.List(pl.String)})
        )
        .alias("entity")
    )


def test_incompatible_branch_dtypes_raise_eager():
    """Genuinely incompatible branch dtypes raise; library does not hide it."""
    df = pl.DataFrame({"type": ["USER", "ORG"], "raw_json": ['{"a": 1}', "{}"]})
    with pytest.raises(pl.exceptions.SchemaError) as excinfo:
        df.with_columns(_incompatible_branches_expr())
    # The constraint is enforced by POLARS (supertype resolution), not us.
    assert "supertype" in str(excinfo.value)


def test_incompatible_branch_dtypes_raise_lazy():
    df = pl.DataFrame({"type": ["USER", "ORG"], "raw_json": ['{"a": 1}', "{}"]})
    with pytest.raises(pl.exceptions.SchemaError) as excinfo:
        df.lazy().with_columns(_incompatible_branches_expr()).collect()
    assert "supertype" in str(excinfo.value)


def test_when_predicate_can_be_computed_expr():
    df = pl.DataFrame(
        {
            "type": ["user", "USER", "Org"],
            "raw_json": [
                '{"id": "u1", "email": "a@x.com"}',
                '{"id": "u2", "email": "b@x.com"}',
                '{"id": "u3", "email": "c@x.com"}',
            ],
        }
    )
    out = df.with_columns(
        pl.when(pl.col("type").str.to_uppercase() == "USER")
        .then(fastjson_decode(pl.col("raw_json"), schema=USER_SCHEMA))
        .alias("user")
    )
    assert out["user"].to_list() == [
        {"id": "u1", "email": "a@x.com"},
        {"id": "u2", "email": "b@x.com"},
        None,
    ]
