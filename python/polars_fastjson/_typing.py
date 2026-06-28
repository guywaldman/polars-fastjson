"""Shared type aliases."""

from __future__ import annotations

from typing import Any, Union

import polars as pl

# A column expression accepted by `fastjson_decode`.
IntoExprColumn = Union[str, pl.Expr]

# Any schema source accepted by `schema.normalize`: a pl.DataType (Struct), a
# dict of {name: pl.DataType}, a dataclass type, a TypedDict type, or a
# pydantic BaseModel subclass.
SchemaSource = Any
