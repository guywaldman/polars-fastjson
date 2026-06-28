"""Schema adapters: normalize any supported source into a SchemaIR dict.

The Rust core never imports schema libraries; Python is solely responsible for
producing the JSON-serializable SchemaIR.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl

from .from_dataclass import from_dataclass
from .from_polars import from_polars
from .from_pydantic import from_pydantic, is_pydantic_model
from .from_typeddict import from_typeddict, is_typeddict

__all__ = ["normalize"]


def normalize(source: Any) -> dict[str, Any]:
    """Dispatch a schema source to the right adapter, returning a SchemaIR dict."""
    if isinstance(source, (pl.DataType, dict)):
        return from_polars(source)

    # Bare pl.DataType class (e.g. pl.Struct subclass / pl.Int64).
    if isinstance(source, type) and issubclass(source, pl.DataType):
        return from_polars(source)

    # Must precede the generic `type` / dataclass checks below.
    if is_typeddict(source):
        return from_typeddict(source)

    if isinstance(source, type) and dataclasses.is_dataclass(source):
        return from_dataclass(source)

    if is_pydantic_model(source):
        return from_pydantic(source)

    raise TypeError(f"unsupported schema source: {source!r}")
