"""The single public entry point: `fastjson_decode`."""

from __future__ import annotations

import polars as pl
from polars.plugins import register_plugin_function

from ._lib import LIB
from ._typing import IntoExprColumn, SchemaSource
from .schema import normalize


def fastjson_decode(
    expr: IntoExprColumn,
    *,
    schema: SchemaSource,
    on_error: str = "null",
    coerce: bool = True,
    extra: str = "ignore",
) -> pl.Expr:
    """Lenient, schema-aware JSON -> Struct projection.

    Args:
        expr: A JSON **string** column.
        schema: Target schema. Accepts a ``pl.DataType`` (Struct), a dict of
            ``{name: pl.DataType}``, a dataclass type, a TypedDict type, or a
            pydantic ``BaseModel`` subclass.
        on_error: ``"null"`` (default) makes a bad/mismatched row a null struct;
            ``"error"`` raises (parity escape hatch with ``str.json_decode``).
        coerce: When ``True`` (default), apply the coercion table; when
            ``False``, leaf type mismatches become null fields.
        extra: ``"ignore"`` (default) drops JSON fields not in the schema.

    Returns:
        A ``pl.Expr`` producing one ``Struct`` column. Fully lazy.
    """
    if on_error not in ("null", "error"):
        raise ValueError(f'on_error must be "null" or "error", got {on_error!r}')
    if extra != "ignore":
        raise ValueError(f'extra must be "ignore" in v1, got {extra!r}')

    ir = normalize(schema)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="fastjson_decode",
        args=[expr],
        kwargs={
            "schema_ir": ir,
            "on_error": on_error,
            "coerce": coerce,
            "extra": extra,
        },
        is_elementwise=True,
    )
