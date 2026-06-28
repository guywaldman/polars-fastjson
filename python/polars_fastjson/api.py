"""The single public entry point: `fastjson_decode`."""

from __future__ import annotations

from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

from ._lib import LIB
from ._typing import IntoExprColumn, SchemaSource
from .schema import normalize

DiagnosticsMode = Literal["off", "summary"]


def fastjson_decode(
    expr: IntoExprColumn,
    *,
    schema: SchemaSource,
    on_error: str = "null",
    coerce: bool = True,
    extra: str = "ignore",
    diagnostics: DiagnosticsMode = "off",
    diagnostics_id: IntoExprColumn | None = None,
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
        diagnostics: ``"off"`` (default) emits no diagnostics. ``"summary"``
            logs clustered batch-level diagnostics through
            ``polars_fastjson.diagnostics``.
        diagnostics_id: Optional column name or expression used to attach
            bounded row IDs to diagnostic clusters.

    Returns:
        A ``pl.Expr`` producing one ``Struct`` column. Fully lazy.
    """
    if on_error not in ("null", "error"):
        raise ValueError(f'on_error must be "null" or "error", got {on_error!r}')
    if extra != "ignore":
        raise ValueError(f'extra must be "ignore" in v1, got {extra!r}')
    if diagnostics not in ("off", "summary"):
        raise ValueError(f'diagnostics must be "off" or "summary", got {diagnostics!r}')

    ir = normalize(schema)
    input_expr = pl.col(expr) if isinstance(expr, str) else expr
    args: list[pl.Expr] = [input_expr]
    if diagnostics == "summary" and diagnostics_id is not None:
        args.append(
            pl.col(diagnostics_id)
            if isinstance(diagnostics_id, str)
            else diagnostics_id
        )

    return register_plugin_function(
        plugin_path=LIB,
        function_name="fastjson_decode",
        args=args,
        kwargs={
            "schema_ir": ir,
            "on_error": on_error,
            "coerce": coerce,
            "extra": extra,
            "diagnostics": diagnostics,
        },
        is_elementwise=True,
    )
