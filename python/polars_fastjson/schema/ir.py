"""Helpers for building SchemaIR dicts and mapping Polars dtypes <-> IR.

The IR is the JSON-serializable, serde-tagged representation shared verbatim
with Rust: ``{"type": "<tag>", ...}``.
"""

from __future__ import annotations

import datetime as _dt
import types
import typing
from typing import Any

# Scalar tags that take no parameters.
_SCALAR_TAGS = frozenset(
    {
        "null",
        "bool",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f32",
        "f64",
        "str",
        "binary",
        "date",
        "time",
    }
)


def scalar(tag: str) -> dict[str, Any]:
    if tag not in _SCALAR_TAGS:
        raise ValueError(f"unknown scalar tag: {tag!r}")
    return {"type": tag}


def list_(inner: dict[str, Any]) -> dict[str, Any]:
    return {"type": "list", "inner": inner}


def struct(fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "struct", "fields": fields}


def field(
    name: str,
    dtype: dict[str, Any],
    *,
    required: bool = False,
    json_key: str | None = None,
) -> dict[str, Any]:
    """Build a struct field IR node.

    ``json_key`` is the key to read this field from in the input JSON - it is
    omitted from the IR when it equals ``name`` (the default behavior is to read
    by the field name). The output struct field is always named ``name``.
    """
    d: dict[str, Any] = {"name": name, "dtype": dtype, "required": required}
    if json_key is not None and json_key != name:
        d["json_key"] = json_key
    return d


def datetime(time_unit: str = "us", time_zone: str | None = None) -> dict[str, Any]:
    return {"type": "datetime", "time_unit": time_unit, "time_zone": time_zone}


def duration(time_unit: str = "us") -> dict[str, Any]:
    return {"type": "duration", "time_unit": time_unit}


def decimal(scale: int, precision: int | None = None) -> dict[str, Any]:
    return {"type": "decimal", "precision": precision, "scale": scale}


# --- Polars TimeUnit <-> IR ------------------------------------------------

_PL_TIME_UNIT_TO_IR = {
    "us": "us",
    "ms": "ms",
    "ns": "ns",
}


def _time_unit_to_ir(tu: str | None) -> str:
    return _PL_TIME_UNIT_TO_IR.get(tu or "us", "us")


_PY_SCALAR_MAP: dict[Any, str] = {
    str: "str",
    int: "i64",
    float: "f64",
    bool: "bool",
    bytes: "binary",
    _dt.date: "date",
    _dt.time: "time",
}


def _type_label(tp: Any) -> str:
    """A short human-readable label for a type, for error messages."""
    return getattr(tp, "__name__", None) or str(tp)


def _looks_like_struct(tp: Any) -> bool:
    """Heuristic: does ``tp`` map to a struct (model / dataclass / TypedDict)?

    Used only to pick the right error message for an unsupported union; it must
    not import pydantic, so pydantic models are detected by duck-typing on
    ``model_fields``.
    """
    import dataclasses

    if not isinstance(tp, type):
        return False
    if dataclasses.is_dataclass(tp):
        return True
    if typing.is_typeddict(tp):
        return True
    # pydantic BaseModel subclass (duck-typed, no import).
    return hasattr(tp, "model_fields")


def raise_union_unsupported(non_none_args: list[Any]) -> typing.NoReturn:
    """Raise ``NotImplementedError`` for an unsupported (>1 member) union.

    Distinguishes model/struct unions (point at the when/then pattern) from
    scalar unions (no widening; pick a single type).
    """
    if any(_looks_like_struct(a) for a in non_none_args):
        raise NotImplementedError(
            "model unions are not supported; decode each type into its own "
            "column using pl.when(...).then(fastjson_decode(..., schema=ModelA))..."
        )
    joined = " | ".join(_type_label(a) for a in non_none_args)
    raise NotImplementedError(
        f"unions are not supported (got {joined}); use a single type"
    )


def _strip_optional(tp: Any) -> Any:
    """Reduce ``Optional[X]`` / ``X | None`` to ``X`` (nullability is implicit).

    A union with more than one non-None member is unsupported and raises a clear
    message (scalar vs model union), rather than silently returning ``tp``.
    """
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is getattr(types, "UnionType", object()):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
        raise_union_unsupported(args)
    return tp


def py_type_to_ir(tp: Any, *, nested_resolver) -> dict[str, Any]:
    """Map a Python type annotation to a SchemaIR node.

    ``nested_resolver`` is a callable mapping a nested model/dataclass/TypedDict
    type to its struct IR; it lets the adapters recurse without importing one
    another.
    """
    tp = _strip_optional(tp)

    # datetime must precede date (datetime subclasses date).
    if tp is _dt.datetime:
        return datetime("us")

    scalar_tag = _PY_SCALAR_MAP.get(tp)
    if scalar_tag is not None:
        return scalar(scalar_tag)

    origin = typing.get_origin(tp)
    if origin in (list, typing.List):  # noqa: UP006
        (inner,) = typing.get_args(tp) or (str,)
        return list_(py_type_to_ir(inner, nested_resolver=nested_resolver))

    # Nested model / dataclass / TypedDict.
    nested = nested_resolver(tp)
    if nested is not None:
        return nested

    raise TypeError(f"unsupported Python type for schema: {tp!r}")
