from __future__ import annotations

import dataclasses
import typing
from typing import Any

from . import ir as _ir


def _resolve_nested(tp: Any) -> dict[str, Any] | None:
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return from_dataclass(tp)
    return None


def from_dataclass(tp: type) -> dict[str, Any]:
    """Build a struct IR from a dataclass type."""
    if not (isinstance(tp, type) and dataclasses.is_dataclass(tp)):
        raise TypeError(f"from_dataclass: {tp!r} is not a dataclass type")

    hints = typing.get_type_hints(tp)
    fields: list[dict[str, Any]] = []
    for f in dataclasses.fields(tp):
        ann = hints.get(f.name, f.type)
        dtype = _ir.py_type_to_ir(ann, nested_resolver=_resolve_nested)
        fields.append(_ir.field(f.name, dtype))
    return _ir.struct(fields)
