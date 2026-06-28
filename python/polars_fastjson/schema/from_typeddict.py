"""Convert a TypedDict type into SchemaIR."""

from __future__ import annotations

import typing
from typing import Any

from . import ir as _ir


def is_typeddict(obj: Any) -> bool:
    if typing.is_typeddict(obj):
        return True
    try:
        # `typing.is_typeddict` does not recognize classes built from
        # `typing_extensions.TypedDict`,
        # so fall back to the typing_extensions detector when available.
        import typing_extensions
    except Exception:  # pragma: no cover - typing_extensions not installed
        return False
    return typing_extensions.is_typeddict(obj)


def _resolve_nested(tp: Any) -> dict[str, Any] | None:
    if is_typeddict(tp):
        return from_typeddict(tp)
    return None


def _is_not_required(ann: Any) -> bool:
    """Return True if an annotation is wrapped in ``NotRequired[...]``."""
    origin = typing.get_origin(ann)
    for mod in (typing,):
        nr = getattr(mod, "NotRequired", None)
        if nr is not None and origin is nr:
            return True
    try:
        import typing_extensions

        if origin is typing_extensions.NotRequired:
            return True
    except Exception:  # pragma: no cover - typing_extensions not installed
        pass
    return False


def _is_required(ann: Any) -> bool:
    """Return True if an annotation is wrapped in ``Required[...]``."""
    origin = typing.get_origin(ann)
    req = getattr(typing, "Required", None)
    if req is not None and origin is req:
        return True
    try:
        import typing_extensions

        if origin is typing_extensions.Required:
            return True
    except Exception:  # pragma: no cover - typing_extensions not installed
        pass
    return False


def _strip_req_wrapper(ann: Any) -> Any:
    """Strip a ``Required[X]`` / ``NotRequired[X]`` wrapper down to ``X``."""
    if _is_not_required(ann) or _is_required(ann):
        args = typing.get_args(ann)
        if args:
            return args[0]
    return ann


def from_typeddict(tp: Any) -> dict[str, Any]:
    """Build a struct IR from a TypedDict type."""
    if not is_typeddict(tp):
        raise TypeError(f"from_typeddict: {tp!r} is not a TypedDict")

    # include_extras=True keeps Required/NotRequired wrappers visible.
    hints = typing.get_type_hints(tp, include_extras=True)
    required_keys = getattr(tp, "__required_keys__", frozenset())
    optional_keys = getattr(tp, "__optional_keys__", frozenset())
    total = getattr(tp, "__total__", True)

    fields: list[dict[str, Any]] = []
    for name, ann in hints.items():
        if _is_not_required(ann):
            required = False
        elif _is_required(ann):
            required = True
        elif name in optional_keys:
            required = False
        elif name in required_keys:
            required = True
        else:
            required = bool(total)

        inner = _strip_req_wrapper(ann)
        dtype = _ir.py_type_to_ir(inner, nested_resolver=_resolve_nested)
        fields.append(_ir.field(name, dtype, required=required))
    return _ir.struct(fields)
