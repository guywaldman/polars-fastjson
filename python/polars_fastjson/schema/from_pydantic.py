from __future__ import annotations

import typing
from typing import Any

from . import ir as _ir


def _get_base_model() -> Any | None:
    """Lazily import ``pydantic.BaseModel``; return None if unavailable."""
    try:
        from pydantic import BaseModel
    except Exception:  # pragma: no cover - pydantic not installed
        return None
    return BaseModel


def is_pydantic_model(obj: Any) -> bool:
    """Return True if `obj` is a pydantic ``BaseModel`` subclass."""
    base = _get_base_model()
    if base is None:
        return False
    return isinstance(obj, type) and issubclass(obj, base)


def _strip_optional_args(tp: Any) -> tuple[Any, bool]:
    """Return (inner, was_optional) for ``Optional[X]`` / ``X | None``.

    Multi-member unions (scalar or model) are unsupported and raise a clear
    message via the shared classifier.
    """
    import types as _types

    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is getattr(_types, "UnionType", object()):
        args = typing.get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
        # Arbitrary / discriminated / scalar / model unions are unsupported.
        _ir.raise_union_unsupported(non_none)
    return tp, False


def _resolve_nested(tp: Any) -> dict[str, Any] | None:
    if is_pydantic_model(tp):
        return from_pydantic(tp)
    return None


def _check_field_unsupported(name: str, field_info: Any) -> None:
    """Raise NotImplementedError for Field constraints (aliases are supported)."""
    # Field constraints (metadata carries Gt/Lt/MinLen/etc.).
    if getattr(field_info, "metadata", None):
        raise NotImplementedError(
            f"pydantic Field constraints are not supported (field {name!r})"
        )


def _resolve_json_key(name: str, field_info: Any) -> str:
    """Resolve the JSON key to read this field from.

    Decoding reads incoming JSON, so only the *validation* alias matters:
    prefer ``validation_alias`` (if a plain string), else ``alias``, else the
    field name. ``serialization_alias`` is irrelevant to decoding and ignored.
    An ``AliasChoices`` / ``AliasPath`` validation alias (multiple/nested keys)
    is unsupported.
    """
    validation_alias = getattr(field_info, "validation_alias", None)
    if validation_alias is not None:
        if isinstance(validation_alias, str):
            return validation_alias
        # AliasChoices / AliasPath (anything that is not a plain str).
        raise NotImplementedError(
            f"pydantic AliasChoices/AliasPath not supported (field {name!r})"
        )
    alias = getattr(field_info, "alias", None)
    if isinstance(alias, str):
        return alias
    return name


def from_pydantic(tp: Any) -> dict[str, Any]:
    """Build a struct IR from a pydantic ``BaseModel`` subclass (structural subset)."""
    if not is_pydantic_model(tp):
        raise TypeError(f"from_pydantic: {tp!r} is not a pydantic BaseModel subclass")

    # Validators present -> unsupported (fail loud).
    if getattr(tp, "__pydantic_decorators__", None) is not None:
        decs = tp.__pydantic_decorators__
        if (
            getattr(decs, "validators", None)
            or getattr(decs, "field_validators", None)
            or getattr(decs, "model_validators", None)
        ):
            raise NotImplementedError(
                "pydantic validators are not supported by polars-fastjson"
            )

    fields: list[dict[str, Any]] = []
    for name, field_info in tp.model_fields.items():
        _check_field_unsupported(name, field_info)
        json_key = _resolve_json_key(name, field_info)
        ann = field_info.annotation
        inner, _was_optional = _strip_optional_args(ann)
        dtype = _ir.py_type_to_ir(inner, nested_resolver=_resolve_nested)
        required = bool(getattr(field_info, "is_required", lambda: False)())
        fields.append(_ir.field(name, dtype, required=required, json_key=json_key))
    return _ir.struct(fields)
