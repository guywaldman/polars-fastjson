from __future__ import annotations

from typing import Any, cast

import polars as pl
from polars.datatypes import DataTypeClass

from . import ir as _ir

# Simple Polars base-type -> scalar IR tag.
_SCALAR_MAP: dict[Any, str] = {
    pl.Null: "null",
    pl.Boolean: "bool",
    pl.Int8: "i8",
    pl.Int16: "i16",
    pl.Int32: "i32",
    pl.Int64: "i64",
    pl.UInt8: "u8",
    pl.UInt16: "u16",
    pl.UInt32: "u32",
    pl.UInt64: "u64",
    pl.Float32: "f32",
    pl.Float64: "f64",
    pl.String: "str",
    pl.Utf8: "str",
    pl.Binary: "binary",
    pl.Date: "date",
    pl.Time: "time",
}


def _dtype_to_ir(dt: pl.DataType | DataTypeClass) -> dict[str, Any]:
    # Normalize a bare class (e.g. ``pl.Int64``) to an instance.
    inst = dt() if isinstance(dt, type) else dt
    base = inst.base_type()

    tag = _SCALAR_MAP.get(base)
    if tag is not None:
        return _ir.scalar(tag)

    if base == pl.Datetime:
        return _ir.datetime(
            time_unit=_ir._time_unit_to_ir(getattr(inst, "time_unit", "us")),
            time_zone=getattr(inst, "time_zone", None),
        )
    if base == pl.Duration:
        return _ir.duration(
            time_unit=_ir._time_unit_to_ir(getattr(inst, "time_unit", "us"))
        )
    if base == pl.Decimal:
        return _ir.decimal(
            scale=getattr(inst, "scale", 0) or 0,
            precision=getattr(inst, "precision", None),
        )
    if base == pl.List:
        assert isinstance(inst, pl.List)
        return _ir.list_(_dtype_to_ir(inst.inner))
    if base == pl.Struct:
        assert isinstance(inst, pl.Struct)
        fields = [_ir.field(f.name, _dtype_to_ir(f.dtype)) for f in inst.fields]
        return _ir.struct(fields)

    raise TypeError(f"unsupported Polars dtype: {dt!r}")


def from_polars(source: pl.DataType | DataTypeClass | dict[str, Any]) -> dict[str, Any]:
    """Convert a ``pl.DataType`` (Struct) or a ``{name: dtype}`` dict to IR."""
    if isinstance(source, dict):
        # ty widens the union-narrowed dict's items to ``object`` (pl.DataType
        # is not final, so it admits a DataType&dict intersection); cast back to
        # the declared mapping type.
        field_map = cast("dict[str, pl.DataType | DataTypeClass]", source)
        fields = [_ir.field(name, _dtype_to_ir(dt)) for name, dt in field_map.items()]
        return _ir.struct(fields)

    inst = source() if isinstance(source, type) else source
    if isinstance(inst, pl.DataType) or isinstance(source, type):
        return _dtype_to_ir(inst)

    raise TypeError(f"from_polars: unsupported source {source!r}")
