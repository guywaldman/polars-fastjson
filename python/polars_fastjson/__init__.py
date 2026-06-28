"""polars-fastjson: lenient, schema-aware JSON -> Struct projection for Polars."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .api import fastjson_decode

try:
    __version__ = version("polars-fastjson")
except PackageNotFoundError:  # pragma: no cover - not installed (editable/source)
    __version__ = "0.0.0"

__all__ = ["fastjson_decode", "__version__"]
