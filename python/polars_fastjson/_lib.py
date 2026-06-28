"""Resolves the compiled plugin library path for `register_plugin_function`."""

from __future__ import annotations

from pathlib import Path

# The compiled extension module (`_internal`) is installed alongside this
# package; `plugin_path` points Polars at this directory.
LIB = Path(__file__).parent
