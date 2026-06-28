"""Smoke test: the package imports and exposes the public API."""

from __future__ import annotations


def test_imports():
    import polars_fastjson

    assert hasattr(polars_fastjson, "fastjson_decode")
    assert callable(polars_fastjson.fastjson_decode)
    assert "fastjson_decode" in polars_fastjson.__all__


def test_fastjson_decode_importable_directly():
    from polars_fastjson import fastjson_decode

    assert callable(fastjson_decode)
