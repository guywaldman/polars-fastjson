"""The same decode expressed via each of the four schema sources.

dict / pl.DataType, dataclass, TypedDict, and pydantic all normalize to the same
internal schema, so they decode identically. This prints each result so the
equivalence is visible.

Requires the compiled plugin (`just build-maturin`) and pydantic installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import polars as pl
from pydantic import BaseModel, Field

from polars_fastjson import fastjson_decode

DATA = [
    '{"user_id": "a", "score": 1.5}',
    "not json",
    '{"user_id": "b", "score": 2.0}',
]


@dataclass
class RowDataclass:
    user_id: str
    score: float


class RowTypedDict(TypedDict):
    user_id: str
    score: float


class RowPydantic(BaseModel):
    # Reads JSON key "user_id"; output struct field is the attribute name.
    user_id: str = Field(alias="user_id")
    score: float


def main() -> None:
    df = pl.DataFrame({"raw": DATA})

    dict_schema = {"user_id": pl.String, "score": pl.Float64}
    sources: dict[str, object] = {
        "dict": dict_schema,
        "dataclass": RowDataclass,
        "TypedDict": RowTypedDict,
        "pydantic": RowPydantic,
    }

    for name, schema in sources.items():
        out = df.with_columns(
            fastjson_decode(pl.col("raw"), schema=schema).alias("parsed")
        )
        print(f"--- {name} ---")
        print(out)


if __name__ == "__main__":
    main()
