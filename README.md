# polars-fastjson

Performant and safe JSON to `Struct` projection for [Polars](https://pola.rs).

---

## Overview & motivation

Polars is a blazing-fast DataFrame library for Python. It is built on top of
Rust's [polars](https://github.com/pola-rs/polars) and is a great choice for
data wrangling.
However, in case you have a large DataFrame with dynamic JSON columns, where the schema may break or the fields may get malformed, the existing ecosystem doesn't have a safe and ergonomic solution.

For example:

1. The polars `str.json_decode` function is not safe, and will raise an error if the JSON is malformed (aborting the entire query)
1. Working around the above with JSON path by using `pl.col("json").str.json_path_match("$.field")` is not performant (requires parsing each field individually, which can add up for a huge JSON) and does not enforce schema

`polars-fastjson` does the opposite: given a JSON string column and a target schema, it projects each row into a `Struct`, and bad JSON / missing fields / wrong leaf types degrade to **null** (or coerced
values) instead of raising (unless you opt into strict mode).
It is a real `pl.Expr` backed by a Rust [pyo3-polars](https://github.com/pola-rs/pyo3-polars) plugin, so it is vectorized, GIL-free, and lazy/streaming-compatible.

`polars-fastjson` supports the following schema sources (see [#schema-sources](#schema-sources) for more details):

- `dict` / `pl.DataType`
- `dataclass`
- `TypedDict`
- `pydantic.BaseModel`

## Install

```bash
# uv
uv add polars-fastjson

# pip
pip install polars-fastjson
```

## Quickstart

```python
import polars as pl
from polars_fastjson import fastjson_decode

df = pl.DataFrame({"raw": ['{"id": "a", "score": 1.5, "tags": ["x"]}', "not json"]})

schema = {"id": pl.String, "score": pl.Float64, "tags": pl.List(pl.String)}

out = df.with_columns(
    fastjson_decode(pl.col("raw"), schema=schema).alias("parsed")
)
# the malformed row becomes a null struct rather than raising.
```

## Schema sources

`schema=` accepts any one of: a dict / `pl.DataType`, a dataclass type, a
`TypedDict` type, or a pydantic `BaseModel` subclass. All four normalize to the
same internal schema, so they decode identically.

### dict / pl.DataType

```python
schema = {"id": pl.String, "score": pl.Float64}
# or a Struct dtype directly:
schema = pl.Struct({"id": pl.String, "score": pl.Float64})
```

### dataclass

```python
from dataclasses import dataclass

@dataclass
class Row:
    id: str
    score: float

fastjson_decode(pl.col("raw"), schema=Row)
```

### TypedDict

```python
from typing import TypedDict

class Row(TypedDict):
    id: str
    score: float

fastjson_decode(pl.col("raw"), schema=Row)
```

### pydantic BaseModel

```python
from pydantic import BaseModel, Field

class Row(BaseModel):
    user_id: str = Field(alias="user_id")
    score: float

fastjson_decode(pl.col("raw"), schema=Row)
```

Validation aliases are supported: `Field(alias="user_id")` reads the JSON key
`user_id`, and the output struct field is the **attribute name** (`user_id`
here — use a differing alias to read one key into another attribute name).
`Optional[X]` / `X | None` is supported (nullable).

> [!NOTE]
>
> While the existing support for pydantic _should_ suffice for the vast majority of use cases, not all features are supported.
>
> For example:
>
> - validators (field / model)
> - `Field` constraints (`gt`, `max_length`, …)
> - aliases of type `AliasChoices` / `AliasPath` (multiple / nested keys)
> - non-`Optional` unions (scalar like `int | str`, or model unions)
> - ...and likely more!

## Leniency & error modes

`polars-json` aims to be tolerant/lenient and avoid at all costs raising an error in case of one "bad apple".

| Situation                                         | `on_error="null"` (default)               | `on_error="error"` |
| ------------------------------------------------- | ----------------------------------------- | ------------------ |
| Invalid JSON (parse failure)                      | null row                                  | raise              |
| Missing field                                     | null field                                | null field         |
| Wrong type at leaf                                | coerced if `coerce=True`, else null field | same               |
| Extra field (not in schema)                       | ignored                                   | ignored            |
| Top-level JSON not an object (`[1,2,3]`, `42`, …) | null row                                  | raise              |
| Top-level JSON `null` literal                     | null row                                  | null row           |

```python
# strict parity with str.json_decode: bad rows raise instead of nulling.
fastjson_decode(pl.col("raw"), schema=schema, on_error="error")
```

Set `strict_required_fields=True` to make required schema fields row-level
failures instead of null fields. With `on_error="null"` the row becomes null;
with `on_error="error"` decoding raises. This is useful when you want to filter
out rows whose required fields did not decode:

```python
parsed = df.with_columns(
    fastjson_decode(
        pl.col("raw"),
        schema=User,
        strict_required_fields=True,
    ).alias("parsed")
).filter(pl.col("parsed").is_not_null())
```

### Diagnostics

You may want informative error messages if some columns fail to parse, and would want this to have minimal overhead.
You can use `diagnostics="summary"` to log parse/decode failures through the standard Python logger
(under `polars_fastjson.diagnostics`, which you can suppresss if needed):

```python
import logging

logging.basicConfig(level=logging.WARNING)

fastjson_decode(
    pl.col("raw"),
    schema=schema,
    on_error="null",
    diagnostics="summary",
    diagnostics_id="event_id",  # optional: attach bounded IDs to each cluster
)
```

Structured data is attached to each `LogRecord` as
`record.fastjson_diagnostics`.

### Type coercion

`coerce=True` (default) applies a conservative coercion table at leaves — e.g.
a JSON string `"123"` decoded into an int field becomes `123`. Set
`coerce=False` to require an exact JSON kind per leaf (mismatches -> null field).

## Heterogeneous rows: one column per type

When rows carry different shapes (e.g. discriminated by a `type` column), use
one column per type, gated by `when`/`then`, each producing its own `Struct`
dtype:

```python
df.with_columns(
    pl.when(pl.col("type") == "USER")
      .then(fastjson_decode(pl.col("raw"), schema=UserSchema))
      .alias("user"),
    pl.when(pl.col("type") == "ORG")
      .then(fastjson_decode(pl.col("raw"), schema=OrgSchema))
      .alias("org"),
)
```

The branches in a single `when`/`then` must share a dtype (Polars enforces
this); use separate columns per type, or a shared superset schema, when row
shapes differ.

## Nested data

Nested structs and lists work by nesting the schema:

```python
schema = {"user": {"id": pl.String}, "tags": pl.List(pl.String)}
# decodes {"user": {"id": "a"}, "tags": ["x", "y"]} into a nested struct.
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
