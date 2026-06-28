from __future__ import annotations

from typing import Optional, Union

import polars as pl
import pytest
from pydantic import AliasChoices, BaseModel, Field, field_validator
from typing_extensions import Literal

from polars_fastjson import fastjson_decode
from polars_fastjson.schema import normalize


class InnerModel(BaseModel):
    x: int
    y: str


class Account(BaseModel):
    name: str
    age: int
    score: float
    active: bool
    nickname: Optional[int]  # Optional[X] -> X
    tags: list[str]
    inner: InnerModel  # nested model -> struct


def test_pydantic_normalize_ir():
    ir = normalize(Account)
    assert ir["type"] == "struct"
    types = {f["name"]: f["dtype"]["type"] for f in ir["fields"]}
    assert types == {
        "name": "str",
        "age": "i64",
        "score": "f64",
        "active": "bool",
        "nickname": "i64",  # Optional stripped
        "tags": "list",
        "inner": "struct",
    }
    tags = next(f for f in ir["fields"] if f["name"] == "tags")
    assert tags["dtype"] == {"type": "list", "inner": {"type": "str"}}
    inner = next(f for f in ir["fields"] if f["name"] == "inner")
    assert inner["dtype"] == {
        "type": "struct",
        "fields": [
            {"name": "x", "dtype": {"type": "i64"}, "required": True},
            {"name": "y", "dtype": {"type": "str"}, "required": True},
        ],
    }


def test_pydantic_optional_annotation_is_not_required():
    class OptionalModel(BaseModel):
        required_value: int
        optional_value: Optional[int]

    ir = normalize(OptionalModel)
    required = {f["name"]: f["required"] for f in ir["fields"]}
    assert required == {"required_value": True, "optional_value": False}


EXPECTED_DTYPE = pl.Struct(
    {
        "name": pl.String,
        "age": pl.Int64,
        "score": pl.Float64,
        "active": pl.Boolean,
        "nickname": pl.Int64,
        "tags": pl.List(pl.String),
        "inner": pl.Struct({"x": pl.Int64, "y": pl.String}),
    }
)

SAMPLE = (
    '{"name": "ml", "age": 12, "score": 3.25, "active": false, '
    '"nickname": 4, "tags": ["g"], "inner": {"x": 2, "y": "w"}}'
)
EXPECTED_VALUE = {
    "name": "ml",
    "age": 12,
    "score": 3.25,
    "active": False,
    "nickname": 4,
    "tags": ["g"],
    "inner": {"x": 2, "y": "w"},
}


def test_pydantic_decode_eager():
    df = pl.DataFrame({"payload": [SAMPLE]})
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=Account).alias("parsed")
    )
    assert out["parsed"].dtype == EXPECTED_DTYPE
    assert out["parsed"].to_list() == [EXPECTED_VALUE]


def test_pydantic_decode_lazy_matches_eager():
    df = pl.DataFrame({"payload": [SAMPLE, "not json"]})
    eager = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=Account).alias("parsed")
    )
    lazy = (
        df.lazy()
        .with_columns(
            fastjson_decode(pl.col("payload"), schema=Account).alias("parsed")
        )
        .collect()
    )
    assert lazy["parsed"].dtype == EXPECTED_DTYPE
    assert lazy.equals(eager)
    assert lazy["parsed"].to_list() == [EXPECTED_VALUE, None]


def test_pydantic_strict_required_missing_field_nulls_row():
    class Metadata(BaseModel):
        key: str

    class User(BaseModel):
        name: str
        value: int
        metadata: Metadata

    df = pl.DataFrame(
        {
            "payload": [
                '{"name": "ok", "value": 1, "metadata": {"key": "x"}}',
                '{"name": "bad", "value_corrupted": 2, "metadata": {"key": "x"}}',
            ]
        }
    )

    lenient = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=User).alias("parsed")
    )
    assert lenient["parsed"].to_list()[1] == {
        "name": "bad",
        "value": None,
        "metadata": {"key": "x"},
    }

    strict = df.with_columns(
        fastjson_decode(
            pl.col("payload"),
            schema=User,
            strict_required_fields=True,
        ).alias("parsed")
    )
    assert strict["parsed"].to_list() == [
        {"name": "ok", "value": 1, "metadata": {"key": "x"}},
        None,
    ]


def test_pydantic_strict_required_missing_field_raises_in_error_mode():
    class User(BaseModel):
        name: str
        value: int

    df = pl.DataFrame({"payload": ['{"name": "bad", "value_corrupted": 2}']})

    with pytest.raises(Exception, match=r"required field \$\.value failed at row 0"):
        df.with_columns(
            fastjson_decode(
                pl.col("payload"),
                schema=User,
                on_error="error",
                strict_required_fields=True,
            ).alias("parsed")
        )


def test_pydantic_alias_ir_uses_json_key():
    class AliasModel(BaseModel):
        user_id_internal: str = Field(alias="user_id")

    ir = normalize(AliasModel)
    (f,) = ir["fields"]
    assert f["name"] == "user_id_internal"
    assert f["json_key"] == "user_id"


def test_pydantic_alias_decode_eager_and_lazy():
    class AliasModel(BaseModel):
        user_id_internal: str = Field(alias="user_id")

    expected_dtype = pl.Struct({"user_id_internal": pl.String})
    df = pl.DataFrame({"payload": ['{"user_id": "u1"}']})
    eager = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=AliasModel).alias("parsed")
    )
    assert eager["parsed"].dtype == expected_dtype
    assert eager["parsed"].to_list() == [{"user_id_internal": "u1"}]

    lazy = (
        df.lazy()
        .with_columns(
            fastjson_decode(pl.col("payload"), schema=AliasModel).alias("parsed")
        )
        .collect()
    )
    assert lazy.equals(eager)


def test_pydantic_validation_alias_plain_str():
    class VAModel(BaseModel):
        user_id_internal: str = Field(validation_alias="user_id")

    ir = normalize(VAModel)
    (f,) = ir["fields"]
    assert f["name"] == "user_id_internal"
    assert f["json_key"] == "user_id"

    df = pl.DataFrame({"payload": ['{"user_id": "u1"}']})
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=VAModel).alias("parsed")
    )
    assert out["parsed"].to_list() == [{"user_id_internal": "u1"}]


def test_pydantic_alias_choices_raises():
    class ACModel(BaseModel):
        user_id_internal: str = Field(validation_alias=AliasChoices("user_id", "uid"))

    with pytest.raises(NotImplementedError):
        normalize(ACModel)


def test_pydantic_serialization_alias_ignored():
    # serialization_alias is irrelevant to decoding: read by the field name,
    # no error, no json_key in the IR.
    class SAModel(BaseModel):
        user_id: str = Field(serialization_alias="userId")

    ir = normalize(SAModel)
    (f,) = ir["fields"]
    assert f["name"] == "user_id"
    assert "json_key" not in f

    df = pl.DataFrame({"payload": ['{"user_id": "u1"}']})
    out = df.with_columns(
        fastjson_decode(pl.col("payload"), schema=SAModel).alias("parsed")
    )
    assert out["parsed"].to_list() == [{"user_id": "u1"}]


def test_pydantic_field_constraint_gt_raises():
    class GtModel(BaseModel):
        a: int = Field(gt=0)

    with pytest.raises(NotImplementedError):
        normalize(GtModel)


def test_pydantic_field_constraint_max_length_raises():
    class MaxLenModel(BaseModel):
        a: str = Field(max_length=5)

    with pytest.raises(NotImplementedError):
        normalize(MaxLenModel)


def test_pydantic_validator_raises():
    class ValModel(BaseModel):
        a: int

        @field_validator("a")
        @classmethod
        def _check(cls, v: int) -> int:
            return v

    with pytest.raises(NotImplementedError):
        normalize(ValModel)


def test_pydantic_scalar_union_raises_clear_message():
    class UnionModel(BaseModel):
        a: Union[int, str]

    with pytest.raises(NotImplementedError, match="unions are not supported"):
        normalize(UnionModel)


def test_pydantic_model_union_raises_when_then_guidance():
    class ModelA(BaseModel):
        a: int

    class ModelB(BaseModel):
        b: int

    class Holder(BaseModel):
        thing: Union[ModelA, ModelB]

    with pytest.raises(NotImplementedError, match="model unions are not supported"):
        normalize(Holder)


def test_pydantic_bare_dict_raises_clear_message():
    class DictModel(BaseModel):
        metadata: dict

    with pytest.raises(TypeError) as exc_info:
        normalize(DictModel)

    message = str(exc_info.value)
    assert "unsupported mapping type for schema" in message
    assert "static output dtype" in message
    assert "nested BaseModel/dataclass/TypedDict" in message
    assert "explicit pl.Struct schema" in message


def test_pydantic_typed_dict_value_map_raises_clear_message():
    class DictModel(BaseModel):
        metadata: dict[str, str]

    with pytest.raises(TypeError) as exc_info:
        normalize(DictModel)

    message = str(exc_info.value)
    assert "unsupported mapping type for schema" in message
    assert "Dynamic dict[str, T] map fields are not supported yet" in message


def test_pydantic_discriminated_union_raises():
    class Cat(BaseModel):
        kind: Literal["cat"]
        meow: bool

    class Dog(BaseModel):
        kind: Literal["dog"]
        bark: bool

    class Pet(BaseModel):
        pet: Union[Cat, Dog] = Field(discriminator="kind")

    with pytest.raises(NotImplementedError, match="model unions are not supported"):
        normalize(Pet)


def test_pydantic_optional_int_still_works():
    class OptModel(BaseModel):
        a: Optional[int]

    ir = normalize(OptModel)
    (f,) = ir["fields"]
    assert f["dtype"] == {"type": "i64"}
