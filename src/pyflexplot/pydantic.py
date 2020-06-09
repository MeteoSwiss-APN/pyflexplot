# -*- coding: utf-8 -*-
# pylint: disable=C0302  # too-many-lines
"""
Plot setup and setup files.
"""
# Standard library
import typing
from typing import Any
from typing import Collection
from typing import Optional
from typing import Sequence
from typing import Type

# Third-party
from pydantic import parse_obj_as
from pydantic import ValidationError
from pydantic.fields import ModelField


def prepare_field_value(
    field: ModelField, value: Any, *, alias_none: Optional[Sequence[Any]] = None
) -> Any:
    """Convert a value to a type compatible with a model field."""
    if alias_none is None:
        alias_none = []
    alias_none = list(alias_none) + [None]
    if value in alias_none and field.allow_none:
        return None
    field_type = field.outer_type_
    try:
        # Try to convert value to the field type
        parse_obj_as(field_type, value)
    except ValidationError as e:
        # Conversion failed, so let's try something else!
        error_type = e.errors()[0]["type"]
        if error_type in ["type_error.sequence", "type_error.tuple"]:
            try:
                # Try again, with the value in a sequence
                parse_obj_as(field_type, [value])
            except ValidationError:
                # Still not working; let's give up!
                raise ValueError(
                    f"value '{value}' with type {type(value).__name__} is incompatible"
                    f" with field type {field_type}, both directly and in a sequence"
                )
            else:
                # Now it worked; wrapping value in list and we're good!
                value = [value]
        else:
            raise NotImplementedError("unknown ValidationError", error_type, e)
    return value


def cast_field_value(cls: Type, param: str, value: Any) -> Any:
    try:
        field = cls.__fields__[param]
    except KeyError:
        raise ValueError("invalid parameter name", param, sorted(cls.__fields__))
    if isinstance(value, Collection) and not isinstance(value, str):
        try:
            outer_type = field_get_outer_type(field, generic=True)
        except TypeError:
            raise ValueError("invalid parameter value: collection", param, value)
        if issubclass(outer_type, (str, bool)):
            raise ValueError(
                "invalid parameter error: not '{outer_type.__name__}}'", param, value
            )
        try:
            outer_type(value)
        except ValueError:
            raise ValueError("invalid parameter value", param, value, outer_type)
        else:
            return [cls.cast(param, val) for val in value]
    if issubclass(field.type_, bool):
        if value in [True, "True"]:
            return True
        elif value in [False, "False"]:
            return False
    else:
        try:
            return field.type_(value)
        except ValueError:
            pass
    raise ValueError("invalid parameter value", param, value, field.type_)


def field_get_outer_type(field: ModelField, *, generic: bool = False) -> Type:
    """Obtain the outer type of a pydantic model field."""
    try:
        field.outer_type_()
    except TypeError:
        pass
    else:
        return field.outer_type_
    s_type = str(field.outer_type_)
    prefix = "typing."
    if not s_type.startswith(prefix):
        raise TypeError(
            f"<field>.outer_type_ does not start with '{prefix}'", s_type, field,
        )
    s_type = s_type[len(prefix) :]
    s_type = s_type.split("[")[0]
    try:
        type_ = getattr(typing, s_type)
    except AttributeError:
        raise TypeError(
            f"cannot derive type from <field>.outer_type_: typing.{s_type} not found",
            field,
        )
    if generic:
        generics = {
            typing.Tuple: tuple,
            typing.List: list,
            typing.Sequence: list,
            typing.Set: set,
            typing.Collection: set,
            typing.Dict: dict,
            typing.Mapping: dict,
        }
        try:
            type_ = generics[type_]
        except KeyError:
            raise NotImplementedError("generic type for tpying type", type_, generics)
    return type_
