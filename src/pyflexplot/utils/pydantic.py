# pylint: disable=C0302  # too-many-lines
"""Plot setup and setup files."""
# Standard library
import typing
from typing import Any
from typing import Collection
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

# Third-party
from pydantic import parse_obj_as
from pydantic import ValidationError
from pydantic.fields import ModelField

# First-party
from srutils.str import split_outside_parens


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
            except ValidationError as e2:
                # Still not working; let's give up!
                raise ValueError(
                    f"value '{value}' with type {type(value).__name__} is incompatible"
                    f" with field type {field_type}, both directly and in a sequence"
                ) from e2
            else:
                # Now it worked; wrapping value in list and we're good!
                value = [value]
        else:
            raise NotImplementedError("unknown ValidationError", error_type) from e
    return value


class InvalidParameterError(Exception):
    """Parameter is invalid."""


class InvalidParameterNameError(InvalidParameterError):
    """Parameter has invalid name."""


class InvalidParameterValueError(InvalidParameterError):
    """Parameter has invalid value."""


# pylint: disable=R0912  # too-many-branches
def cast_field_value(cls: Type, param: str, value: Any, many_ok: bool = False) -> Any:
    try:
        field: ModelField = cls.__fields__[param]
    except KeyError as e:
        raise InvalidParameterNameError(
            f"{param} ({value} [{type(value).__name__}])"
            f"; choices: {sorted(cls.__fields__)}"
        ) from e

    def invalid_value_exception(
        type_: Union[Type, str], param: str, value: Any
    ) -> InvalidParameterValueError:
        if isinstance(type_, str):
            s_type = type_
        else:
            s_type = type_.__name__
        return InvalidParameterValueError(
            f"{type(value).__name__}: {value} ({s_type}: {param})"
        )

    if str(field.type_).startswith("typing.Union["):
        union_content = str(field.type_).replace("typing.Union[", "")[:-1]
        s_sub_types = [
            s.strip() for s in split_outside_parens(union_content, ",", parens="[]")
        ]
        if "str" in s_sub_types:
            s_sub_types.remove("str")
            s_sub_types.insert(0, "str")
        for s_sub_type in s_sub_types:
            sub_type = str_get_outer_type(s_sub_type, generic=True)
            if (
                issubclass(sub_type, str)
                and isinstance(value, Sequence)
                and not isinstance(value, str)
            ):
                continue
            try:
                return sub_type(value)
            except Exception:  # pylint: disable=W0703  # broad-except
                continue
        raise Exception(
            f"failed to cast value '{value}' of '{field.type_}' parameter '{param}'"
            f" to one of {s_sub_types}"
        )

    if isinstance(value, Collection) and not isinstance(value, str):
        try:
            outer_type = field_get_outer_type(field, generic=True)
        except TypeError as e:
            raise invalid_value_exception("???", param, value) from e
        if issubclass(outer_type, (str, bool)):
            raise ValueError(
                f"invalid value type of {outer_type.__name__} parameter '{param}'"
                f": {type(value).__name__} ({value})"
            )
        try:
            outer_type(value)
        except (ValueError, TypeError) as e:
            if many_ok:
                try:
                    _values = [cast_field_value(cls, param, v) for v in value]
                except TypeError as e2:
                    raise invalid_value_exception(outer_type, param, value) from e2
                else:
                    return type(value)(_values)  # type: ignore
            raise invalid_value_exception(outer_type, param, value) from e
        else:
            return [cls.cast(param, val) for val in value]
    if issubclass(field.type_, bool):
        if value in [True, "True", "true"]:
            return True
        elif value in [False, "False", "false"]:
            return False
    else:
        try:
            return field.type_(value)
        except (TypeError, ValueError) as e:
            raise invalid_value_exception(field.type_, param, value) from e


def field_get_outer_type(field: ModelField, *, generic: bool = False) -> Type:
    """Obtain the outer type of a pydantic model field."""
    try:
        field.outer_type_()
    except TypeError:
        pass
    else:
        return field.outer_type_
    s_type = str(field.outer_type_)
    return str_get_outer_type(s_type, generic=generic)


def str_get_outer_type(s: str, *, generic: bool = False) -> Type:
    """Obtain the outer type from a string."""
    try:
        return {"tuple": tuple, "list": list, "set": set, "dict": dict, "str": str}[s]
    except KeyError:
        pass
    prefix = "typing."
    if not s.startswith(prefix):
        raise ValueError(
            f"type string '{s}' is neither generic (list, str etc.) nor starting with"
            f" '{prefix}'"
        )
    s = s[len(prefix) :].split("[")[0]
    try:
        type_ = getattr(typing, s)
    except AttributeError as e:
        raise TypeError(
            f"cannot derive type from <field>.outer_type_: typing.{s} not found",
        ) from e
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
        except KeyError as e:
            raise NotImplementedError(f"generic type for '{type_}'") from e
    return type_
