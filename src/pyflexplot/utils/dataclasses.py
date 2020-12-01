"""Utilities for dataclasses."""
# Standard library
from typing import Any
from typing import Collection
from typing import Dict
from typing import get_type_hints
from typing import Sequence
from typing import Type
from typing import Union

# First-party
from srutils.str import split_outside_parens

# Local
from .exceptions import InvalidParameterNameError
from .exceptions import InvalidParameterValueError


def cast_field_value(cls: Type, field: str, value: Any, **kwargs: Any) -> Any:
    """Cast a value to the type of a dataclass field.

    Args:
        cls: A dataclass.

        field: Name of dataclass field.

        value: Value to cast to type of ``field``.

        kwargs (optional): Keyword arguments passed on to ``cast_value``.

    """
    try:
        type_ = get_type_hints(cls)[field]
    except KeyError as e:
        raise InvalidParameterNameError(field) from e
    try:
        return cast_value(type_, value, **kwargs)
    except ValueError as e:
        msg = (
            f"value {{}} incompatible with type {type_} of param {cls.__name__}.{field}"
        ).format(f"'{value}'" if isinstance(value, str) else value)
        raise InvalidParameterValueError(msg) from e


# pylint: disable=R0911  # too-many-return-statements (>6)
# pylint: disable=R0912  # too-many-branches (>12)
# pylint: disable=R0915  # too-many-statements (>50)
def cast_value(
    type_: Union[Type, str],
    value: Any,
    *,
    auto_wrap: bool = False,
    bool_mode: str = "native",
    unpack_str: bool = True,
) -> Any:
    """Cast a value to a type.

    Args:
        type_: Type to cast ``value`` to.

        value: Value to cast to ``type_``.

        auto_wrap (optional): Cast scalars to collection types by wrapping them
            in a list before casting.

        bool_mode (optional): Whether cast the string "False" to True ('native')
            or to False ('intuitive').

        unpack_str (optional): Unpack strings, i.e., treat a string as a
            sequence of one-character strings, as is their default behavior.

    """
    try:
        type_ = type_.__name__  # type: ignore
    except AttributeError:
        type_ = str(type_)
    if bool_mode not in ["native", "intuitive"]:
        raise ValueError(f"bool_mode neither 'native' nor 'intuitive': {bool_mode}")

    # Bundle keyword arguments to pass to recursive calls
    kwargs: Dict[str, Any] = {
        "auto_wrap": auto_wrap,
        "bool_mode": bool_mode,
        "unpack_str": unpack_str,
    }

    def value_error(value: Any, type_: str, msg: str = "") -> ValueError:
        msg = f": {msg}" if msg else ""
        return ValueError(
            f"type '{type(value).__name__}' incompatible with '{type_}'{msg}"
        )

    def has_same_type(value: Any, type_: str) -> bool:
        """Check whether the type of a value matches the type string."""
        try:
            equiv_type = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "NoneType": type(None),
                "typing.Tuple": tuple,
                "typing.List": list,
            }[type_.split("[")[0]]
        except KeyError as e:
            raise NotImplementedError(f"type check for '{type_}'") from e
        else:
            return isinstance(value, equiv_type)

    if type_ == "NoneType":
        if isinstance(value, type(None)) or value == "None":
            return None
        raise value_error(value, type_)

    elif type_ == "bool":
        if isinstance(value, str):
            if bool_mode == "intuitive" and value == "False":
                return False
            return bool(value)
        elif isinstance(value, Collection):
            raise value_error(value, type_, "is a collection")
        else:
            return bool(value)

    elif type_ == "str":
        if isinstance(value, Collection) and not isinstance(value, str):
            raise value_error(value, type_, "is a collection")
        return str(value)

    elif type_ in ["int", "float"]:
        try:
            return {"int": int, "float": float}[type_](value)
        except (TypeError, ValueError) as e:
            raise value_error(value, type_) from e

    elif type_.startswith("typing.Tuple["):
        if not isinstance(value, Sequence):
            if auto_wrap:
                value = [value]
            else:
                raise value_error(value, type_, "not a sequence")
        elif isinstance(value, str) and not unpack_str:
            if auto_wrap:
                value = [value]
            else:
                raise value_error(
                    value, type_, "to unpack strings, pass unpack_str=True"
                )
        if type_.endswith(", ...]"):
            inner_type = type_[len("typing.Tuple[") : -len(", ...]")]
            inner_values = [
                cast_value(inner_type, inner_value, **kwargs) for inner_value in value
            ]
            return tuple(inner_values)
        else:
            inner_types = split_outside_parens(
                type_[len("typing.Tuple[") : -len("]")], ", ", parens="[]"
            )
            if len(value) != len(inner_types):
                raise value_error(value, type_, "wrong length")
            inner_values = [
                cast_value(inner_type, inner_value, **kwargs)
                for inner_type, inner_value in zip(inner_types, value)
            ]
            return tuple(inner_values)

    elif type_.startswith("typing.Union["):
        inner_types = split_outside_parens(
            type_[len("typing.Union[") : -len("]")], ", ", parens="[]"
        )
        if "NoneType" in inner_types:
            inner_types.remove("NoneType")
            inner_types.insert(0, "NoneType")
        for inner_type in inner_types:
            if has_same_type(value, inner_type):
                return cast_value(inner_type, value, **kwargs)
        for inner_type in inner_types:
            try:
                return cast_value(inner_type, value, **kwargs)
            except ValueError:
                pass
        raise value_error(value, type_, f"no compatible inner type: {inner_types}")
    else:
        raise NotImplementedError(f"type '{type_}'")
