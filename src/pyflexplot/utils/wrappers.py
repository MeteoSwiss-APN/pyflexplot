"""Wrappers for utility functions with select default arguments."""
# Standard library
from typing import Any
from typing import Type

# First-party
from srutils.dataclasses import cast_field_value as cast_field_value_


def cast_field_value(cls: Type, name: str, value: Any, **kwargs: Any) -> Any:
    kwargs = {
        "auto_wrap": True,
        "bool_mode": "intuitive",
        "timedelta_unit": "hours",
        "unpack_str": False,
        **kwargs,
    }
    return cast_field_value_(cls, name, value, **kwargs)
