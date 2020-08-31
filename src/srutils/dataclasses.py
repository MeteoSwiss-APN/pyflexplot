# -*- coding: utf-8 -*-
"""
Dataclasses utilities.
"""
# Standard library
from dataclasses import is_dataclass
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

DataclassT = TypeVar("DataclassT")


# SR_TODO Add option to retain duplicate values
def dataclass_merge(
    objs: Sequence[DataclassT], expect_equal_except: Optional[Collection[str]] = None
) -> DataclassT:
    """Merge multiple dataclass objects by merging their values into tuples.

    Args:
        objs: Objects of the same dataclass type.

        expect_equal_except (optional): List of parameter names that are allowed
            to differ between ``objs``. All other parameters are expected to be
            equal, and an exception is raised if any of them differ.

    """
    cls = type(objs[0])
    if not is_dataclass(cls):
        raise ValueError(f"not a dataclass: {cls.__name__}")
    kwargs: Dict[str, Any] = {}
    # pylint: disable=E1101  # no-member
    for param in getattr(cls, "__dataclass_fields__"):  # mypy
        vals: List[Union[float, str]] = []
        for obj in objs:
            val = getattr(obj, param)
            for sub_val in val if isinstance(val, tuple) else [val]:
                if sub_val not in vals:
                    vals.append(sub_val)
        if not expect_equal_except or param in expect_equal_except:
            kwargs[param] = tuple(vals)
        elif len(vals) == 1:
            kwargs[param] = next(iter(vals))
        else:
            raise Exception(
                f"values parameter '{param}' differ between {len(objs)}"
                f" {cls.__name__} objects: {vals}"
            )
    return cls(**kwargs)  # type: ignore
