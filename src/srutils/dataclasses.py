"""
Dataclasses utilities.
"""
# Standard library
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import TypeVar

DataclassT = TypeVar("DataclassT")


def get_dataclass_fields(obj: DataclassT) -> List[str]:
    if not is_dataclass(obj):
        raise ValueError(f"not a dataclass: {type(obj).__name__}")
    # Use getattr to prevent mypy from complaining
    # pylint: disable=E1101  # no-member
    return getattr(obj, "__dataclass_fields__")


# SR_TODO Add option to retain duplicate values
def dataclass_merge(objs: Sequence[DataclassT]) -> DataclassT:
    """Merge multiple dataclass objects by merging their values into tuples."""
    cls = type(objs[0])
    if not is_dataclass(cls):
        raise ValueError(f"not a dataclass: {cls.__name__}")
    for obj in objs:
        if not isinstance(obj, cls):
            raise ValueError(f"classes differ: {cls.__name__} != {type(obj).__name__}")
    kwargs: Dict[str, Any] = {}
    for param in get_dataclass_fields(cls):
        kwargs[param] = tuple(map(lambda obj: getattr(obj, param), objs))
    return cls(**kwargs)  # type: ignore
