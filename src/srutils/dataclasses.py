"""
Dataclasses utilities.
"""
# Standard library
from dataclasses import is_dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import TypeVar

# First-party
from srutils.str import sfmt

DataclassT = TypeVar("DataclassT")


def get_dataclass_fields(obj: DataclassT) -> List[str]:
    if not is_dataclass(obj):
        raise ValueError(f"not a dataclass: {type(obj).__name__}")
    # Use getattr to prevent mypy from complaining
    # pylint: disable=E1101  # no-member
    return getattr(obj, "__dataclass_fields__")


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


def dataclass_repr(
    obj: DataclassT,
    indent: int = 2,
    *,
    nested: int = 0,
    fmt: Callable[[Any], str] = sfmt,
) -> str:
    """Create string representation with one argument per line."""
    body: List[str] = []
    for field in get_dataclass_fields(obj):
        value = getattr(obj, field)
        body.append(f"{field}={fmt(value)},")
    head = f"{type(obj).__name__}("
    foot = f"\n{' ' * indent * nested})"
    return f"\n{' ' * indent * (nested + 1)}".join([head] + body) + foot
