"""Dataclasses utilities."""
# Standard library
from dataclasses import is_dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import TypeVar

DataclassT = TypeVar("DataclassT")


def get_dataclass_fields(obj: DataclassT) -> List[str]:
    if not is_dataclass(obj):
        raise ValueError(f"expected dataclass, not {type(obj).__name__}: {obj}")
    # Use getattr to prevent mypy from complaining
    # pylint: disable=E1101  # no-member
    return getattr(obj, "__dataclass_fields__")


def dataclass_merge(
    objs: Sequence[DataclassT], reduce_equal: bool = False
) -> DataclassT:
    """Merge multiple dataclass objects by merging their values into tuples."""
    obj0 = objs[0]
    cls = type(obj0)
    objs1 = objs[1:]
    if not is_dataclass(obj0):
        raise ValueError(f"first obj is a {cls.__name__}, not a dataclass: {obj0}")
    for obj in objs1:
        if not isinstance(obj, cls):
            raise ValueError(f"classes differ: {cls.__name__} != {type(obj).__name__}")
    kwargs: Dict[str, Any] = {}
    for param in get_dataclass_fields(cls):
        if is_dataclass(type(getattr(obj0, param))):
            kwargs[param] = dataclass_merge(
                [getattr(obj, param) for obj in objs], reduce_equal=reduce_equal
            )
        else:
            values = tuple([getattr(obj, param) for obj in objs])
            if reduce_equal and all(value == values[0] for value in values[1:]):
                kwargs[param] = values[0]
            else:
                kwargs[param] = values
    return cls(**kwargs)  # type: ignore


def dataclass_repr(
    obj: DataclassT,
    indent: int = 2,
    *,
    nested: int = 0,
    fmt: Callable[[Any], str] = repr,
) -> str:
    """Create string representation with one argument per line."""
    body: List[str] = []
    for field in get_dataclass_fields(obj):
        value = getattr(obj, field)
        body.append(f"{field}={fmt(value)},")
    head = f"{type(obj).__name__}("
    foot = f"\n{' ' * indent * nested})"
    return f"\n{' ' * indent * (nested + 1)}".join([head] + body) + foot
