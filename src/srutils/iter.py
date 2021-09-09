"""Iteration utilities."""
# Standard library
from typing import Sequence


def flatten(obj, cls=None, max_depth=None, *, _depth=0):
    """Flatten a nested sequence recursively."""
    max_depth_reached = max_depth is not None and 0 <= max_depth < _depth

    def is_expandable(obj):
        if not isiterable(obj, str_ok=False):
            return False
        if cls is not None:
            return isinstance(obj, cls)
        return True

    if max_depth_reached or not is_expandable(obj):
        return [obj]

    flat = []
    for ele in obj:
        flat.extend(flatten(ele, cls=cls, max_depth=max_depth, _depth=_depth + 1))

    if _depth == 0:
        try:
            flat = cls(flat)
        except TypeError:
            pass
    return flat


def isiterable(obj, str_ok=True):
    """Check whether an object is iterable.

    Args:
        obj (object): Object to check.

        str_ok (bool, optional): Whether strings are considered iterable.
            Defaults to True.

    """
    if isinstance(obj, str):
        return str_ok
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def resolve_negative_indices(idcs: Sequence[int], n: int) -> Sequence[int]:
    """Turn negative indices into positive ones, given the sequence length.

    Args:
        idcs: Indices. Not required to be consecutive, in order, or unique.
            May be positive (from start) or negative (from end).

        n: Length of the sequence which the indices refer to.

    """
    idcs_new = []
    for i in idcs:
        i_in = i
        if i < 0:
            i += n
        if i < 0:
            raise ValueError(f"invalid index {i_in}: {i} < 0")
        if i >= n:
            raise ValueError(f"invalid index {i_in}: {i} >= {n}")
        idcs_new.append(i)
    return type(idcs)(idcs_new)  # type: ignore
