"""Various utilities."""
from __future__ import annotations

# Standard library
import functools
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


def check_array_indices(
    shape: Tuple[int, ...], inds: Sequence[Union[int, slice]]
) -> None:
    """Check that slicing indices are consistent with array shape."""

    def inds2str(inds: Sequence[Union[int, slice]]) -> str:
        """Convert indices to string, turning slices into '::' etc."""
        strs = []
        for ind in inds:
            if isinstance(ind, slice):
                strs.append(
                    f"{'' if ind.start is None else ind.start}:"
                    f"{'' if ind.stop is None else ind.stop}:"
                    f"{'' if ind.step is None else ind.step}"
                )
            else:
                strs.append(str(ind))
        return f"[{', '.join(strs)}]"

    if len(inds) > len(shape):
        inds_str = inds2str(inds)
        raise IndexError(f"too many indices for shape {shape}: {inds_str}")

    for j, (n, i) in enumerate(zip(shape, inds)):
        if isinstance(i, slice):
            continue
        elif i >= n:
            e = f"{i} >= {n}"
        elif i < -n:
            e = f"{i} < -{n}"
        else:
            continue
        inds_str = inds2str(inds)
        raise IndexError(
            f"index {j} of {inds_str} out of bounds for shape {shape}: {e}"
        )


def group_kwargs(
    name: str, name_out: Optional[str] = None, separator: Optional[str] = None
) -> Callable:
    """Collect all keyword arguments whose name starts with a prefix.

    All keyword arguments '<name>__foo', '<name>__bar', etc. are collected and
    put in a dictionary as 'foo', 'bar', etc., which is passed on as a keyword
    argument '<name>'.

    Args:
        name: Name of the group. Constitutes the prefix of the arguments to be
            collected, separated from the latter by ``separator``.

        name_out (optional): Name of the dictionary which the collected
            arguments are grouped into. Defaults to ``name``.

        separator (optional): Separator between the prefixed ``name`` and the
            argument.

    Usage example:

        @group_kwargs('test', 'kwargs_test')
        def test(arg1, kwargs_test):
            print(kwargs_test)

        test(arg1='foo', test__arg2='bar', test__arg3='baz')

        > {'arg2': 'bar', 'arg3': 'baz'}

    """
    if name_out is None:
        name_out = name
    if separator is None:
        separator = "__"
    prefix = f"{name}{separator}"

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if name_out in kwargs:
                raise ValueError(f"keyword argument '{name_out}' already present")
            group = {}
            for key in list(kwargs):
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    group[new_key] = kwargs.pop(key)
            kwargs[name_out] = group
            return f(*args, **kwargs)

        return wrapper

    return decorator
