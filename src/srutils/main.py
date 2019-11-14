# -*- coding: utf-8 -*-
"""
Various utilities.
"""
import functools
import itertools
import re

from pprint import pformat


#======================================================================
# Various
#======================================================================


class Degrees:
    """Degrees, useful for instance to convert between notations."""

    def __init__(self, deg):
        """Create an instance of ``Degrees``.

        Args:
            deg: Degrees in one of the following formats:
                float: Fraction notation.
                tuple[int*1]: Full degrees.
                tuple[int*2]: Full degrees/minutes.
                tuple[int*3]: Full degrees/minutes/seconds.

        """

        # Check for fraction
        if isinstance(deg, (int, float)):
            self._frac = float(deg)
            return

        # Check for most common non-stable sequence
        if isinstance(deg, set):
            raise ValueError(f"deg cannot be a set")

        # Check for `(degs,)`
        try:
            degs, = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0])
            return

        # Check for `(degs, mins)`
        try:
            degs, mins = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0]) + deg[1]/60.0
            return

        # Check for `(degs, mins, secs)`
        try:
            degs, mins, secs = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0]) + deg[1]/60.0 + deg[2]/3600.0

        raise ValueError(
            f"invalid deg='{deg}'; "
            f"must be `float` or `(degs, [mins, [secs, ]])`")

    def frac(self):
        """Return degrees as a fraction."""
        return self._frac

    def dms(self):
        """Return full degrees, minutes, and seconds (int)."""
        degs = self._frac
        mins = degs%1*60
        secs = mins%1*60
        return int(degs), int(mins), int(secs)

    def degs(self):
        """Return full degrees (float)."""
        return self.dms()[0]

    def mins(self):
        """Return full minutes (int)."""
        return self.dms()[1]

    def secs(self):
        """Return full seconds (int)."""
        return self.dms()[2]


def isiterable(obj, str_ok=True):
    """Check whether an object is iterable.

    Args:
        obj (object): Object to check.

        str_ok (bool, optional): Whether strings are considered
            iterable. Defaults to True.

    """
    if isinstance(obj, str):
        return str_ok
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def check_array_indices(shape, inds):
    """Check that slicing indices are consistent with array shape."""

    def inds2str(inds):
        """Convert indices to string, turning slices into '::' etc."""
        strs = []
        for ind in inds:
            if isinstance(ind, slice):
                strs.append(
                    f"{'' if ind.start is None else ind.start}:"
                    f"{'' if ind.stop is None else ind.stop}:"
                    f"{'' if ind.step is None else ind.step}")
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
            e = f'{i} >= {n}'
        elif i < -n:
            e = f'{i} < -{n}'
        else:
            continue
        inds_str = inds2str(inds)
        raise IndexError(
            f"index {j} of {inds_str} out of bounds for shape {shape}: {e}")


def group_kwargs(name, name_out=None, separator=None):
    """Collect all keyword arguments whose name starts with a prefix.

    All keyword arguments '<name>__foo', '<name>__bar', etc. are
    collected and put in a dictionary as 'foo', 'bar', etc., which
    is passed on as a keyword argument '<name>'.

    Args:
        name (str): Name of the group. Constitutes the prefix of the
            arguments to be collected, separated from the latter by
            ``separator``.

        name_out (str, optional): Name of the dictionary which the
            collected arguments are grouped into. Defaults to ``name``.

        separator (str, optional): Separator between the prefixed
            ``name`` and the argument. Defaults to '__'.

    Usage example:

        @collect_kwargs('test', 'kwargs_test')
        def test(arg1, kwargs_test):
            print(kwargs_test)

        test(arg1='foo', test__arg2='bar', test__arg3='baz')

        > {'arg2': 'bar', 'arg3': 'baz'}

    """
    if name_out is None:
        name_out = name
    if separator is None:
        separator = '__'
    prefix = f'{name}{separator}'

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if name_out in kwargs:
                raise ValueError(
                    f"keyword argument '{name_out}' already present")
            group = {}
            for key in [k for k in kwargs]:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    group[new_key] = kwargs.pop(key)
            kwargs[name_out] = group
            return f(*args, **kwargs)

        return wrapper

    return decorator


def to_varname(s, lower=True):
    """Reformat a string to a valid Python variable name."""

    err = f"cannot convert '{s}' to varname"

    # Check input is valid string
    try:
        v = str(s)
    except Exception as e:
        raise ValueError(f"{err}: not valid string")
    if not v:
        raise ValueError(f"{err}: is empty")

    # Turn spaces and dashes into underscores
    v = re.sub(r'[ -]', '_', v)

    # Drop all other special characters
    v = re.sub(r'[^a-zA-Z0-9_]', '', v)

    # Check validity
    if re.match(r'^[0-9]', v):
        raise ValueError(f"{err}: '{v}' starts with number")
    if not re.match(r'^[a-zA-Z0-9_]*$', v):
        raise ValueError(f"{err}: '{v}' contains invalid characters")

    if lower:
        v = v.lower()

    return v


#======================================================================
# Dictionary-related functions
#======================================================================


def merge_dicts(*dicts, unique_keys=True):
    """Merge multiple dictionaries with or without shared keys.

    Args:
        dicts (list[dict]) Dictionaries to be merged.

        unique_keys (bool, optional): Whether keys must be unique.
            If True, duplicate keys raise a ``KeyConflictError``
            exception. If False, dicts take precedence in reverse order
            of ``dicts``, i.e., keys occurring in multiple dicts will
            have the value of the last dict containing that key.

    Raises:
        KeyConflictError: If ``unique_keys=True`` when a key occurs
            in multiple dicts.

    Returns:
        dict: Merged dictionary.

    """
    merged = {}
    for dict_ in dicts:
        if not unique_keys:
            merged.update(dict_)
        else:
            for key, val in dict_.items():
                if key in merged:
                    raise KeyConflictError(key)
                merged[key] = val
    return merged


def dict_mult_vals_product(dict_):
    """Combine value lists in a dict in all possible ways.

    Example:
        in: {'foo_lst': [1, 2, 3], 'bar': 4, 'baz_lst': [5, 6]}

        out: [
                {'foo': 1, 'bar: 4, 'baz': 5},
                {'foo': 2, 'bar: 4, 'baz': 5},
                {'foo': 3, 'bar: 4, 'baz': 5},
                {'foo': 1, 'bar: 4, 'baz': 6},
                {'foo': 2, 'bar: 4, 'baz': 6},
                {'foo': 3, 'bar: 4, 'baz': 6},
            ]

    TODOs:
        Improve wording of short description of docstring.

    """
    keys, vals = [], []
    for key, val in sorted(dict_.items()):
        keys.append(key.replace('_lst', ''))
        vals.append(val if key.endswith('_lst') else [val])
    return [dict(zip(keys, vals_i)) for vals_i in itertools.product(*vals)]


def nested_dict_set(dct, keys, val):
    """Set a value in a nested dict, creating subdicts if necessary.

    Args:
        dct (dict): Dictionary.

        keys (list[<key>]): List of keys.

        val (object): Value.

    """
    parent = dct
    n = len(keys)
    for i, key in enumerate(keys):
        if i < n - 1:
            if key not in parent:
                parent[key] = {}
            parent = parent[key]
        else:
            parent[key] = val


def pformat_dictlike(obj):
    """Pretty-format a dict-like object."""

    s = f'{obj.__class__.__name__}({pformat(dict(obj))})'

    # Insert line breaks between braces and content
    s = s.replace("({'", "({\n '").replace('})', ',\n)}')

    # Increase indent
    #s = s.replace('\n ', '\n  ')

    return s
