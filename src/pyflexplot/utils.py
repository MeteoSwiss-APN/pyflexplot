# -*- coding: utf-8 -*-
"""
Utils for the command line tool.
"""
import itertools
import logging as log
import numpy as np

from pprint import pformat

from .utils_dev import ipython  #SR_DEV


class ParentClass:
    """Enable easy access to subclasses via their 'name' attribute."""

    @classmethod
    def subclass(cls, name):
        """Find a subclass by name."""
        subclasses_by_name = cls.subclasses_by_name()
        try:
            return subclasses_by_name[name]
        except KeyError:
            raise ValueError(
                f"class '{cls.__name__}' has no subclass '{name}'; "
                f"options: {sorted(subclasses_by_name)}")

    @classmethod
    def subclasses_by_name(cls, unique=True):
        """Recursively collect named subclasses by 'name' attribute."""
        result = {}
        for sub_cls in cls.__subclasses__():
            try:
                name = sub_cls.name
            except AttributeError:
                continue
            if name is None:
                continue
            if unique and result.get(name) is sub_cls:
                raise Exception(f"duplicate class name '{name}'")
            result[name] = sub_cls
            result.update(sub_cls.subclasses_by_name())
        return result


class Summarizable:

    def __init__(self, *args, **kwargs):
        raise Exception(f"{type(self).__name__} must be subclassed")

    @property
    def summarizable_attrs(self):
        raise Exception(
            f"`summarizable_attrs` must be an attribute of subclasses of "
            f"{type(self).__name__}")

    def summarize(self):
        data = {}
        data['type'] = type(self).__name__
        for attr in self.summarizable_attrs:
            val = getattr(self, attr)
            try:
                val = val.summarize()
            except AttributeError:
                pass
            data[attr] = val
        return data


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""
    pass


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""
    pass


def count_to_log_level(count: int) -> int:
    """Map the occurence of the command line option verbose to the log level"""
    if count == 0:
        return log.ERROR
    elif count == 1:
        return log.WARNING
    elif count == 2:
        return log.INFO
    else:
        return log.DEBUG


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


def pformat_dictlike(obj):
    """Pretty-format a dict-like object."""

    s = f'{obj.__class__.__name__}({pformat(dict(obj))})'

    # Insert line breaks between braces and content
    s = s.replace("({'", "({\n '").replace('})', ',\n)}')

    # Increase indent
    #s = s.replace('\n ', '\n  ')

    return s


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
