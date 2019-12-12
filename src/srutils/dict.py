# -*- coding: utf-8 -*-
"""
Dictionary utilities.
"""
import itertools
from pprint import pformat

from .various import isiterable


def merge_dicts(*dicts, unique_keys=True):
    """Merge multiple dictionaries with or without shared keys.

    Args:
        dicts (list[dict]) Dictionaries to be merged.

        unique_keys (bool, optional): Whether keys must be unique. If True,
            duplicate keys raise a ``KeyConflictError`` exception. If False,
            dicts take precedence in reverse order of ``dicts``, i.e., keys
            occurring in multiple dicts will have the value of the last dict
            containing that key.

    Raises:
        KeyConflictError: If ``unique_keys=True`` when a key occurs in multiple
            dicts.

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


def dict_mult_vals_product(dct, lst_sfx="_lst", allowed_iterables=None):
    """Create multiple dicts by building all combinations of all list elements.

    Args:
        dct (dict): Dictionary with list elements to be combined.

        lst_sfc (str, optional): Suffix of key of list elements. Removed from
            the key names in the output dictionaries. Defaults to "_lst".

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
    if allowed_iterables is None:
        allowed_iterables = []
    elif isinstance(allowed_iterables, type):
        allowed_iterables = [allowed_iterables]
    keys, vals = [], []
    for key, val in dct.items():
        if isiterable(val, str_ok=False):
            if key.endswith(lst_sfx):
                key = key[: -len(lst_sfx)]
            elif type(val) not in allowed_iterables:
                s_val = f"'{val}'" if isinstance(val, str) else str(val)
                raise Exception(
                    f"{s_val} is iterable, but key '{key}' ends not in '{lst_sfx}' "
                    f"and type {type(val).__name__} is not among allowed iterables "
                    f"({', '.join([t.__name__ for t in allowed_iterables])})"
                )
        else:
            val = [val]
        keys.append(key)
        vals.append(val)
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

    s = f"{type(obj).__name__}({pformat(dict(obj))})"

    # Insert line breaks between braces and content
    s = s.replace("({'", "({\n '").replace("})", ",\n)}")

    # Increase indent
    # s = s.replace('\n ', '\n  ')

    return s
