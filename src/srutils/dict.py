# -*- coding: utf-8 -*-
"""
Dictionary utilities.
"""
import itertools
from pprint import pformat


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
