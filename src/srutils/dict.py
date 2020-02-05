# -*- coding: utf-8 -*-
"""
Dictionary utilities.
"""
import itertools

from copy import deepcopy
from pprint import pformat

from .exceptions import KeyConflictError
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


def decompress_dict_multivals(
    dct, depth, cls_expand=(list, tuple), f_expand=None, flatten=False
):
    """Combine dict with some nested list values into object-value dicts.

    Args:
        dct (dict): Specifications dict.

        depth (int): Depth to which nested list values are resolved.

        cls_expand (type or list[type], optional): One or more types, instances
            of which will be expanded. Overridden by ``f_expand``. Defaults to
            (list, tuple).

        f_expand (callable, optional): Function to evaluate whether an object
            is expandable. Trivial example (equivalent to ``cls_expand=lst``):
            ``lambda obj: isinstance(obj, lst)``. Overrides ``cls_expand``.
            Defaults to None.

        flatten (bool, optional): Flatten the nested results list. Defaults to
            False.

    Example:
        >>> f = decompress_dict_multivals
        >>> d = {
                "a": [1, [3, 4]],
                "b": 5,
            }
        >>> f(d, depth=3)
        [
            [[{"a": 1, "b": 5}]],
            [[{"a": 3, "b": 5}, {"a": 4, "b": 5}]]
        ]

    """
    if not isinstance(depth, int) or depth <= 0:
        raise ValueError(f"depth must be a positive integer", depth)

    def run_rec(dct, depth, _curr_depth=1):
        """Run recursively."""
        dct_lst = _dict_mult_vals_product(dct, cls_expand=cls_expand, f_expand=f_expand)
        if len(dct_lst) == 1 or _curr_depth == depth:
            for _ in range(depth - _curr_depth):
                # Nest further until target depth reached
                dct_lst = [dct_lst]
            result = dct_lst
        else:
            result = []
            for dct_i in dct_lst:
                obj = run_rec(dct_i, depth, _curr_depth + 1)
                result.append(obj)
        return result

    res = run_rec(dct, depth)

    def flatten_rec(lst):
        if not isinstance(lst, list):
            return [lst]
        flat = []
        for obj in lst:
            flat.extend(flatten_rec(obj))
        return flat

    if flatten:
        res = flatten_rec(res)

    return res


def _dict_mult_vals_product(dct, cls_expand=list, f_expand=None):
    if isinstance(cls_expand, type):
        cls_expand = [cls_expand]
    if f_expand is None:
        f_expand = lambda obj: any(isinstance(obj, t) for t in cls_expand)
    keys, vals = [], []
    for key, val in dct.items():
        if not f_expand(val):
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


def format_dictlike(obj, multiline=False, indent=1):
    """Format a dict-like object to a string."""
    if not multiline:
        indent = 0
    try:
        s = pformat(dict(obj), indent=indent, sort_dicts=False)[1:-1]
    except TypeError as e:
        # Option 'sort_dicts' only available in Python3.8+
        s = pformat(dict(obj), indent=indent)[1:-1]
    if multiline:
        s = "\n{s}\n"
    else:
        s = s.replace("\n", " ")
    return f"{type(obj).__name__}({s})"


def flatten_dict(dct, **kwargs):
    return NestedDictFlattener(**kwargs).run(dct)


class NestedDictFlattener:
    def __init__(self, *, retain_depth=False):
        self.retain_depth = retain_depth

    def run(self, dct):
        self._curr_depth = -1
        flat = self._run_rec(dct)
        if not self.retain_depth:
            flat = self._remove_depth(flat)
        return flat

    def _run_rec(self, dct):
        self._curr_depth += 1
        flat = self._merge_children(self._collect_children(dct))
        self._curr_depth -= 1
        return flat

    def _collect_children(self, dct):
        """
        TODO
        """
        children = []
        for key, val in dct.items():
            if isinstance(val, dict):
                child = self._run_rec(val)
            else:
                child = {key: (self._curr_depth, val)}
            children.append(child)
        return children

    def _merge_children(self, children):
        """
        TODO
        """
        flat = {}
        for child in children:
            for key, (depth, val) in child.items():
                if key in flat:
                    depth_flat, _ = flat[key]
                    if depth_flat > depth:
                        continue
                    elif depth_flat == depth:
                        raise KeyConflictError(
                            f"key conflict at depth {depth}: '{key}'"
                        )
                flat[key] = (depth, val)
        return flat

    def _remove_depth(self, dct):
        """
        TODO
        """
        return {k: v for k, (d, v) in dct.items()}
