# -*- coding: utf-8 -*-
"""
Dictionary utilities.
"""
import itertools

from copy import deepcopy
from pprint import pformat

from .exceptions import KeyConflictError
from .various import isiterable


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


def decompress_multival_dict(
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
        >>> f = decompress_multival_dict
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

    def dict_mult_vals_product(dct, cls_expand=list, f_expand=None):
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

    def run_rec(dct, depth, curr_depth=1):
        """Run recursively."""
        dct_lst = dict_mult_vals_product(dct, cls_expand=cls_expand, f_expand=f_expand)
        if len(dct_lst) == 1 or curr_depth == depth:
            for _ in range(depth - curr_depth):
                # Nest further until target depth reached
                dct_lst = [dct_lst]
            result = dct_lst
        else:
            result = []
            for dct_i in dct_lst:
                obj = run_rec(dct_i, depth, curr_depth + 1)
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


def flatten_nested_dict(dct, *, retain_depth=False, tie_breaker=None):
    """
    Flatten a nested dict by updating inward-out.

    Args:
        dct (dict): Nested dict.

        retain_depth (bool, optional): If true, dicts values constitute two-
            element tuples comprised of the nesting depth which the value
            originates from and the value itself. Defaults to False.

        tie_breaker (callable, optional): Function to determine which of two
            values with the same key at the same nesting depth has precedence.
            If omitted, an exception is raised in case of identical keys at the
            same nesting depth. Defaults to None.

    Examples:
        TODO
    """
    def run_rec(dct, *, retain_depth, curr_depth=0):
        def collect_children(dct):
            children = []
            for key, val in dct.items():
                if isinstance(val, dict):
                    child = run_rec(val, retain_depth=True, curr_depth=curr_depth + 1)
                else:
                    child = {key: (curr_depth, val)}
                children.append(child)
            return children

        def merge_children(children):
            flat = {}
            for child in children:
                for key, (depth, val) in child.items():
                    if key in flat:
                        depth_flat, _ = flat[key]
                        if depth_flat > depth:
                            continue
                        elif depth_flat == depth:
                            if tie_breaker is None:
                                raise KeyConflictError(
                                    f"key conflict at depth {depth}: '{key}'"
                                )
                            else:
                                raise NotImplementedError(
                                    f"tie_breaker is not None", tie_breaker,
                                )
                    flat[key] = (depth, val)
            return flat

        flat = merge_children(collect_children(dct))
        if not retain_depth:
            flat = {k: v for k, (d, v) in flat.items()}
        return flat

    return run_rec(dct, retain_depth=retain_depth)


def linearize_nested_dict(dct):
    """
    Convert a nested dict with N branches into N linearly nested dicts.

    Args:
        dct (dict): Nested dict.
    """

    def run_rec(dct):

        def separate_subdicts(dct):
            subdicts, elements = {}, {}
            for key, val in dct.items():
                if isinstance(val, dict):
                    subdicts[key] = val
                else:
                    elements[key] = val
            return subdicts, elements

        def linearize_subdicts(subdicts, elements):
            if not subdicts:
                return [elements]
            linears = []
            for key, val in subdicts.items():
                for sublinear in run_rec(val):
                    linears.append({**elements, key: sublinear})
            return linears

        return linearize_subdicts(*separate_subdicts(dct))

    return run_rec(dct)


def decompress_nested_dict(dct):
    """
    Convert a nested dict with N branches into N unnested dicts.

    Args:
        dct (dict): Nested dict.
    """
    return [flatten_nested_dict(d) for d in linearize_nested_dict(dct)]
