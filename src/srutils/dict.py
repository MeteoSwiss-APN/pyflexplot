# -*- coding: utf-8 -*-
"""
Dictionary utilities.
"""
# Standard library
import itertools
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pprint import pformat
from typing import Any
from typing import Dict
from typing import Optional

# Local
from .exceptions import KeyConflictError


def format_dictlike(obj, multiline=False, indent=1):
    """Format a dict-like object to a string."""
    if not multiline:
        indent = 0
    try:
        s = pformat(dict(obj), indent=indent, sort_dicts=False)[1:-1]
    except TypeError:
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

    def _run_rec(dct, depth, curr_depth=1):
        """Run recursively."""
        dct_lst = _dict_mult_vals_product(dct, cls_expand=cls_expand, f_expand=f_expand)
        if len(dct_lst) == 1 or curr_depth == depth:
            for _ in range(depth - curr_depth):
                # Nest further until target depth reached
                dct_lst = [dct_lst]
            result = dct_lst
        else:
            result = []
            for dct_i in dct_lst:
                obj = _run_rec(dct_i, depth, curr_depth + 1)
                result.append(obj)
        return result

    res = _run_rec(dct, depth)

    def _flatten_rec(lst):
        if not isinstance(lst, list):
            return [lst]
        flat = []
        for obj in lst:
            flat.extend(_flatten_rec(obj))
        return flat

    if flatten:
        res = _flatten_rec(res)

    return res


def _dict_mult_vals_product(dct, cls_expand=list, f_expand=None):
    if isinstance(cls_expand, type):
        cls_expand = [cls_expand]

    def f_expand_default(obj):
        return any(isinstance(obj, t) for t in cls_expand)

    if f_expand is None:
        f_expand = f_expand_default

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


def flatten_nested_dict(
    dct, *, return_paths=False, return_depths=False, tie_breaker=None
):
    """
    Flatten a nested dict by updating inward-out.

    Args:
        dct (dict): Nested dict.

        return_paths (bool, optional): Whether to return the key path of each
            value in the nested dicts. If true, the values and paths are
            returned as separate dicts in a named tuple (as "values" and
            "paths", respectively). Defaults to False.

        return_depths (bool, optional): Whether to return the nesting depth of
            each value in the nested dicts. If true, the values and depths are
            returned as separate dicts in a named tuple (as "values" and
            "depths", respectively). Defaults to False.

        tie_breaker (callable, optional): Function to determine which of two
            values with the same key at the same nesting depth has precedence.
            If omitted, an exception is raised in case of identical keys at the
            same nesting depth. Defaults to None.

    """

    def run_rec(dct, *, curr_depth=0):
        def collect_children(dct):
            children = []
            for key, val in dct.items():
                if isinstance(val, dict):
                    subchildren = run_rec(val, curr_depth=curr_depth + 1)
                    for subchild in subchildren.values():
                        subchild["path"] = tuple([key] + list(subchild["path"]))
                    children.append(subchildren)
                else:
                    child = {key: {"path": tuple(), "depth": curr_depth, "value": val}}
                    children.append(child)
            return children

        return _merge_children(collect_children(dct))

    flat = run_rec(dct)

    values, paths, depths = {}, {}, {}
    for key, val in flat.items():
        values[key] = val["value"]
        paths[key] = val["path"]
        depths[key] = val["depth"]

    if return_paths and return_depths:
        return namedtuple("result", "values paths depths")(values, paths, depths)
    elif return_paths:
        return namedtuple("result", "values paths")(values, paths)
    elif return_depths:
        return namedtuple("result", "values depths")(values, depths)
    else:
        return values


def _merge_children(children):
    flat = {}
    for child in children:
        for key, val in child.items():
            if key in flat:
                val_flat = flat[key]
                if val_flat["depth"] > val["depth"]:
                    continue
                elif val_flat["depth"] == val["depth"]:
                    if tie_breaker is None:
                        raise KeyConflictError(
                            f"key conflict at depth {val_flat['depth']}", key,
                        )
                    raise NotImplementedError(
                        f"tie_breaker is not None", tie_breaker,
                    )
            flat[key] = val
    return flat


def linearize_nested_dict(dct, match_end=None):
    """
    Convert a nested dict with N branches into N linearly nested dicts.

    Args:
        dct (dict): Nested dict.

        match_end (callable, optional): Function to match intermediate points.
            If the key of a nested dict matches, then an additional linear
            dict is created which ends at this level, containing all non-dict
            elements in the dict with the matching key, but excluding any
            further nested dicts. Defaults to None.
    """

    def _run_rec(dct):
        def separate_subdicts(dct):
            subdcts, elements = {}, {}
            for key, val in dct.items():
                if isinstance(val, dict):
                    subdcts[key] = val
                else:
                    elements[key] = val
            return subdcts, elements

        def linearize_subdicts(subdcts, elements):
            if not subdcts:
                return [elements]
            linears = []
            for key, val in subdcts.items():
                for sublinear in _run_rec(val):
                    linears.append({**elements, key: sublinear})
            return linears

        return linearize_subdicts(*separate_subdicts(dct))

    return _apply_match_end(_run_rec(dct), match_end)


@dataclass
class _State:
    subdct: Dict[str, Any]
    curr_key: str = None
    active: Dict[str, Any] = field(default_factory=dict)
    head: Optional[Dict[str, any]] = None

    def __post_init__(self):
        if self.head is None:
            self.head = self.active

    def nondict_to_head(self):
        """
        Copy non-dict elements from subdct to head.
        """
        for key, val in self.subdct.items():
            if not isinstance(val, dict):
                self.head[key] = val


def _apply_match_end(dcts, match_end):
    """
    Create a sub-branch copy up to each subdict key matching the criterion.

    The criterion is given by ``match_end`` and may be a check whether the
    key starts with an underscore. If so, each linear nested dict is
    traversed, and every time such a key (of a further nested dict) is
    encountered, a copy is made from the part of the branch that has
    already been traversed. This copy branch includes all non-dict elements
    of the dict with the matching key, but no further-nested dict elements.
    """

    if match_end is None:
        return dcts

    def _core_rec(result, dct, state=None):

        if state is None:
            state = _State(dct)

        state.nondict_to_head()

        if state.curr_key is not None and match_end(state.curr_key):
            if state.active not in result:
                result.append(deepcopy(state.active))

        done = True
        for key, val in state.subdct.items():
            if isinstance(val, dict):
                new_head = {}
                state.head[key] = new_head
                state.head = new_head
                state.curr_key = key
                state.subdct = val
                _core_rec(result, dct, state)
                done = False
        if done and state.active not in result:
            result.append(state.active)

    result = []
    for dct in dcts:
        _core_rec(result, dct)

    return result


def decompress_nested_dict(dct, return_paths=False, match_end=None):
    """
    Convert a nested dict with N branches into N unnested dicts.

    Args:
        dct (dict): Nested dict.

        return_paths (bool, optional): Whether to return the path of each value
            in the nested dicts as a separate dict. Defaults to False.

        match_end (callable, optional): Function to match intermediate points.
            See docstring of ``linearize_nested_dict`` for details. Defaults
            to None.
    """
    values, paths = [], []
    for dct_lin in linearize_nested_dict(dct, match_end=match_end):
        result = flatten_nested_dict(dct_lin, return_paths=return_paths)
        if not return_paths:
            values.append(result)
        else:
            values.append(result.values)
            paths.append(result.paths)
    if return_paths:
        return values, paths
    return values
