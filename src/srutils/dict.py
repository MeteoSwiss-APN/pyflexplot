# -*- coding: utf-8 -*-
"""
Dictionary utilities.
"""
# Standard library
import itertools
from collections import namedtuple
from copy import copy
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pprint import pformat
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

# Local
from .exceptions import KeyConflictError


def format_dictlike(obj, multiline=False, indent=1):
    """Format a dict-like object to a string."""
    if not multiline:
        indent = 0
    try:
        obj = obj.dict()
    except AttributeError:
        obj = dict(obj)
    try:
        # pylint: disable=E1123  # unexpected-keyword-arg (sort_dicts)
        s = pformat(obj, indent=indent, sort_dicts=False)[1:-1]
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


def compress_multival_dicts(
    dcts: Sequence[Mapping[str, Any]],
    cls_seq: Type[Union[List[Any], Tuple[Any, ...]]] = list,
) -> Dict[str, Any]:
    """Compress multiple dicts with shared keys into one multi-value dict.

    Args:
        dcts: Dicts.

        cls_seq (optional): Type of sequence for multi-values.

    """
    if not dcts:
        raise ValueError("missing dicts")

    # SR_TODO Consider adding option to allow differing keys
    if not all(dct.keys() == dcts[0].keys() for dct in dcts):
        raise ValueError("keys differ between dicts", [dct.keys() for dct in dcts])

    dct: Mapping[str, Any] = {
        key: list(val) if isinstance(val, cls_seq) else [copy(val)]
        for key, val in dcts[0].items()
    }
    for dct_i in dcts[1:]:
        for key, val in dct.items():
            val_i = dct_i[key]
            if isinstance(val_i, cls_seq):
                val_i = list(val_i)
            else:
                val_i = [val_i]
            for val_ij in val_i:
                if val_ij not in val:
                    val.append(val_ij)
    dct = {
        key: next(iter(val)) if len(val) == 1 else cls_seq(val)
        for key, val in dct.items()
    }
    return dct


def decompress_multival_dict(
    dct: Mapping[str, Any],
    *,
    select: Optional[Collection[str]] = None,
    skip: Optional[Collection[str]] = None,
    depth: int = 1,
    cls_expand: Union[type, Collection[type]] = (list, tuple),
    f_expand: Optional[Callable[[Any], bool]] = None,
    flatten: bool = False,
) -> List[Mapping[str, Any]]:
    """Combine dict with some nested list values into object-value dicts.

    Args:
        dct: Specifications dict.

        select: Names of keys to be expanded. If passed, all unlisted keys are
            skipped.

        skip: Names of keys that are not expanded.

        depth: Depth to which nested list values are resolved.

        cls_expand: One or more types, instances of which will be expanded.
            Overridden by ``f_expand``.

        f_expand: Function to evaluate whether an object is expandable. Trivial
            example (equivalent to ``cls_expand=lst``): ``lambda obj:
            isinstance(obj, lst)``. Overrides ``cls_expand``.

        flatten: Flatten the nested results list.

    """
    if not isinstance(depth, int) or depth <= 0:
        raise ValueError(f"depth must be a positive integer", depth)

    def run_rec(dct, depth, curr_depth=1):
        """Run recursively."""
        dct_lst = _dict_mult_vals_product(dct, select, skip, cls_expand, f_expand)
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


def _dict_mult_vals_product(dct, select, skip, cls_expand, f_expand):
    def select_key(key):
        if select is None:
            return True
        return key in select

    def skip_key(key):
        if skip is None:
            return False
        return key in skip

    def expand_val(val):
        if f_expand is not None:
            return f_expand(val)
        elif isinstance(cls_expand, type):
            return isinstance(val, cls_expand)
        return any(isinstance(val, t) for t in cls_expand)

    keys, vals = [], []
    for key, val in dct.items():
        if not select_key(key) or skip_key(key) or not expand_val(val):
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
                if isinstance(val, Mapping):
                    subchildren = run_rec(val, curr_depth=curr_depth + 1)
                    for subchild in subchildren.values():
                        subchild["path"] = tuple([key] + list(subchild["path"]))
                    children.append(subchildren)
                else:
                    child = {key: {"path": tuple(), "depth": curr_depth, "value": val}}
                    children.append(child)
            return children

        return _merge_children(collect_children(dct), tie_breaker)

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


def _merge_children(children, tie_breaker):
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


def linearize_nested_dict(dct, branch_end_criterion=None):
    return NestedDictLinearizer(dct).run(branch_end_criterion)


# SR_TODO Refactor this class! Eliminate _State etc.!
class NestedDictLinearizer:
    """Convert a nested dict with N branches into N linearly nested dicts."""

    def __init__(self, dct):
        """Create an instance of ``NestedDictLinearizer``.

        Args:
            dct (dict): Nested dict.

        """
        self.dct = dct

    def run(self, branch_end_criterion=None):
        """
        Linearize the nested dict.

        Args:
            branch_end_criterion (callable, optional): Function to match branch
                end points. For each key that fulfils this criterion (i.e., for
                which the function returns True), a linear branch is created
                from the beginning to this key. Defaults to None.

        """
        linears = self._run_rec(self.dct)
        if branch_end_criterion:
            linears = self._apply_branch_end_criterion(linears, branch_end_criterion)
        return linears

    def _run_rec(self, dct=None):

        if dct is None:
            dct = self.dct

        def separate_subdicts(dct):
            subdcts, elements = {}, {}
            for key, val in dct.items():
                if isinstance(val, Mapping):
                    subdcts[key] = val
                else:
                    elements[key] = val
            return subdcts, elements

        def linearize_subdicts(subdcts, elements):
            if not subdcts:
                return [elements]
            linears = []
            for key, val in subdcts.items():
                for sublinear in self._run_rec(val):
                    linears.append({**elements, key: sublinear})
            return linears

        return linearize_subdicts(*separate_subdicts(dct))

    def _apply_branch_end_criterion(
        self, dcts: List[Dict[str, Any]], criterion: Callable[[str], bool],
    ) -> List[Dict[str, Any]]:
        """
        Create a sub-branch copy up to each subdict key matching the criterion.

        The criterion is given by ``criterion`` and may be a check whether
        the key starts with an underscore. If so, each linear nested dict is
        traversed, and every time such a key (of a further nested dict) is
        encountered, a copy is made from the part of the branch that has
        already been traversed. This copy branch includes all non-dict elements
        of the dict with the matching key, but no further-nested dict elements.
        """

        @dataclass
        class _State:
            subdct: Dict[str, Any]
            curr_key: Optional[str] = None
            active: Dict[str, Any] = field(default_factory=dict)
            head: Optional[Dict[str, Any]] = None

            def __post_init__(self):
                if self.head is None:
                    self.head = self.active

        def _core_rec(result, dct, state=None):

            if state is None:
                state = _State(dct)

            self._nondict_to_head(state)

            if (
                state.curr_key is not None
                and criterion(state.curr_key)
                and state.active not in result
            ):
                result.append(deepcopy(state.active))

            for key, val in state.subdct.items():
                if isinstance(val, Mapping):
                    new_head = {}
                    state.head[key] = new_head
                    state.head = new_head
                    state.curr_key = key
                    state.subdct = val
                    _core_rec(result, dct, state)

        result: List[Dict[str, Any]] = []
        for dct in dcts:
            _core_rec(result, dct)

        return result

    def _nondict_to_head(self, state):
        """Copy non-dict elements from subdct to head."""
        for key, val in state.subdct.items():
            if not isinstance(val, Mapping):
                state.head[key] = val


def decompress_nested_dict(dct, return_paths=False, branch_end_criterion=None):
    """
    Convert a nested dict with N branches into N unnested dicts.

    Args:
        dct (dict): Nested dict.

        return_paths (bool, optional): Whether to return the path of each value
            in the nested dicts as a separate dict. Defaults to False.

        branch_end_criterion (callable, optional): Function to match end points
            of linear branches. See docstring of ``linearize_nested_dict`` for
            details. Defaults to None.

    """
    values, paths = [], []
    for dct_lin in linearize_nested_dict(
        dct, branch_end_criterion=branch_end_criterion
    ):
        result = flatten_nested_dict(dct_lin, return_paths=return_paths)
        if not return_paths:
            values.append(result)
        else:
            values.append(result.values)
            paths.append(result.paths)
    if return_paths:
        return values, paths
    return values


def nested_dict_resolve_wildcards(
    dct, *, single=True, double=True, double_only_to_ends=False,
):
    """Update regular subdicts with single/double-star wildcard subdicts.

    Args:
        dct (Dict[str, Any]): Nested dict.

        single (bool, optional): Resolve single-star wildcards. Defaults to
            True.

        double (bool, optional): Resolve double-star wildcards. Defaults to
            True.

        double_only_to_ends (bool, optional): Insert double-starred wildcard
            dicts only at the ends of branches, i.e., update only dicts that do
            not contain further nested dicts. Defaults to False.

    """
    if single and double:
        return nested_dict_resolve_double_star_wildcards(
            nested_dict_resolve_single_star_wildcards(dct),
            only_to_ends=double_only_to_ends,
        )
    elif single:
        return nested_dict_resolve_single_star_wildcards(dct)
    elif double:
        return nested_dict_resolve_double_star_wildcards(
            dct, only_to_ends=double_only_to_ends,
        )
    else:
        return deepcopy(dct)


def nested_dict_resolve_single_star_wildcards(dct):
    """Update regular subdicts with single-star wildcard subdicts.

    Args:
        dct (Dict[str, Any]): Nested dict.

    """
    dct = deepcopy(dct)

    wildcards = {}
    for key, val in dct.copy().items():
        if key == "*":
            wildcards[key] = dct.pop(key)

    for wild_val in wildcards.values():
        for key, val in dct.items():
            if isinstance(val, Mapping):
                val.update(wild_val)

    for key, val in dct.items():
        if isinstance(val, Mapping):
            dct[key] = nested_dict_resolve_single_star_wildcards(val)

    return dct


def nested_dict_resolve_double_star_wildcards(dct, only_to_ends=False):
    """Update regular subdicts with double-star wildcard subdicts.

    Args:
        dct (Dict[str, Any]): Nested dict.

        only_to_ends (bool, optional): Insert wildcard dicts only at the ends
            of branches, i.e., update only dicts that do not contain further
            nested dicts. Defaults to False.

    """
    dct = deepcopy(dct)

    def _apply_double_star(dct, wild_val, depth=0):
        subdcts = []
        for val in dct.values():
            if isinstance(val, Mapping):
                subdcts.append(val)
        if subdcts:
            for subdct in subdcts:
                _apply_double_star(subdct, wild_val, depth=depth + 1)
        if depth > 0 and (not only_to_ends or not subdcts):
            dct.update(deepcopy(wild_val))

    wildcards = {}
    for key, val in dct.copy().items():
        if key == "**":
            wildcards[key] = dct.pop(key)

    for wild_val in wildcards.values():
        _apply_double_star(dct, wild_val)

    for key, val in dct.items():
        if isinstance(val, Mapping):
            dct[key] = nested_dict_resolve_double_star_wildcards(
                val, only_to_ends=only_to_ends,
            )

    return dct


def print_dict_skeleton(dct, s="  ", _depth=0):
    for key, val in dct.items():
        if isinstance(val, Mapping):
            print(f"{s * _depth}{key}")
            print_dict_skeleton(val, s=s, _depth=_depth + 1)
