#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some testing utils.
"""
# Standard library
from dataclasses import dataclass
from pprint import pformat
from typing import Any
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence

# Third-party
import numpy as np

# Local
from .iter import isiterable
from .str import sfmt


class CheckFailedError(Exception):
    pass


def property_obj(cls, *args, **kwargs):
    """Define a class property creating a given object on-the-fly.

    The purpose of creating the object on-the-fly in a property method is to
    isolate any errors during instatiation in the test methods using the
    property. This prevents the whole test suite from being aborted, as it
    would if the object were defined as a simple class attribute and its
    instatiation failed -- instead, only the tests attempting to use the object
    will fail.

    And yes, this is indeed merely a convenience function to save two lines of
    code wherever it is used. :-)

    Usage:
        The following class definitions are equivalent:

        >>> class C1:
        ...     @property
        ...     def w(self):
        ...         return TranslatedWord(en='train', de='Zug')

        >>> class C2:
        ...     w = property_word(en='train', de='Zug')

    """

    def create_obj(self):
        return cls(*args, **kwargs)

    return property(create_obj)


class IgnoredElement:
    """Element that is ignored in comparisons."""

    def __init__(self, description=None):
        self.description = description

    def __repr__(self):
        return f"{type(self).__name__}({sfmt(self.description)})"


class UnequalElement:
    """Element unequal to any other; useful for force-fail tests."""

    def __init__(self, description=None):
        self.description = description

    def __repr__(self):
        return f"{type(self).__name__}({sfmt(self.description)})"

    def __eq__(self, other):
        return False


def ignored(obj):
    return isinstance(obj, IgnoredElement)


def check_summary_dict_is_subdict(
    subdict, superdict, subname="sub", supername="super", idx_list=None
):
    """Check that one summary dict is a subdict of another."""

    if ignored(subdict) or ignored(superdict):
        return

    if not isinstance(subdict, dict):
        raise CheckFailedError(
            f"subdict `{subname}` is not a dict, but a {type(subdict).__name__}",
            {"subdict": subdict},
        )

    if not isinstance(superdict, dict):
        raise CheckFailedError(
            f"`superdict` ('{supername}') is a {type(superdict).__name__}, not a dict",
            {"superdict": superdict},
        )

    for idx, (key, val_sub) in enumerate(subdict.items()):
        val_super = get_dict_element(superdict, key, "superdict", CheckFailedError)
        check_summary_dict_element_is_subelement(
            val_sub, val_super, subname, supername, idx_dict=idx, idx_list=idx_list
        )


def check_summary_dict_element_is_subelement(
    obj_sub,
    obj_super,
    name_sub="sub",
    name_super="super",
    idx_list=None,
    idx_dict=None,
):

    if ignored(obj_sub) or ignored(obj_super):
        return

    if obj_sub == obj_super:
        return

    # Collect objects passed to exceptions
    err_objs = {
        "name_super": name_super,
        "obj_super": obj_super,
        "name_sub": name_sub,
        "obj_sub": obj_sub,
        "idx_list": idx_list,
        "idx_dict": idx_dict,
    }

    # Check types
    t_super, t_sub = type(obj_super), type(obj_sub)
    if not isinstance(obj_sub, t_super) and not isinstance(obj_super, t_sub):
        raise CheckFailedError(
            f"incompatible types {t_super.__name__} and {t_sub.__name__} "
            f"(neither is an instance of the other)",
            {**err_objs, "t_super": t_super, "t_sub": t_sub},
        )

    if isinstance(obj_sub, dict):
        # Compare dicts
        check_summary_dict_is_subdict(
            obj_sub, obj_super, name_sub, name_super, idx_list=idx_list
        )

    elif isiterable(obj_sub, str_ok=False):
        # Compare other (non-str) iterables

        if not isiterable(obj_super, str_ok=False):
            raise CheckFailedError(f"superdict element not iterable", err_objs)

        n_sub, n_super = len(obj_sub), len(obj_super)
        if n_sub != n_super:
            raise CheckFailedError(
                f"iterable elements differ in size: {n_sub} != {n_super}",
                {**err_objs, "n_super": n_super, "n_sub": n_sub},
            )

        for idx, (subobj_sub, subobj_super) in enumerate(zip(obj_sub, obj_super)):
            check_summary_dict_element_is_subelement(
                subobj_sub,
                subobj_super,
                name_sub,
                name_super,
                idx_list=idx,
                idx_dict=idx_dict,
            )

    else:
        raise CheckFailedError(
            f"elements differ ('{name_sub}' vs. '{name_super}')", err_objs,
        )


def get_dict_element(dict_, key, name="dict", exception_type=ValueError):
    """Get an element from a dict, raising an exception otherwise."""
    try:
        return dict_[key]
    except KeyError:
        err = f"key missing in {name}: {key}"
    except TypeError:
        err = f"{name} has wrong type: {type(dict_)}"
    lines = pformat(dict_).split("\n")
    if len(lines) > 10:
        lines = lines[:5] + ["..."] + lines[-5:]
    raise exception_type(err, {"name": name, "key": key, "dict_": dict_})


@dataclass(frozen=True)
class TestConfBase:
    def derive(self, **kwargs):
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update(kwargs)
        return type(self)(**data)


def check_is_list_like(obj, *args, **kwargs):
    is_list_like(obj, *args, raise_=CheckFailedError, **kwargs)


def return_or_raise(msg, kwargs, raise_):
    if not raise_:
        return False
    if kwargs is None:
        kwargs = {}
    raise Exception(msg, kwargs)


def is_list_like(
    obj, *, len_=None, not_=None, t_children=None, f_children=None, raise_=False,
):
    """Assert that an object is list-like, with optional additional checks.

    Args:
        obj (type): Presumably list-like object.

        len_ (int, optional): Length of list-like object. Defaults to None.

        not_ (type or list[type], optional): Type(s) that ``obj`` must not be
            an instance of. Defaults to None.

        t_children (type or list[type], optional): Type(s) that the elements in
            ``obj`` must be an instance of. Defaults to None.

        f_children (callable, optional): Function used in assert with each
            element in ``obj``. Defaults to None.

        raise_ (bool, optional): Raise an exception instead of returning False.
            Defaults to False.

    """

    kwargs = {
        "obj": obj,
        "len_": len_,
        "not_": not_,
        "t_children": t_children,
        "f_children": f_children,
        "raise_": raise_,
    }

    if not isiterable(obj, str_ok=False):
        return_or_raise(
            f"{type(obj).__name__} instance `obj` is not iterable", kwargs, raise_
        )

    if len_ is not None:
        if len(obj) != len_:
            return_or_raise(
                f"obj has wrong length: {len(obj)} != {len_}", kwargs, raise_,
            )

    if not_ is not None:
        if isinstance(obj, not_):
            return_or_raise(
                f"obj has unexpected type {type(obj).__name__}", kwargs, raise_
            )

    _check_children(obj, t_children, f_children, kwargs)

    return True


def _check_children(obj, t_children, f_children, kwargs):

    if t_children is not None:
        for idx, child in enumerate(obj):
            if not isinstance(child, t_children):
                return_or_raise(
                    f"child has unexpected type {type(child).__name__}",
                    {**kwargs, "child": child, "idx": idx},
                )

    if f_children is not None:
        for idx, child in enumerate(obj):
            if not f_children(child):
                return_or_raise(
                    f"f_children returns False for child",
                    {**kwargs, "child": child, "idx": idx},
                )


def assert_nested_equal(
    obj1: Collection,
    obj2: Collection,
    float_close_ok: bool = False,
    kwargs_close: Optional[Dict[str, Any]] = None,
) -> None:
    """Compare two nested collections (dicts etc.) for equality.

    Args:
        obj1: Object compared against ``obj2``.

        obj2: Object compared against ``obj1``.

        float_close_ok (optional): Whether it is sufficient for floats to be
            close instead of identical.

        kwargs_close (optional): Keyword arguments passed to ``np.close``.

    """
    if not isinstance(obj1, Collection):
        raise ValueError(f"expecting Collection, not {type(obj1).__name__}")
    if not isinstance(obj2, Collection):
        raise ValueError(f"expecting Collection, not {type(obj2).__name__}")

    def error(msg, path, obj1=None, obj2=None):
        err = f"\n{msg}\n\nPath ({len(path)}):\n{pformat(path)}\n"
        if obj1 is not None:
            err += f"\nobj1 ({type(obj1).__name__}):\n{pformat(obj1)}\n"
        if obj2 is not None:
            err += f"\nobj2 ({type(obj2).__name__}):\n{pformat(obj2)}\n"
        return AssertionError(err)

    def recurse(obj1, obj2, path):
        try:
            if obj1 == obj2:
                return
        except ValueError:
            # Numpy array?
            try:
                if (obj1 == obj2).all():
                    return
            except Exception:
                pass

        if isinstance(obj1, Mapping):
            if not isinstance(obj2, Mapping):
                raise error(
                    f"unequivalent types (expected mappings): "
                    f"{type(obj1).__name__} vs. {type(obj2).__name__}",
                    *[path, obj1, obj2],
                )
            if obj1.keys() != obj2.keys():
                raise error(
                    f"mappings differ in keys: {obj1.keys()} vs. {obj2.keys()}",
                    *[path, obj1, obj2],
                )
            for key, val1 in obj1.items():
                val2 = obj2[key]
                recurse(val1, val2, path + [f"key: {key}"])

        elif isinstance(obj1, Sequence):
            if not isinstance(obj2, Sequence):
                raise error(
                    f"unequivalent types (expected sequences): "
                    f"{type(obj1).__name__} vs. {type(obj2).__name__}",
                    *[path, obj1, obj2],
                )
            if len(obj1) != len(obj2):
                raise error(
                    f"sequences differ in length: {len(obj1)} vs. {len(obj2)}",
                    *[path, obj1, obj2],
                )
            for idx, (ele1, ele2) in enumerate(zip(obj1, obj2)):
                recurse(ele1, ele2, path + [f"idx: {idx}"])

        elif isinstance(obj1, Collection):
            if not isinstance(obj2, Collection):
                raise error(
                    f"unequivalent types (expected collections): "
                    f"{type(obj1).__name__} vs. {type(obj2).__name__}",
                    *[path, obj1, obj2],
                )
            if len(obj1) != len(obj2):
                raise error(
                    f"collections differ in length: {len(obj1)} vs. {len(obj2)}",
                    *[path, obj1, obj2],
                )
            try:
                obj1 = sorted(obj1)
                obj2 = sorted(obj2)
            except Exception:
                raise error(f"unequal collections are unsortable", path, obj1, obj2)
            for idx, (ele1, ele2) in enumerate(zip(obj1, obj2)):
                recurse(ele1, ele2, path + [f"idx: {idx}"])

        elif np.isreal(obj1):
            if not np.isreal(obj2):
                raise error(
                    f"unequivalent types (expected real numbers): "
                    f"{type(obj1).__name__} vs. {type(obj2).__name__}, ",
                    *[path, obj1, obj2],
                )
            if float_close_ok:
                if np.isclose(obj1, obj2, **(kwargs_close or {})):
                    return
                msg = f"unequal floats not even close: {obj1} vs. {obj2}"
                if kwargs_close:
                    msg += " ({})".format(
                        ", ".join([f"{k}={sfmt(v)}" for k, v in kwargs_close.items()]),
                    )
                raise error(msg, path)
            raise error(
                f"unequal floats: {obj1} vs. {obj2} (consider float_close_ok)", path,
            )
        else:
            raise error(f"unequal objects", path, obj1, obj2)

    return recurse(obj1, obj2, path=[])
