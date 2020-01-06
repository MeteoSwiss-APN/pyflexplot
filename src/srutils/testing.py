#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some testing utils.
"""
from dataclasses import dataclass
from pprint import pformat
from typing import Optional

from .str import str_or_none
from .various import isiterable


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


@dataclass(frozen=True)
class IgnoredElement:
    """Element that is ignored in comparisons."""

    description: Optional[str] = None

    def __repr__(self):
        return f"{type(self).__name__}({str_or_none(self.description)})"


@dataclass(frozen=True)
class UnequalElement:
    """Element unequal to any other; useful for force-fail tests."""

    description: Optional[str] = None

    def __repr__(self):
        return f"{type(self).__name__}({str_or_none(self.description)})"

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

    def return_or_raise(msg, kwargs=None):
        if not raise_:
            return False
        if kwargs is None:
            kwargs = {}
        kwargs = {
            "obj": obj,
            "len_": len_,
            "not_": not_,
            "t_children": t_children,
            "f_children": f_children,
            "raise_": raise_,
            **kwargs,
        }
        raise Exception(msg, kwargs)

    if not isiterable(obj, str_ok=False):
        return_or_raise(f"{type(obj).__name__} instance `obj` is not iterable")

    if len_ is not None:
        if len(obj) != len_:
            return_or_raise(f"obj has wrong length {len(obj)}")

    if not_ is not None:
        if isinstance(obj, not_):
            return_or_raise(f"obj has unexpected type {type(obj).__name__}")

    if t_children is not None:
        for idx, child in enumerate(obj):
            if not isinstance(child, t_children):
                return_or_raise(
                    f"child has unexpected type {type(child).__name__}",
                    {"child": child, "idx": idx},
                )

    if f_children is not None:
        for idx, child in enumerate(obj):
            if not f_children(child):
                return_or_raise(
                    f"f_children returns False for child", {"child": child, "idx": idx},
                )

    return True
