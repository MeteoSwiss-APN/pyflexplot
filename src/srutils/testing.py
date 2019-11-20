#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some testing utils.
"""

from .various import isiterable

#======================================================================


def property_obj(cls, *args, **kwargs):
    """Define a class property creating a given object on-the-fly.

    The purpose of creating the object on-the-fly in a property method
    is to isolate any errors during instatiation in the test methods
    using the property. This prevents the whole test suite from being
    aborted, as it would if the object were defined as a simple class
    attribute and it's instatiation failed -- instead, only the tests
    attempting to use the object will fail.

    And yes, this is indeed merely a convenience function to save two
    lines of code wherever it is used. :-)

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


#======================================================================


class IgnoredElement:
    """Element that is ignored in comparisons."""

    def __init__(self, description=None):
        self.description = description

    def __str__(self):
        s = type(self).__name__
        if self.description:
            s += f': {self.description}'
        return s


class UnequalElement:
    """Element that is not equal to any other.

    Useful to ensure that a test fails for wrong data.

    """

    def __init__(self, description=None):
        self.description = description

    def __str__(self):
        s = type(self).__name__
        if self.description:
            s += f': {self.description}'
        return s

    def __eq__(self, other):
        return False


def ignored(obj):
    return isinstance(obj, IgnoredElement)


#======================================================================


def assert_summary_dict_is_subdict(
        subdict, superdict, subname='sub', supername='super', i_list=None):
    """Check that one summary dict is a subdict of another."""

    if ignored(subdict) or ignored(superdict):
        return

    if not isinstance(subdict, dict):
        raise AssertionError(
            f"subdict '{subname}' is not a dict, but of type "
            f"{type(subdict).__name__}")

    if not isinstance(superdict, dict):
        raise AssertionError(
            f"superdict '{subpername}' is not a dict, but of type "
            f"{type(superdict).__name__}")

    for i, (key, val_sub) in enumerate(subdict.items()):
        val_super = get_dict_element(
            superdict, key, 'superdict', AssertionError)
        assert_summary_dict_element_is_subelement(
            val_sub, val_super, subname, supername, i_dict=i, i_list=i_list)


def assert_summary_dict_element_is_subelement(
        obj_sub,
        obj_super,
        name_sub='sub',
        name_super='super',
        i_list=None,
        i_dict=None):

    if ignored(obj_sub) or ignored(obj_super):
        return

    if (i_list, i_dict) == (None, None):
        s_i = ''
    elif i_dict is None:
        s_i = f'(list[{i_list}]) '
    elif i_list is None:
        s_i = f'(dict[{i_dict}]) '
    else:
        s_i = f'(list[{i_list}], dict[{i_dict}]) '

    if obj_sub == obj_super:
        return

    elif isinstance(obj_sub, dict):
        assert_summary_dict_is_subdict(
            obj_sub, obj_super, name_sub, name_super, i_list=i_list)

    elif isiterable(obj_sub, str_ok=False):

        if not isiterable(obj_super, str_ok=False):
            raise AssertionError(
                f"superdict element {s_i}not iterable:"
                f"\n\n{pformat(obj_super)}")

        if len(obj_sub) != len(obj_super):
            raise AssertionError(
                f"iterable elements {s_i}differ in size: "
                f"{len(obj_sub)} != {len(obj_super)}"
                f"\n\n{name_super}:\n{pformat(obj_super)}"
                f"\n\n{name_sub}:\n{pformat(obj_sub)}")

        for i, (subobj_sub, subobj_super) in enumerate(zip(obj_sub,
                                                           obj_super)):
            assert_summary_dict_element_is_subelement(
                subobj_sub,
                subobj_super,
                name_sub,
                name_super,
                i_list=i,
                i_dict=i_dict)

    else:
        raise AssertionError(
            f"elements {s_i}differ:\n{name_sub}:\n{obj_sub}\n"
            f"{name_super}:\n{obj_super}")


def get_dict_element(dict_, key, name='dict', exception_type=ValueError):
    """Get an element from a dict, raising an exception otherwise."""
    try:
        return dict_[key]
    except KeyError:
        err = f"key missing in {name}: {key}"
    except TypeError:
        err = f"{name} has wrong type: {type(dict_)}"
    raise exception_type(f"{err}\n\n{name}:\n{pformat(dict_)}")
