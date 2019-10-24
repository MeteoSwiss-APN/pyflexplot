# -*- coding: utf-8 -*-
"""
Utilities for testing.
"""
import distutils.dir_util
import pytest
import os

from pprint import pformat

from pyflexplot.utils import isiterable

#======================================================================


@pytest.fixture
def datadir_rel(tmpdir, request):
    """Return path to temporary data directory named like test file.

    Pytest fixture to find a data folder with the same name as the test
    module and -- if found -- mirror it to a temporary directory, which
    allows the tests to use the data files freely, even in parallel.

    Adapted from `https://stackoverflow.com/a/29631801`.
    """
    test_file_path = request.module.__file__
    test_dir_path, _ = os.path.splitext(test_file_path)
    if os.path.isdir(test_dir_path):
        distutils.dir_util.copy_tree(test_dir_path, str(tmpdir))
    return tmpdir


@pytest.fixture
def datadir(tmpdir, request):
    """Return path to temporary directory named 'data'."""
    test_file_path = request.module.__file__
    test_dir_path = f'{os.path.dirname(test_file_path)}/data'
    if os.path.isdir(test_dir_path):
        distutils.dir_util.copy_tree(test_dir_path, str(tmpdir))
    return tmpdir


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


def ignored(obj):
    return isinstance(obj, IgnoredElement)


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
        obj_sub, obj_super, name_sub='sub', name_super='super', i_list=None,
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
                "{len(obj_sub)} != {len(obj_super)}"
                f"\n\n{name_super}:\n{obj_super}\n\n{name_sub}:\n{obj_sub}")

        for i, (subobj_sub, subobj_super) in enumerate(zip(obj_sub, obj_super)):
            assert_summary_dict_element_is_subelement(
                subobj_sub, subobj_super, name_sub, name_super, i_list=i,
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
