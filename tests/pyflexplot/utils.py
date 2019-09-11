# -*- coding: utf-8 -*-
"""Utilities for testing."""
import distutils.dir_util
import pytest
import os

from pyflexplot.utils import isiterable

from pyflexplot.utils_dev import ipython  #SR_DEV

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


def assert_summary_dict_is_subdict(subdict, superdict):
    """Check that one summary dict is a subdict of another."""
    if not isinstance(subdict, dict):
        raise AssertionError(
            f"subdict is not a dict, but of type {type(subdict).__name__}")
    if not isinstance(superdict, dict):
        raise AssertionError(
            f"superdict is not a dict, but of type {type(superdict).__name__}")
    for key, val_sub in subdict.items():
        val_super = get_dict_element(
            superdict, key, 'superdict', AssertionError)
        assert_summary_dict_element_is_subelement(val_sub, val_super)


def assert_summary_dict_element_is_subelement(obj_sub, obj_super):

    if obj_sub == obj_super:
        return

    elif isinstance(obj_sub, dict):
        assert_summary_dict_is_subdict(obj_sub, obj_super)

    elif isiterable(obj_sub, str_ok=False):

        if not isiterable(obj_super, str_ok=False):
            raise AssertionError(
                f"superdict element not iterable:\n\n{pformat(obj_super)}")

        if len(obj_sub) != len(obj_super):
            raise AssertionError(
                f"iterable elements differ in size: {len(obj_sub)} != "
                f"{len(obj_super)}\n\nsuper:\n{obj_super}\n\nsub:\n{obj_sub}")

        for subobj_sub, subobj_super in zip(obj_sub, obj_super):
            assert_summary_dict_element_is_subelement(subobj_sub, subobj_super)

    else:
        raise AssertionError(f"elements differ: {obj_sub} != {obj_super}")


def get_dict_element(dict_, key, name='dict', exception_type=ValueError):
    """Get an element from a dict, raising an exception otherwise."""
    try:
        return dict_[key]
    except KeyError:
        err = f"key missing in {name}: {key}"
    except TypeError:
        err = f"{name} has wrong type: {type(dict_)}"
    raise exception_type(f"{err}\n\n{name}:\n{pformat(dict_)}")
