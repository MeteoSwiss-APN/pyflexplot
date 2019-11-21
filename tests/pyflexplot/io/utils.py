# -*- coding: utf-8 -*-
"""
Utilities for testing.
"""
import distutils.dir_util
import pytest
import os

from pprint import pformat

from srutils.various import isiterable


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
    test_dir_path = f"{os.path.dirname(test_file_path)}/data"
    if os.path.isdir(test_dir_path):
        distutils.dir_util.copy_tree(test_dir_path, str(tmpdir))
    return tmpdir
