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
    file = request.module.__file__
    dir, _ = os.path.splitext(file)
    if os.path.isdir(dir_rel):
        distutils.dir_util.copy_tree(dir, str(tmpdir))
    return tmpdir


@pytest.fixture
def datadir(tmpdir, request):
    """Return path to temporary data directory."""
    file = request.module.__file__
    dir, _ = os.path.splitext(file)
    data_root = os.path.abspath(f"{os.path.abspath(dir)}/../../../../data")
    data_dir = f"{data_root}/pyflexplot/io/reduced"
    if os.path.isdir(data_dir):
        distutils.dir_util.copy_tree(data_dir, str(tmpdir))
    return tmpdir
