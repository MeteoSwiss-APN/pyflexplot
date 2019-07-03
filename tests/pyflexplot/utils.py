# -*- coding: utf-8 -*-
"""Utilities for testing."""
import distutils.dir_util
import pytest
import os


@pytest.fixture
def datadir(tmpdir, request):
    """Return path to temporary directory containint test data.

    Pytest fixture to find a data folder with the same name as the test
    module and -- if found -- mirror it to a temporary directory, which
    allows the tests to use the data files freely, even in parallel.

    Adapted from `https://stackoverflow.com/a/29631801`.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        distutils.dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
