# -*- coding: utf-8 -*-
"""
Shared functions and classes for tests of elements of complete plots.
"""
# Standard library
import distutils.dir_util
import importlib
import os
import re
from textwrap import dedent
from typing import Any
from typing import Dict

# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.io import read_fields
from pyflexplot.plot import plot_fields
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection
from srutils.testing import assert_nested_equal

try:
    import black
except ImportError:
    black = None


PACKAGE = "test_plots"


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


class _TestBase:
    """Base class to test complete plots for their elements.

    Notes:
     *  The pytest fixture ``datadir`` from this module must be imported by any
        modules in which ``_TestBase`` is subclassed because it is required by
        ``_TestBase.test`` even implicitly if the latter is not overridden!

        In case ``datadir`` is not explicitly used, make sure to guard the
        seemingly obsolete import from linters with a ``noqa`` directive:

            from .shared import datadir  # noqa:F401

     *  Test reference files, as specified by the class attribute ``reference``,
        are created by changing the parent module of a test class temporarily
        from ``_TestBase`` to ``_TestCreateReference`` and running the tests (which
        will fail with ``ReferenceFileCreationSuccess`` if all goes well).

        This creates a file for each test (``<reference>.py``) in the current
        directory. Move these new reference files to the appropriate directory
        under ``tests`` to replace the old ones.

    """

    reference: str
    setup_dct: Dict[str, Any]

    def get_setups(self):
        setup = InputSetup(**self.setup_dct)
        return InputSetupCollection([setup])

    def get_field_and_mdata(self, datadir):
        infile = f"{datadir}/{self.setup_dct['infile']}"

        setups = self.get_setups()
        var_setups_lst = setups.decompress_grouped_by_time()

        fields, mdata_lst = read_fields(infile, var_setups_lst, add_ts0=True)
        assert len(fields) == len(mdata_lst) == 1
        field = next(iter(fields))
        mdata = next(iter(mdata_lst))

        return field, mdata

    def get_plot(self, field, mdata):
        outfiles, plots = zip(*plot_fields([field], [mdata], write=False))
        assert len(outfiles) == len(plots) == 1
        plot = next(iter(plots))
        return plot

    def get_reference(self, key=None):
        mod = importlib.import_module(f"{PACKAGE}.{self.reference}")
        ref = {"field_summary": mod.field_summary, "plot_summary": mod.plot_summary}
        if key is not None:
            return ref[key]
        return ref

    def test(self, datadir):
        field, mdata = self.get_field_and_mdata(datadir)
        res = field.summarize()
        sol = self.get_reference("field_summary")
        try:
            assert_nested_equal(res, sol, float_close_ok=True)
        except AssertionError as e:
            msg = f"field summaries differ (result vs. solution):\n\n {e}"
            raise AssertionError(msg) from None

        plot = self.get_plot(field, mdata)
        res = plot.summarize()
        sol = self.get_reference("plot_summary")
        try:
            assert_nested_equal(res, sol, float_close_ok=True)
        except AssertionError as e:
            msg = f"plot summaries differ (result vs. solution):\n\n{e}"
            raise AssertionError(msg) from None


class ReferenceFileCreationSuccess(Exception):
    """Test reference file successfully written to disk."""


class ReferenceFileCreationError(Exception):
    """Error writing test reference file to disk."""


class _TestCreateReference(_TestBase):
    """Test parent class to create a new test reference for a test."""

    def test(self, datadir):
        if black is None:
            raise ImportError("must install black to create test reference")
        ref_file = f"{self.reference}.py"
        field, mdata = self.get_field_and_mdata(datadir)
        field_summary = field.summarize()
        plot = self.get_plot(field, mdata)
        plot_summary = plot.summarize()
        module_path_rel = os.path.relpath(__file__, ".")
        cls_name = type(self).__name__
        content = f'''\
            # -*- coding: utf-8 -*-
            # flake8: noqa
            """
            Test reference for pytest test.

            {module_path_rel}
                ::{cls_name}
                ::test

            Created by temporarily changing the parent class of
            ``{cls_name}``
            from ``_TestBase`` to ``_TestCreateReference`` and running pytest.
            """
            import numpy as np

            field_summary = {field_summary}

            plot_summary = {plot_summary}
            '''
        content = dedent(content)
        # Replace non-importable nan/inf objects by np.nan/np.inf
        content_np = re.sub(r": (-?)\b(nan|inf)\b", r": \1np.\2", content)
        if content_np != content:
            content = content_np
        else:
            # No np.nan/np.inf: Remove obsolete numpy import
            content = content.replace("import numpy as np\n", "")
        content = black_format(content)
        try:
            with open(ref_file, "w") as f:
                f.write(content)
        except Exception:
            raise ReferenceFileCreationError(ref_file)
        else:
            raise ReferenceFileCreationSuccess(ref_file)


def black_format(code: str) -> str:
    """Format a string of code with black."""
    return str(black.format_str(code, mode=black.FileMode()))


class PlotCreationSuccess(Exception):
    """Test plot successfully written to disk."""


class PlotCreationError(Exception):
    """Error writing test plot to disk."""


class _TestCreatePlot(_TestBase):
    """Test parent class to create plots based on a test setup."""

    def test(self, datadir):
        plot_file = f"{self.reference}.png"
        field, mdata = self.get_field_and_mdata(datadir)
        plot = self.get_plot(field, mdata)
        try:
            plot.save(plot_file)
        except Exception:
            raise PlotCreationError(plot_file)
        else:
            raise PlotCreationSuccess(plot_file)


# Uncomment to create test references for all tests at once
# _TestBase = _TestCreateReference

# Uncomment to create test plots for all tests at once
# _TestBase = _TestCreatePlot
