"""Shared functions and classes for tests of elements of complete plots."""
# Standard library
import distutils.dir_util
import importlib
import logging
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Dict

# Third-party
import pytest  # type: ignore

# First-party
from pyflexplot.input.read_fields import read_fields
from pyflexplot.plots import create_plot
from pyflexplot.plots import format_out_file_paths
from pyflexplot.setups.plot_setup import PlotSetup
from pyflexplot.setups.plot_setup import PlotSetupGroup
from pyflexplot.utils.summarize import summarize
from srutils.testing import assert_nested_equal

# Black is only required to create test reference files, not to run the tests
try:
    # Third-party
    import black
except ImportError:
    black = None  # type: ignore

logging.getLogger().setLevel(logging.ERROR)

PACKAGE = "test_plots"


@pytest.fixture
def datadir(tmpdir, request):
    """Return path to temporary data directory."""
    data_root = Path(__file__).parents[3] / "data"
    # data_dir = data_root / "pyflexplot/flexpart/original"
    data_dir = data_root / "pyflexplot/flexpart/reduced"
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
        from ``_TestBase`` to ``_TestCreateReference`` and running the tests
        (which fail with ``ReferenceFileCreationSuccess`` if all goes well).

        This creates a file for each test (``<reference>.py``) in the current
        directory. Move these new reference files to the appropriate directory
        under ``tests`` to replace the old ones.

    """

    reference: str
    setup_dct: Dict[str, Any]
    n_plots: int = 1

    def get_setups(self):
        setup = PlotSetup.create(self.setup_dct)
        return PlotSetupGroup([setup])

    def get_field_group(self, datadir):
        infile = f"{datadir}/{self.setup_dct['files']['input']}"
        setups = self.get_setups()
        field_groups = read_fields(
            setups, {"add_ts0": True, "missing_ok": True}, _override_infile=infile
        )
        assert len(field_groups) == self.n_plots
        # SR_TMP <
        assert self.n_plots == 1
        field_group = next(iter(field_groups))
        # SR_TMP >
        return field_group

    def get_plot(self, field_group):
        outfiles, plots = [], []
        outfiles = format_out_file_paths(field_group, prev_paths=outfiles)
        plot = create_plot(field_group, outfiles, write=False, show_version=False)
        plots.append(plot)
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
        field_group = self.get_field_group(datadir)

        res = summarize(field_group)
        sol = self.get_reference("field_summary")
        try:
            assert_nested_equal(res, sol, "res", "sol", float_close_ok=True)
        except AssertionError as e:
            msg = f"field summaries differ (result vs. solution):\n\n {e}"
            raise AssertionError(msg)

        plot = self.get_plot(field_group)
        res = summarize(plot)
        plot.clean()
        sol = self.get_reference("plot_summary")
        try:
            assert_nested_equal(res, sol, "res", "sol", float_close_ok=True)
        except AssertionError as e:
            msg = f"plot summaries differ (result vs. solution):\n\n{e}"
            raise AssertionError(msg)


class ReferenceFileCreationSuccess(Exception):
    """Test reference file successfully written to disk."""


class ReferenceFileCreationError(Exception):
    """Error writing test reference file to disk."""


class _TestCreateReference(_TestBase):
    """Replacement parent class for test classes to create new reference.

    For test classes that (temporarily) inherit from ``_TestCreateReference``
    instead of ``_TestBase``, a new test reference file will be created when
    the respective test is run.

    Upon successful reference file creation, ``ReferenceFileCreationSuccess`` is
    raised, otherwise ``ReferenceFileCreationError``.

    To check a newly created reference, the corresponding plot can be created
    by temporarily inheriting from ``_TestCreatePlot``.

    """

    def test(self, datadir):
        if black is None:
            raise ImportError("must install black to create test reference")
        target_dir = Path(__file__).resolve()().parent
        ref_file = target_dir / f"{self.reference}.py"  # pylint: disable=no-member
        field_group = self.get_field_group(datadir)
        plot = self.get_plot(field_group)

        field_summary = summarize(field_group)
        plot_summary = summarize(plot)
        plot.clean()

        module_path_rel = os.path.relpath(__file__, ".")
        cls_name = type(self).__name__
        head = f'''\
            # flake8: noqa
            """Test reference for pytest test.

            {module_path_rel}
                ::{cls_name}
                ::test

            Created by temporarily changing the parent class of
            ``{cls_name}``
            from ``_TestBase`` to ``_TestCreateReference`` and running pytest.

            """
            '''
        body = f"""\
            field_summary = {field_summary}

            plot_summary = {plot_summary}
            """
        head = dedent(head)
        body = dedent(body)

        # Replace non-importable nan/inf objects by np.nan/np.inf
        body_np = re.sub(r"(?<!['\"])(?<!np\.)\b(nan|inf)\b(?!['\"])", r"np.\1", body)
        if body_np != body:
            body = body_np
            head += "# Third-party\nimport numpy as np"

        content = black_format(f"{head}\n{body}")
        try:
            with open(ref_file, "w") as f:
                f.write(content)
        except Exception:
            raise ReferenceFileCreationError(ref_file)
        else:
            raise ReferenceFileCreationSuccess(ref_file)


def black_format(code: str) -> str:
    """Format a string of code with black."""
    try:
        return str(black.format_str(code, mode=black.FileMode()))
    except black.InvalidInput as e:
        msg = str(e)
        if len(msg) > 400:
            msg = f"{msg[:150]} ... {msg[-150:]}"
        raise Exception("black cannot format code: {msg}") from None


class PlotCreationSuccess(Exception):
    """Test plot successfully written to disk."""


class PlotCreationError(Exception):
    """Error writing test plot to disk."""


class _TestCreatePlot(_TestBase):
    """Replacement parent class for test classes to create plot.

    For test classes that (temporarily) inherit from ``_TestCreatePlot`` instead
    of ``_TestBase``, the plot will be created when the respective test is run.

    Upon successful reference file creation, ``PlotCreationSuccess`` is raised,
    otherwise ``PlotCreationError``.

    If the plot looks good, a new test reference can be created by temporarily
    inheriting from ``_TestCreateReference``.

    """

    def test(self, datadir):
        # target_dir = Path(__file__).resolve()().parent
        target_dir = Path(".")
        plot_file = target_dir / f"{self.reference}.png"  # pylint: disable=no-member
        field_group = self.get_field_group(datadir)
        plot = self.get_plot(field_group)
        try:
            plot.write(plot_file)
        except Exception:
            raise PlotCreationError(plot_file)
        else:
            raise PlotCreationSuccess(plot_file)


# Uncomment to create test references for all tests at once
# _TestBase = _TestCreateReference

# Uncomment to create test plots for all tests at once
# _TestBase = _TestCreatePlot
