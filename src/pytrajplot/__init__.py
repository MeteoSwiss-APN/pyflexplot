"""Top-level package for PyTrajPlot."""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.0.1"

# Standard library
import logging as _logging
import sys as _sys
import warnings as _warnings
from pathlib import Path as _Path
from typing import List as _List

# Third-party
import matplotlib as _mpl

# First-party
from pyflexplot.utils.logging import add_logging_level as _add_logging_level

try:
    import cartopy as _cartopy  # isort:skip
except Exception as e:  # pylint: disable=W0703  # broad-except
    if "libproj" in str(e):
        dep = "proj"
    if "libgeos" in str(e):
        dep = "geos"
    else:
        raise
    msg = (
        f"Cannot import 'cartopy': missing external dependency '{dep}'! -- "
        f'{type(e).__name__}("{e}")'
    )
    print(msg, file=_sys.stderr)
    _sys.exit(1)


__all__: _List[str] = []

verbose_level = 15


_logging.basicConfig(format="%(message)s")
_logging.getLogger().handlers = [_logging.StreamHandler(_sys.stdout)]
_add_logging_level("verbose", 15)

# Disable third-party debug messages
_logging.getLogger("matplotlib").setLevel(_logging.WARNING)


def _check_dir_exists(path):
    """Check that a directory exists."""
    if not path.exists():
        raise Exception("data directory is missing", path)
    if not path.is_dir():
        raise Exception("data directory is not a directory", path)


# Set some paths
_data_path = _Path(__file__).parent / "data"
presets_data_path = _data_path / "presets"
earth_data_path = _Path(__file__).parents[1] / "pyflexplot/data/naturalearthdata"
_check_dir_exists(_data_path)
_check_dir_exists(presets_data_path)
_check_dir_exists(earth_data_path)


# Point cartopy to storerd offline data
_cartopy.config["pre_existing_data_dir"] = earth_data_path


# Set matplotlib backend
_mpl.use("Agg")


# pylint: disable=R0913,W0613  # too-many-arguments, unused-argument (line)
def _custom_showwarnings(message, category, filename, lineno, file=None, line=None):
    """Show warnings without code excerpt.

    See https://docs.python.org/3/library/warnings.html#warnings.showwarning

    """
    if file is None:
        file = _sys.stderr
    key = "src/pytrajplot/"
    if key in filename:
        filename = f"pytrajplot.{filename.split(key)[-1].replace('/', '.')}"
    text = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    file.write(text)


# Custom warnings formatting
_warnings.showwarning = _custom_showwarnings


# Shorthand to embed IPython shell (handy during development/debugging)
try:
    import IPython  # isort:skip
except ImportError:
    _ipy = None
else:
    _ipy = IPython.terminal.embed.embed
