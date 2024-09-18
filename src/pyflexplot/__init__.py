"""Top-level package for PyFlexPlot."""

# Standard library
import os
import importlib.metadata
import logging as _logging
import sys as _sys
import warnings as _warnings
from pathlib import Path as _Path
from typing import List as _List

# Third-party
import matplotlib as _mpl

# Local
from .utils.logging import add_logging_level as _add_logging_level
from .config.service_settings import ServiceSettings

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

__author__ = "Stefan Ruedisuehli"
__email__ = "stefan.ruedisuehli@env.ethz.ch"

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = 'n/a'
    pass

del importlib

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


# Set data paths
_data_path = _Path(__file__).parent / "data"
earth_data_path = _data_path / "naturalearthdata"
presets_data_path = _data_path / "presets"
_check_dir_exists(_data_path)
_check_dir_exists(earth_data_path)
_check_dir_exists(presets_data_path)


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
    key = "src/pyflexplot/"
    if key in filename:
        filename = f"pyflexplot.{filename.split(key)[-1].replace('/', '.')}"
    text = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    file.write(text)


# Custom warnings formatting
_warnings.showwarning = _custom_showwarnings


CONFIG = ServiceSettings(
    'settings.yaml', 
    os.path.join(os.path.dirname(__file__), 'config')
    )
