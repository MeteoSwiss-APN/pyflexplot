"""
Top-level package for PyFlexPlot.
"""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.13.8"

# Standard library
import logging as pylogging
import sys
import warnings
from pathlib import Path
from typing import List

# Third-party
import matplotlib

# Local
from .utils.logging import add_logging_level

try:
    import cartopy  # isort:skip
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
    print(msg, file=sys.stderr)
    sys.exit(1)


__all__: List[str] = []

verbose_level = 15


pylogging.basicConfig(format="%(message)s")
pylogging.getLogger().handlers = [pylogging.StreamHandler(sys.stdout)]
add_logging_level("verbose", 15)

# Disable third-party debug messages
pylogging.getLogger("matplotlib").setLevel(pylogging.WARNING)


def check_dir_exists(path):
    """Check that a directory exists."""
    if not path.exists():
        raise Exception("data directory is missing", path)
    if not path.is_dir():
        raise Exception("data directory is not a directory", path)


# Set some paths
root_path: Path = Path(__file__).parent
data_path: Path = root_path / "data"
check_dir_exists(data_path)


# Point cartopy to storerd offline data
cartopy.config["pre_existing_data_dir"] = data_path


# Set matplotlib backend
matplotlib.use("Agg")


# pylint: disable=R0913,W0613  # too-many-arguments, unused-argument (line)
def custom_showwarnings(message, category, filename, lineno, file=None, line=None):
    """Show warnings without code excerpt.

    See https://docs.python.org/3/library/warnings.html#warnings.showwarning

    """
    if file is None:
        file = sys.stderr
    key = "src/pyflexplot/"
    if key in filename:
        filename = f"pyflexplot.{filename.split(key)[-1].replace('/', '.')}"
    text = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    file.write(text)


# Custom warnings formatting
warnings.showwarning = custom_showwarnings


# Shorthand to embed IPython shell (handy during development/debugging)
try:
    import IPython  # isort:skip
except ImportError:
    ipy = None
else:
    ipy = IPython.terminal.embed.embed
