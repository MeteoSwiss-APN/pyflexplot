# -*- coding: utf-8 -*-
"""
Top-level package for PyFlexPlot.
"""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.6.2"

# Standard library
import os.path
import sys
import warnings
from typing import Any
from typing import List

# Third-party
import matplotlib

try:
    import cartopy  # isort:skip
except Exception as e:
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

__all__: List[Any] = []


# Format warning messages (remove code)
# See https://docs.python.org/3/library/warnings.html#warnings.showwarning
def custom_showwarnings(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stderr
    key = "src/pyflexplot/"
    if key in filename:
        filename = f"pyflexplot.{filename.split(key)[-1].replace('/', '.')}"
    text = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    file.write(text)


warnings.showwarning = custom_showwarnings

# Point cartopy to storerd offline data
here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(f"{here}/data")
if not os.path.isdir(data_dir):
    raise Exception(f"data directory missing: {data_dir}")
cartopy.config["pre_existing_data_dir"] = data_dir

# Set matplotlib backend
matplotlib.use("Agg")
