# -*- coding: utf-8 -*-
"""Top-level package for PyFlexPlot."""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.1.2"

import sys
import os.path
import matplotlib

try:
    import cartopy
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

__all__ = []

# Point cartopy to storerd offline data
here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(f"{here}/data")
if not os.path.isdir(data_dir):
    raise Exception(f"data directory missing: {data_dir}")
cartopy.config["pre_existing_data_dir"] = data_dir

# Set matplotlib backend
matplotlib.use("Agg")
