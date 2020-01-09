# -*- coding: utf-8 -*-
"""Top-level package for PyFlexPlot."""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.1.1"

import sys
import os.path

try:
    import cartopy
except ImportError as e:
    if "libproj" in str(e):
        msg = f"cannot import 'cartopy': external dependency 'proj' missing ({e})"
    if "libgeos" in str(e):
        msg = f"cannot import 'cartopy': expernal dependency 'geos' missing ({e})"
    else:
        raise
    print(msg, file=sys.stderr)
    sys.exit(1)

# Point cartopy to storerd offline data
dir = os.path.dirname(os.path.abspath(__file__))
cartopy_dir = os.path.abspath(f"{dir}/cartopy_data")
cartopy.config["pre_existing_data_dir"] = cartopy_dir

# Set matplotlib backend
import matplotlib

matplotlib.use("Agg")
