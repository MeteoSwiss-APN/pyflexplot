# -*- coding: utf-8 -*-
"""Top-level package for PyPlot."""

__author__ = """Stefan Ruedisuehli"""
__email__ = 'stefan.ruedisuehli@env.ethz.ch'
__version__ = '0.1.0'

import os.path

# Point cartopy to storerd offline data
import cartopy
dir = os.path.dirname(os.path.abspath(__file__))
cartopy_dir = os.path.abspath(f'{dir}/../../cartopy_data')
cartopy.config['pre_existing_data_dir'] = cartopy_dir

# Set up matplotlib
import matplotlib
matplotlib.use('Agg')
# import matplotlib.backend_bases
# import matplotlib.backends.backend_pgf
# matplotlib.use('pdf')
# matplotlib.backend_bases.register_backend(
#     'pdf', matplotlib.backends.backend_pgf.FigureCanvasPgf)
# pgf_with_latex = {
#     "text.usetex": True,  # use LaTeX to write all text
#     "pgf.rcfonts": False,  # ignore matplotlibrc
#     "pgf.preamble": [
#         r'\usepackage{color}',  # xcolor for colors
#     ],
# }
# matplotlib.rcParams.update(pgf_with_latex)
