# -*- coding: utf-8 -*-
"""Top-level package for PyPlot."""

__author__ = """Stefan Ruedisuehli"""
__email__ = 'stefan.ruedisuehli@env.ethz.ch'
__version__ = '0.1.0'

import os.path
import cartopy

# Point cartopy to storerd offline data
dir = os.path.dirname(os.path.abspath(__file__))
cartopy_dir = os.path.abspath(f'{dir}/../../cartopy_data')
cartopy.config['pre_existing_data_dir'] = cartopy_dir
