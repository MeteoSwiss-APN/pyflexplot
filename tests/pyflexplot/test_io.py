#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyflexplot/io.py` package."""
import pytest
import logging as log
import netCDF4 as nc4
import numpy as np
import os

from utils import datadir

from pyflexplot.io import FlexFieldSpecs
from pyflexplot.io import FlexFileRotPole

from pyflexplot.utils import merge_dicts

from pyflexplot.utils_dev import ipython  #SR_DBG

def read_nc_var(path, var_name, dim_inds):
    with nc4.Dataset(path, 'r') as fi:
        var = fi.variables[var_name]
        inds = [dim_inds.get(d, slice(None)) for d in var.dimensions]
        return var[inds]

def test_concentration_TODO(datadir):
    datafile = f'{datadir}/flexpart_cosmo1_case2.nc'

    dims = {
        'nageclass': 0,
        'numpoint': 0,
        'time': 3,
        'level': 1,
    }

    kwargs_specs = {
        'prefix': None,
        'integrate': False,
        'species_id': 2,
    }

    kwargs_specs.update(dims)
    field_specs = FlexFieldSpecs(**kwargs_specs)

    flex_file = FlexFileRotPole(datafile)
    flex_data = flex_file.read(field_specs)

    var_name = f'spec{kwargs_specs["species_id"]:03d}'
    fld_ref = read_nc_var(datafile, var_name, dims)

    assert flex_data.field.shape == fld_ref.shape
    assert np.allclose(flex_data.field, fld_ref*1e-12)

    #ipython(globals(), locals())
