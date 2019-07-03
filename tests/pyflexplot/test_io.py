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


class TestReadFieldSimple:

    #------------------------------------------------------------------

    @property
    def species_id(self):
        return 2

    @property
    def dims(self):
        return {
            'nageclass': 0,
            'numpoint': 0,
            'time': 3,
            'level': 1,
        }

    @property
    def kwargs_specs_shared(self):
        return {
            #'prefix':
            'integrate': False,
            'species_id': self.species_id,
            **self.dims,
        }

    #------------------------------------------------------------------

    def run(self, datadir, var_name, **kwargs_specs):

        datafile = f'{datadir}/flexpart_cosmo1_case2.nc'

        # Initialize specifications
        field_specs = FlexFieldSpecs(
            **merge_dicts(self.kwargs_specs_shared, kwargs_specs))

        # Read input data
        flex_file = FlexFileRotPole(datafile)
        flex_data = flex_file.read(field_specs)

        # Read reference field
        fld_ref = read_nc_var(datafile, var_name, self.dims)
        fld_ref *= 1e-12  #SR_TMP fix magnitude of input field

        assert flex_data.field.shape == fld_ref.shape
        assert np.allclose(flex_data.field, fld_ref)

    #------------------------------------------------------------------

    def test_concentration(self, datadir):
        """Read concentration field."""
        var_name = f'spec{self.species_id:03d}'
        self.run(datadir, var_name, prefix=None)

    def test_deposition_dry(self, datadir):
        """Read dry deposition field."""
        var_name = f'DD_spec{self.species_id:03d}'
        self.run(datadir, var_name, prefix='DD')

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        var_name = f'WD_spec{self.species_id:03d}'
        self.run(datadir, var_name, prefix='WD')

    #ipython(globals(), locals())
