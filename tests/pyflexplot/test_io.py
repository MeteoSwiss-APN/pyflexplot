#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyflexplot/io.py` package."""
import pytest
import logging as log
import netCDF4 as nc4
import numpy as np
import os

from utils import datadir

from pyflexplot.io import FlexFieldSpecsConcentration
from pyflexplot.io import FlexFieldSpecsDeposition
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

    species_id = 2

    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time': 3,
    }

    @property
    def kwargs_specs_shared(self):
        return {
            'integrate': False,
            'species_id': self.species_id,
        }

    #------------------------------------------------------------------

    def run(self, datadir, cls_fld_specs, dims, var_names_ref, **kwargs_specs):

        datafile = f'{datadir}/flexpart_cosmo1_case2.nc'

        # Initialize specifications
        var_specs = merge_dicts(self.kwargs_specs_shared, dims, kwargs_specs)
        fld_specs = cls_fld_specs(var_specs)

        # Read input data
        flex_file = FlexFileRotPole(datafile)
        flex_data = flex_file.read(fld_specs)

        # Read reference field
        fld_ref = np.nansum(
            [read_nc_var(datafile, n, dims) for n in var_names_ref], axis=0)
        fld_ref *= 1e-12  #SR_TMP fix magnitude of input field

        assert flex_data.fld.shape == fld_ref.shape
        assert np.allclose(flex_data.fld, fld_ref)

    #------------------------------------------------------------------

    def test_concentration(self, datadir):
        """Read concentration field."""
        var_names_ref = [f'spec{self.species_id:03d}']
        self.run(
            datadir,
            FlexFieldSpecsConcentration,
            {
                **self.dims_shared, 'level': 1
            },
            var_names_ref,
        )

    def test_deposition_dry(self, datadir):
        """Read dry deposition field."""
        var_names_ref = [f'DD_spec{self.species_id:03d}']
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            self.dims_shared,
            var_names_ref,
            deposition='dry',
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        var_names_ref = [f'WD_spec{self.species_id:03d}']
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            self.dims_shared,
            var_names_ref,
            deposition='wet',
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        var_names_ref = [
            f'WD_spec{self.species_id:03d}',
            f'DD_spec{self.species_id:03d}',
        ]
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            self.dims_shared,
            var_names_ref,
            deposition='tot',
        )

    #ipython(globals(), locals())
