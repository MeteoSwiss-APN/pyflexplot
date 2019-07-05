#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyflexplot/io.py` package."""
import itertools
import logging as log
import netCDF4 as nc4
import numpy as np
import os
import pytest

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


class TestReadFieldSingle:
    """Read a single 2D field from a FLEXPART NetCDF file."""

    # Dimensions shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time': 3,
    }

    # Variable specifications shared by all tests
    var_specs_shared = {
        'integrate': False,
        'species_id': 2,
    }

    @property
    def species_id(self):
        return self.var_specs_shared['species_id']

    def datafile(self, datadir):
        return f'{datadir}/flexpart_cosmo1_case2.nc'

    #------------------------------------------------------------------

    def run(
            self, datadir, cls_fld_specs, dims, var_names_ref,
            var_specs_unshared):
        """Run an individual test."""

        datafile = self.datafile(datadir)

        # Initialize specifications
        var_specs = merge_dicts(
            self.var_specs_shared, dims, var_specs_unshared)
        fld_specs = cls_fld_specs(var_specs)

        # Read input data
        flex_file = FlexFileRotPole(datafile)
        flex_data = flex_file.read(fld_specs)

        # Read reference field
        fld_ref = np.nansum(
            [read_nc_var(datafile, n, dims) for n in var_names_ref], axis=0)
        fld_ref *= 1e-12  #SR_TMP fix magnitude of input field

        # Check array
        assert flex_data.fld.shape == fld_ref.shape
        assert np.allclose(flex_data.fld, fld_ref)

    #------------------------------------------------------------------

    def test_concentration(self, datadir):
        """Read concentration field."""
        self.run(
            datadir,
            FlexFieldSpecsConcentration,
            dims={
                **self.dims_shared, 'level': 1
            },
            var_names_ref=[f'spec{self.species_id:03d}'],
            var_specs_unshared={},
        )

    def test_deposition_dry(self, datadir):
        """Read dry deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_ref=[f'DD_spec{self.species_id:03d}'],
            var_specs_unshared={'deposition': 'dry'},
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_ref=[f'WD_spec{self.species_id:03d}'],
            var_specs_unshared={'deposition': 'wet'},
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_ref=[
                f'WD_spec{self.species_id:03d}',
                f'DD_spec{self.species_id:03d}',
            ],
            var_specs_unshared={'deposition': 'tot'},
        )


class TestReadFieldMultiple:
    """Read multiple 2D fields from a FLEXPART NetCDF file."""

    # Dimensions arguments shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time_lst': [0, 3, 4, 5, 9],
    }

    # Variables specification arguments shared by all tests
    var_specs_shared = {
        'integrate_lst': [False, True],
        'species_id_lst': [1, 2],
    }

    @property
    def species_id_lst(self):
        return self.var_specs_shared['species_id_lst']

    def datafile(self, datadir):
        return f'{datadir}/flexpart_cosmo1_case2.nc'

    #------------------------------------------------------------------

    def create_field_specs_lst_ref(self, cls_fld_specs, vars_specs):

        # Create all variable specifications combinations
        keys, vals = [], []
        for key, val in sorted(vars_specs.items()):
            keys.append(key.replace('_lst', ''))
            vals.append(val if key.endswith('_lst') else [val])
        vals_prod = itertools.product(*vals)
        var_specs_lst = [dict(zip(keys, vals_i)) for vals_i in vals_prod]

        # Create field specifications
        field_specs_lst = [cls_fld_specs(vs) for vs in var_specs_lst]

        return field_specs_lst

    #------------------------------------------------------------------

    def test_field_specs_concentration(self):

        # Create field specs
        vars_specs = {
            **self.dims_shared,
            **self.var_specs_shared,
            'level_lst': [0, 2],
        }
        field_specs_lst = FlexFieldSpecsConcentration.multiple(vars_specs)

        # Create reference field specs
        field_specs_lst_ref = self.create_field_specs_lst_ref(
            FlexFieldSpecsConcentration, vars_specs)

        assert sorted(field_specs_lst) == sorted(field_specs_lst_ref)

    def test_field_specs_deposition(self):

        # Create field specs
        vars_specs = {
            **self.dims_shared,
            **self.var_specs_shared,
            'deposition_lst': ['wet', 'dry', 'tot'],
        }
        field_specs_lst = FlexFieldSpecsDeposition.multiple(vars_specs)

        # Create reference field specs
        field_specs_lst_ref = self.create_field_specs_lst_ref(
            FlexFieldSpecsDeposition, vars_specs)

        assert sorted(field_specs_lst) == sorted(field_specs_lst_ref)

    #------------------------------------------------------------------

    def run(
            self, datadir, cls_fld_specs, dims, var_names_lst_ref,
            var_specs_unshared):

        datafile = self.datafile(datadir)

    #------------------------------------------------------------------

    def test_concentration(self, datadir):
        """Read concentration fields."""
        self.run(
            datadir,
            FlexFieldSpecsConcentration,
            dims={
                **self.dims_shared,
                'level_lst': [0, 2],
            },
            var_names_lst_ref=[[
                f'spec{species_id:03d}',
            ] for species_id in self.species_id_lst],
            var_specs_unshared={},
        )

    def test_deposition_dry(self, datadir):
        """Read dry deposition fields."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_lst_ref=[[
                f'DD_spec{species_id:03d}',
            ] for species_id in self.species_id_lst],
            var_specs_unshared={'deposition': 'dry'},
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_lst_ref=[[
                f'WD_spec{species_id:03d}',
            ] for species_id in self.species_id_lst],
            var_specs_unshared={'deposition': 'wet'},
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_lst_ref=[[
                f'WD_spec{species_id:03d}',
                f'DD_spec{species_id:03d}',
            ] for species_id in self.species_id_lst],
            var_specs_unshared={'deposition': 'tot'},
        )
