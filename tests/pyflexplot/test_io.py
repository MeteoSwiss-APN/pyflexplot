#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyflexplot/io.py` package."""
import logging as log
import netCDF4 as nc4
import numpy as np
import os
import pytest

from utils import datadir

from pyflexplot.io import FlexFieldSpecsConcentration
from pyflexplot.io import FlexFieldSpecsDeposition
from pyflexplot.io import FlexFileRotPole

from pyflexplot.utils import dict_mult_vals_product

from pyflexplot.utils_dev import ipython  #SR_DEV


#SR_TMP<<<
def fix_nc_fld(fld):
    """Fix field read directly from NetCDF file."""
    fld[:] *= 1e-12


def read_nc_var(path, var_name, dim_inds):
    with nc4.Dataset(path, 'r') as fi:
        var = fi.variables[var_name]
        inds = []
        for name in var.dimensions:
            if name in ['rlat', 'rlon']:
                inds.append(slice(*dim_inds.get(name, [None])))
            else:
                inds.append(dim_inds.get(name, slice(None)))
        fld = var[inds]
        fix_nc_fld(fld)  #SR_TMP
        return fld


class TestReadFieldSingle:
    """Read a single 2D field from a FLEXPART NetCDF file."""

    # Dimensions shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time': 3,
    }

    # Variable specifications shared by all tests
    var_specs_mult_shared = {
        'integrate': False,
        'species_id': 2,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared['species_id']

    @staticmethod
    def datafile(datadir):
        return f'{datadir}/flexpart_cosmo1_case2.nc'

    #------------------------------------------------------------------

    def run(
            self, datadir, cls_fld_specs, dims, var_names_ref,
            var_specs_mult_unshared):
        """Run an individual test."""

        # Initialize specifications
        var_specs = {
            **dims,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs = cls_fld_specs(var_specs)

        # Read input field
        flex_field = FlexFileRotPole(self.datafile(datadir)).read(fld_specs)
        fld = flex_field.fld

        # Read reference field
        fld_ref = np.nansum(
            [
                read_nc_var(self.datafile(datadir), var_name, dims)
                for var_name in var_names_ref
            ],
            axis=0,
        )

        # Check array
        assert fld.shape == fld_ref.shape
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)

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
            var_specs_mult_unshared={},
        )

    def test_deposition_dry(self, datadir):
        """Read dry deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_ref=[f'DD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'dry'},
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datadir,
            FlexFieldSpecsDeposition,
            dims=self.dims_shared,
            var_names_ref=[f'WD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'wet'},
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
            var_specs_mult_unshared={'deposition': 'tot'},
        )


class TestFieldSpecsMultiple:
    """Create multiple field specifications."""

    # Dimensions arguments shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time_lst': [0, 3, 4, 5, 9],
    }

    # Variables specification arguments shared by all tests
    var_specs_mult_shared = {
        'integrate_lst': [False, True],
        'species_id_lst': [1, 2],
    }

    @property
    def species_id_lst(self):
        return self.var_specs_mult_shared['species_id_lst']

    #------------------------------------------------------------------

    def create_fld_specs_mult_lst_ref(self, cls_fld_specs, vars_specs_mult):

        # Create all variable specifications combinations
        var_specs_lst = dict_mult_vals_product(vars_specs_mult)

        # Create field specifications
        fld_specs_mult_lst = [cls_fld_specs(vs) for vs in var_specs_lst]

        return fld_specs_mult_lst

    #------------------------------------------------------------------

    def test_concentration(self):

        # Create field specifications list
        var_specs_mult = {
            **self.dims_shared,
            **self.var_specs_mult_shared,
            'level_lst': [0, 2],
        }
        fld_specs_mult_lst = FlexFieldSpecsConcentration.multiple(
            var_specs_mult)

        # Create reference field specifications list
        fld_specs_mult_lst_ref = self.create_fld_specs_mult_lst_ref(
            FlexFieldSpecsConcentration, var_specs_mult)

        assert sorted(fld_specs_mult_lst) == sorted(fld_specs_mult_lst_ref)

    def test_deposition(self):

        # Create field specifications list
        var_specs_mult = {
            **self.dims_shared,
            **self.var_specs_mult_shared,
            'deposition_lst': ['wet', 'dry', 'tot'],
        }
        fld_specs_mult_lst = FlexFieldSpecsDeposition.multiple(var_specs_mult)

        # Create reference field specifications list
        fld_specs_mult_lst_ref = self.create_fld_specs_mult_lst_ref(
            FlexFieldSpecsDeposition, var_specs_mult)

        assert sorted(fld_specs_mult_lst) == sorted(fld_specs_mult_lst_ref)


class TestReadFieldMultiple:
    """Read multiple 2D fields from a FLEXPART NetCDF file."""

    # Dimensions arguments shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time_lst': [0, 3, 9],
    }

    # Variables specification arguments shared by all tests
    var_specs_mult_shared = {
        'integrate': True,
        'species_id': 1,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared['species_id']

    @staticmethod
    def datafile(datadir):
        return f'{datadir}/flexpart_cosmo1_case2.nc'

    #------------------------------------------------------------------

    def run(
            self, *, separate, datafile, cls_fld_specs, dims_mult,
            var_names_ref, var_specs_mult_unshared):
        """Run an individual test, reading one field after another."""

        # Create field specifications list
        var_specs_mult = {
            **dims_mult,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs_lst = cls_fld_specs.multiple(var_specs_mult)

        dim_names = sorted([d.replace('_lst', '') for d in dims_mult.keys()])

        if separate:
            # Process field specifications one after another
            for fld_specs in fld_specs_lst:
                self._run_core(datafile, dim_names, var_names_ref, [fld_specs])
        else:
            self._run_core(datafile, dim_names, var_names_ref, fld_specs_lst)

    @staticmethod
    def _run_core(datafile, dim_names, var_names_ref, fld_specs_lst):

        # Read input fields
        flex_field_lst = (FlexFileRotPole(datafile).read(fld_specs_lst))
        flds = np.array([flex_field.fld for flex_field in flex_field_lst])

        # Collect dimensions
        dims_lst = []
        for fld_specs in fld_specs_lst:
            dims = {}
            for name in dim_names:
                var_specs = fld_specs.var_specs_merged()
                if name == 'time' and var_specs['integrate']:
                    dims[name] = slice(None, var_specs[name] + 1)
                else:
                    dims[name] = var_specs[name]
            dims_lst.append(dims)

        # Read reference fields
        flds_ref = []
        for dims in dims_lst:
            flds_ref_i = [
                read_nc_var(datafile, var_name, dims)
                for var_name in var_names_ref
            ]
            if isinstance(dims['time'], slice):
                flds_ref_i = np.nansum(flds_ref_i, axis=1)
            fld_ref_i = np.nansum(flds_ref_i, axis=0)
            flds_ref.append(fld_ref_i)
        flds_ref = np.array(flds_ref)

        assert flds.shape == flds_ref.shape
        assert np.isclose(np.nanmean(flds), np.nanmean(flds_ref))
        np.testing.assert_allclose(flds, flds_ref, equal_nan=True, rtol=1e-6)

    #------------------------------------------------------------------

    def run_concentration(self, datadir, *, separate):
        """Read multiple concentration fields."""
        self.run(
            separate=separate,
            datafile=self.datafile(datadir),
            cls_fld_specs=FlexFieldSpecsConcentration,
            dims_mult={
                **self.dims_shared,
                'level_lst': [0, 2],
            },
            var_names_ref=[f'spec{self.species_id:03d}'],
            var_specs_mult_unshared={},
        )

    def test_concentration_separate(self, datadir):
        self.run_concentration(datadir, separate=True)

    def test_concentration_together(self, datadir):
        self.run_concentration(datadir, separate=False)

    #------------------------------------------------------------------

    def run_deposition_dry(self, datadir, *, separate):
        """Read dry deposition fields."""
        self.run(
            separate=separate,
            datafile=self.datafile(datadir),
            cls_fld_specs=FlexFieldSpecsDeposition,
            dims_mult=self.dims_shared,
            var_names_ref=[f'DD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'dry'},
        )

    def test_deposition_dry_separate(self, datadir):
        self.run_deposition_dry(datadir, separate=True)

    def test_deposition_dry_together(self, datadir):
        self.run_deposition_dry(datadir, separate=False)

    #------------------------------------------------------------------

    def run_deposition_wet(self, datadir, *, separate):
        """Read wet deposition field."""
        self.run(
            separate=separate,
            datafile=self.datafile(datadir),
            cls_fld_specs=FlexFieldSpecsDeposition,
            dims_mult=self.dims_shared,
            var_names_ref=[f'WD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'wet'},
        )

    def test_deposition_wet_separate(self, datadir):
        self.run_deposition_wet(datadir, separate=True)

    def test_deposition_wet_together(self, datadir):
        self.run_deposition_wet(datadir, separate=False)

    #------------------------------------------------------------------

    def run_deposition_tot(self, datadir, *, separate):
        """Read total deposition field."""
        self.run(
            separate=separate,
            datafile=self.datafile(datadir),
            cls_fld_specs=FlexFieldSpecsDeposition,
            dims_mult=self.dims_shared,
            var_names_ref=[
                f'WD_spec{self.species_id:03d}',
                f'DD_spec{self.species_id:03d}',
            ],
            var_specs_mult_unshared={'deposition': 'tot'},
        )

    def test_deposition_tot_separate(self, datadir):
        self.run_deposition_tot(datadir, separate=True)

    def test_deposition_tot_together(self, datadir):
        self.run_deposition_tot(datadir, separate=False)
