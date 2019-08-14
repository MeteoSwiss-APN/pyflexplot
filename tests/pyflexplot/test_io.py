#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `pyflexplot/io.py` package."""
import logging as log
import netCDF4 as nc4
import numpy as np
import os
import pytest

from utils import datadir

from pyflexplot.io import FlexFieldSpecs
from pyflexplot.io import FlexFileReader

from pyflexplot.utils import dict_mult_vals_product

from pyflexplot.utils_dev import ipython  #SR_DEV


#SR_TMP<<<
def fix_nc_fld(fld):
    """Fix field read directly from NetCDF file."""
    fld[:] *= 1e-12


def read_nc_var(path, var_name, var_specs):
    with nc4.Dataset(path, 'r') as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        inds = []
        for name in var.dimensions:
            if name in ['rlat', 'rlon']:
                ind = slice(*getattr(var_specs, name, [None]))
            elif name == 'time':
                # Read all timesteps until the selected one
                ind = slice(getattr(var_specs, name) + 1)
            else:
                ind = getattr(var_specs, name, slice(None))
            inds.append(ind)

        # Read field
        fld = var[inds]
        assert len(fld.shape) == 3

        # Reduce time dimension
        if isinstance(var_specs, FlexFieldSpecs.Concentration.cls_var_specs):
            if var_specs.integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif isinstance(var_specs, FlexFieldSpecs.Deposition.cls_var_specs):
            if not var_specs.integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(
                f"var specs of type '{type(var_specs).__name__}'")
        fld = fld[-1]

        # Fix some issues with the input data
        fix_nc_fld(fld)  #SR_TMP

        return fld


class TestReadField_Single:
    """Read one 2D field from a FLEXPART NetCDF file."""

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

    def datafile(self, datadir):
        return f'{datadir}/flexpart_cosmo-1_case2.nc'

    #------------------------------------------------------------------

    def run(
            self, datadir, cls_fld_specs, dims, var_names_ref,
            var_specs_mult_unshared):
        """Run an individual test."""

        # Initialize specifications
        var_specs_raw = {
            **dims,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs = cls_fld_specs(var_specs_raw)
        var_specs = cls_fld_specs.cls_var_specs(**var_specs_raw)

        # Read input field
        flex_field = FlexFileReader(self.datafile(datadir)).run(fld_specs)
        fld = flex_field.fld

        # Read reference field
        fld_ref = np.nansum(
            [
                read_nc_var(
                    self.datafile(datadir),
                    var_name,
                    var_specs,
                ) for var_name in var_names_ref
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
            FlexFieldSpecs.Concentration,
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
            FlexFieldSpecs.Deposition,
            dims=self.dims_shared,
            var_names_ref=[f'DD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'dry'},
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datadir,
            FlexFieldSpecs.Deposition,
            dims=self.dims_shared,
            var_names_ref=[f'WD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'wet'},
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        self.run(
            datadir,
            FlexFieldSpecs.Deposition,
            dims=self.dims_shared,
            var_names_ref=[
                f'WD_spec{self.species_id:03d}',
                f'DD_spec{self.species_id:03d}',
            ],
            var_specs_mult_unshared={'deposition': 'tot'},
        )


class TestFieldSpecs_Multiple:
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
        fld_specs_mult_lst = FlexFieldSpecs.Concentration.multiple(
            var_specs_mult)

        # Create reference field specifications list
        fld_specs_mult_lst_ref = self.create_fld_specs_mult_lst_ref(
            FlexFieldSpecs.Concentration, var_specs_mult)

        assert sorted(fld_specs_mult_lst) == sorted(fld_specs_mult_lst_ref)

    def test_deposition(self):

        # Create field specifications list
        var_specs_mult = {
            **self.dims_shared,
            **self.var_specs_mult_shared,
            'deposition_lst': ['wet', 'dry', 'tot'],
        }
        fld_specs_mult_lst = FlexFieldSpecs.Deposition.multiple(var_specs_mult)

        # Create reference field specifications list
        fld_specs_mult_lst_ref = self.create_fld_specs_mult_lst_ref(
            FlexFieldSpecs.Deposition, var_specs_mult)

        assert sorted(fld_specs_mult_lst) == sorted(fld_specs_mult_lst_ref)


class TestReadField_Multiple:
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

    def datafile(self, datadir):
        return f'{datadir}/flexpart_cosmo-1_case2.nc'

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

    def _run_core(self, datafile, dim_names, var_names_ref, fld_specs_lst):

        # Read input fields
        flex_field_lst = FlexFileReader(datafile).run(fld_specs_lst)
        flds = np.array([flex_field.fld for flex_field in flex_field_lst])

        # Collect merged variables specifications
        var_specs_lst = [fs.var_specs_merged() for fs in fld_specs_lst]

        # Read reference fields
        flds_ref = []
        for var_specs in var_specs_lst:
            flds_ref_i = [
                read_nc_var(datafile, var_name, var_specs)
                for var_name in var_names_ref
            ]
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
            cls_fld_specs=FlexFieldSpecs.Concentration,
            dims_mult={**self.dims_shared, 'level_lst': [0, 2]},
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
            cls_fld_specs=FlexFieldSpecs.Deposition,
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
            cls_fld_specs=FlexFieldSpecs.Deposition,
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
            cls_fld_specs=FlexFieldSpecs.Deposition,
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


class TestReadFieldEnsemble_Single:
    """Read one ensemble of 2D fields from FLEXPART NetCDF files."""

    # Dimensions shared by all tests
    dims_shared = {
        'nageclass': 0,
        'numpoint': 0,
        'time': 10,
    }

    # Variable specifications shared by all tests
    var_specs_mult_shared = {
        'integrate': False,
        'species_id': 2,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared['species_id']

    # Ensemble member ids
    member_ids = [0, 1, 5, 10, 15, 20]

    def datafile_fmt(self, datadir):
        return f'{datadir}/grid_conc_20190727120000_{{member_id:03d}}.nc'

    def datafile(self, member_id, *, datadir=None, datafile_fmt=None):
        if datafile_fmt is None:
            datafile_fmt = self.datafile_fmt(datadir)
        return datafile_fmt.format(member_id=member_id)

    #------------------------------------------------------------------

    def run(
            self, datadir, *, cls_fld_specs, dims, var_names_ref,
            var_specs_mult_unshared, ens_var, fct_reduce_mem):
        """Run an individual test."""

        datafile_fmt = self.datafile_fmt(datadir)

        # Initialize specifications
        var_specs_raw = {
            **dims,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs = cls_fld_specs(var_specs_raw, self.member_ids)
        var_specs = cls_fld_specs.cls_var_specs(**var_specs_raw)

        # Read input fields
        flex_field = FlexFileReader(datafile_fmt).run(
            fld_specs, ens_var=ens_var)
        fld = flex_field.fld

        # Read reference fields
        fld_ref = fct_reduce_mem(
            np.nansum(
                [[
                    read_nc_var(
                        self.datafile(member_id, datafile_fmt=datafile_fmt),
                        var_name,
                        var_specs,
                    ) for member_id in self.member_ids
                ] for var_name in var_names_ref],
                axis=0,
            ),
            axis=0,
        )

        # Check array
        assert fld.shape == fld_ref.shape
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)

    #------------------------------------------------------------------

    def test_ens_mean_concentration(self, datadir):
        """Read concentration field."""
        self.run(
            datadir,
            cls_fld_specs=FlexFieldSpecs.Concentration,
            dims={
                **self.dims_shared, 'level': 1
            },
            var_names_ref=[f'spec{self.species_id:03d}'],
            var_specs_mult_unshared={},
            ens_var='mean',
            fct_reduce_mem=np.nanmean,
        )


class TestReadFieldEnsemble_Multiple:
    """Read multiple 2D field ensembles from FLEXPART NetCDF files."""

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

    # Ensemble member ids
    member_ids = [0, 1, 5, 10, 15, 20]

    def datafile_fmt(self, datadir):
        return f'{datadir}/grid_conc_20190727120000_{{member_id:03d}}.nc'

    def datafile(self, member_id, *, datafile_fmt=None, datadir=None):
        if datafile_fmt is None:
            datafile_fmt = self.datafile_fmt(datadir)
        return datafile_fmt.format(member_id=member_id)

    #------------------------------------------------------------------

    def run(
            self, *, separate, datafile_fmt, cls_fld_specs, dims_mult,
            var_names_ref, var_specs_mult_unshared, ens_var, fct_reduce_mem):
        """Run an individual test, reading one field after another."""

        # Create field specifications list
        var_specs_mult = {
            **dims_mult,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs_lst = cls_fld_specs.multiple(var_specs_mult, self.member_ids)

        dim_names = sorted([d.replace('_lst', '') for d in dims_mult.keys()])

        if separate:
            # Process field specifications one after another
            for fld_specs in fld_specs_lst:
                self._run_core(
                    datafile_fmt, dim_names, var_names_ref, [fld_specs],
                    ens_var, fct_reduce_mem)
        else:
            self._run_core(
                datafile_fmt, dim_names, var_names_ref, fld_specs_lst, ens_var,
                fct_reduce_mem)

    def _run_core(
            self, datafile_fmt, dim_names, var_names_ref, fld_specs_lst,
            ens_var, fct_reduce_mem):

        # Read input fields
        flex_field_lst = FlexFileReader(datafile_fmt).run(
            fld_specs_lst, ens_var=ens_var)
        flds = np.array([flex_field.fld for flex_field in flex_field_lst])

        # Collect merged variables specifications
        var_specs_lst = [fs.var_specs_merged() for fs in fld_specs_lst]

        # Read reference fields
        fld_ref_lst = []
        for var_specs in var_specs_lst:
            fld_ref_mem_time = [[
                read_nc_var(
                    self.datafile(member_id, datafile_fmt=datafile_fmt),
                    var_name,
                    var_specs,
                ) for member_id in self.member_ids
            ] for var_name in var_names_ref]
            fld_ref_lst.append(
                fct_reduce_mem(
                    np.nansum(fld_ref_mem_time, axis=0),
                    axis=0,
                ))
        fld_ref_arr = np.array(fld_ref_lst)

        assert flds.shape == fld_ref_arr.shape
        assert np.isclose(np.nanmean(flds), np.nanmean(fld_ref_arr))
        np.testing.assert_allclose(
            flds, fld_ref_arr, equal_nan=True, rtol=1e-6)

    #------------------------------------------------------------------
    # Concentration
    #------------------------------------------------------------------

    def run_concentration(self, datadir, ens_var, *, separate=False):
        """Read ensemble concentration field."""
        fct_reduce_mem = {
            'mean': np.nanmean,
            'max': np.nanmax,
        }[ens_var]
        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            cls_fld_specs=FlexFieldSpecs.Concentration,
            dims_mult={**self.dims_shared, 'level': 1},
            var_names_ref=[f'spec{self.species_id:03d}'],
            var_specs_mult_unshared={},
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_concentration(self, datadir):
        self.run_concentration(datadir, 'mean', separate=False)

    #------------------------------------------------------------------
    # Deposition
    #------------------------------------------------------------------

    def run_deposition_tot(self, datadir, ens_var, *, separate=False):
        """Read ensemble total deposition field."""
        fct_reduce_mem = {
            'mean': np.nanmean,
            'max': np.nanmax,
        }[ens_var]
        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            cls_fld_specs=FlexFieldSpecs.Deposition,
            dims_mult=self.dims_shared,
            var_names_ref=[
                f'WD_spec{self.species_id:03d}',
                f'DD_spec{self.species_id:03d}',
            ],
            var_specs_mult_unshared={'deposition': 'tot'},
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_deposition_tot_separate(self, datadir):
        self.run_deposition_tot(datadir, 'mean', separate=True)

    def test_ens_mean_deposition_tot(self, datadir):
        self.run_deposition_tot(datadir, 'mean', separate=False)

    def test_ens_max_deposition_tot(self, datadir):
        self.run_deposition_tot(datadir, 'max')
