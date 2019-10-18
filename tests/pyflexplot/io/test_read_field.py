#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import numpy as np
import pytest

from utils import datadir

from pyflexplot.io import FieldSpecs
from pyflexplot.io import FileReader

from io_utils import read_nc_var


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
        flex_field = FileReader(self.datafile(datadir)).run(fld_specs)
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
            FieldSpecs.subclass('concentration'),
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
            FieldSpecs.subclass('deposition'),
            dims=self.dims_shared,
            var_names_ref=[f'DD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'dry'},
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datadir,
            FieldSpecs.subclass('deposition'),
            dims=self.dims_shared,
            var_names_ref=[f'WD_spec{self.species_id:03d}'],
            var_specs_mult_unshared={'deposition': 'wet'},
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        self.run(
            datadir,
            FieldSpecs.subclass('deposition'),
            dims=self.dims_shared,
            var_names_ref=[
                f'WD_spec{self.species_id:03d}',
                f'DD_spec{self.species_id:03d}',
            ],
            var_specs_mult_unshared={'deposition': 'tot'},
        )


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
        flex_field_lst = FileReader(datafile).run(fld_specs_lst)
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
            cls_fld_specs=FieldSpecs.subclass('concentration'),
            dims_mult={
                **self.dims_shared, 'level_lst': [0, 2]
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
            cls_fld_specs=FieldSpecs.subclass('deposition'),
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
            cls_fld_specs=FieldSpecs.subclass('deposition'),
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
            cls_fld_specs=FieldSpecs.subclass('deposition'),
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
