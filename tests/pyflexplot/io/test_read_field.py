#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import numpy as np
import pytest

from pyflexplot.io import FileReader
from pyflexplot.field_specs import FieldSpecs
from pyflexplot.var_specs import MultiVarSpecs

from utils import datadir
from io_utils import read_nc_var


def get_var_name_ref(var_specs, var_names_ref):
    if var_specs.issubcls("concentration"):
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif var_specs.issubcls("deposition"):
        for var_name in var_names_ref:
            if (var_specs.deposition, var_name[:2]) in [("dry", "DD"), ("wet", "WD")]:
                return var_name
    raise NotImplementedError(f"{var_specs}")


class Test_ReadField_Single:
    """Read one 2D field from a FLEXPART NetCDF file."""

    # Dimensions shared by all tests
    dims_shared = {
        "nageclass": 0,
        "numpoint": 0,
        "time": 3,
    }

    # Variable specifications shared by all tests
    var_specs_mult_shared = {
        "integrate": False,
        "species_id": 2,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared["species_id"]

    def datafile(self, datadir):
        return f"{datadir}/flexpart_cosmo-1_case2.nc"

    def run(
        self,
        *,
        datadir,
        name,
        dims,
        var_names_ref,
        var_specs_mult_unshared,
        scale_fld_ref=1.0,
    ):
        """Run an individual test."""

        # Initialize variable specifications
        var_specs_dct = {
            **dims,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        multi_var_specs_lst = MultiVarSpecs.create(
            name, var_specs_dct, lang=None, words=None,
        )
        assert len(multi_var_specs_lst) == 1
        multi_var_specs = next(iter(multi_var_specs_lst))

        # Initialize field specifications
        fld_specs = FieldSpecs(name, multi_var_specs)

        # Read input field
        flex_field = FileReader(self.datafile(datadir)).run(fld_specs)
        fld = flex_field.fld

        # Read reference field
        fld_ref = (
            np.nansum(
                [
                    read_nc_var(
                        self.datafile(datadir),
                        get_var_name_ref(var_specs, var_names_ref),
                        var_specs,
                    )
                    for var_specs in multi_var_specs
                ],
                axis=0,
            )
            * scale_fld_ref
        )

        # Check array
        assert fld.shape == fld_ref.shape
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)

    def test_concentration(self, datadir):
        """Read concentration field."""
        self.run(
            datadir=datadir,
            name="concentration",
            dims={**self.dims_shared, "level": 1},
            var_names_ref=[f"spec{self.species_id:03d}"],
            var_specs_mult_unshared={},
        )

    def test_deposition_dry(self, datadir):
        """Read dry deposition field."""
        self.run(
            datadir=datadir,
            name="deposition",
            dims=self.dims_shared,
            var_names_ref=[f"DD_spec{self.species_id:03d}"],
            var_specs_mult_unshared={"deposition": "dry"},
            scale_fld_ref=1 / 3,
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datadir=datadir,
            name="deposition",
            dims=self.dims_shared,
            var_names_ref=[f"WD_spec{self.species_id:03d}"],
            var_specs_mult_unshared={"deposition": "wet"},
            scale_fld_ref=1 / 3,
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        self.run(
            datadir=datadir,
            name="deposition",
            dims=self.dims_shared,
            var_names_ref=[
                f"WD_spec{self.species_id:03d}",
                f"DD_spec{self.species_id:03d}",
            ],
            var_specs_mult_unshared={"deposition": ("wet", "dry")},
            scale_fld_ref=1 / 3,
        )


class Test_ReadField_Multiple:
    """Read multiple 2D fields from a FLEXPART NetCDF file."""

    # Dimensions arguments shared by all tests
    dims_shared = {
        "nageclass": 0,
        "numpoint": 0,
        "time": [0, 3, 9],
    }

    # Variables specification arguments shared by all tests
    var_specs_mult_shared = {
        "integrate": True,
        "species_id": 1,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared["species_id"]

    def datafile(self, datadir):
        return f"{datadir}/flexpart_cosmo-1_case2.nc"

    def run(
        self,
        *,
        datafile,
        name,
        dims_mult,
        var_names_ref,
        var_specs_mult_unshared,
        scale_fld_ref=1.0,
    ):
        """Run an individual test, reading one field after another."""

        # Create field specifications list
        var_specs_dct = {
            **dims_mult,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        multi_var_specs_lst = MultiVarSpecs.create(
            name, var_specs_dct, lang=None, words=None,
        )
        fld_specs_lst = [
            FieldSpecs(name, multi_var_specs) for multi_var_specs in multi_var_specs_lst
        ]

        dim_names = list(dims_mult.keys())

        # Process field specifications one after another
        for fld_specs in fld_specs_lst:
            self._run_core(datafile, dim_names, var_names_ref, fld_specs, scale_fld_ref)

    def _run_core(self, datafile, dim_names, var_names_ref, fld_specs, scale_fld_ref):

        # Read input fields
        flex_field_lst = FileReader(datafile).run([fld_specs])
        fld = np.array([flex_field.fld for flex_field in flex_field_lst])
        assert fld.shape[0] == 1
        fld = fld[0]

        # Read reference fields
        fld_ref = None
        for var_specs in fld_specs.multi_var_specs:
            flds_ref_i = [
                read_nc_var(
                    datafile, get_var_name_ref(var_specs, var_names_ref), var_specs,
                )
            ]
            fld_ref_i = np.nansum(flds_ref_i, axis=0)
            if fld_ref is None:
                fld_ref = fld_ref_i
            else:
                fld_ref += fld_ref_i
        fld_ref *= scale_fld_ref

        assert fld.shape == fld_ref.shape
        assert np.isclose(np.nanmean(fld), np.nanmean(fld_ref), rtol=1e-6)
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)

    def test_concentration(self, datadir):
        """Read multiple concentration fields."""
        self.run(
            datafile=self.datafile(datadir),
            name="concentration",
            dims_mult={**self.dims_shared, "level": [0, 2]},
            var_names_ref=[f"spec{self.species_id:03d}"],
            var_specs_mult_unshared={},
            scale_fld_ref=3.0,
        )

    def test_deposition_dry(self, datadir):
        """Read dry deposition fields."""
        self.run(
            datafile=self.datafile(datadir),
            name="deposition",
            dims_mult=self.dims_shared,
            var_names_ref=[f"DD_spec{self.species_id:03d}"],
            var_specs_mult_unshared={"deposition": "dry"},
        )

    def test_deposition_wet(self, datadir):
        """Read wet deposition field."""
        self.run(
            datafile=self.datafile(datadir),
            name="deposition",
            dims_mult=self.dims_shared,
            var_names_ref=[f"WD_spec{self.species_id:03d}"],
            var_specs_mult_unshared={"deposition": "wet"},
        )

    def test_deposition_tot(self, datadir):
        """Read total deposition field."""
        self.run(
            datafile=self.datafile(datadir),
            name="deposition",
            dims_mult=self.dims_shared,
            var_names_ref=[
                f"WD_spec{self.species_id:03d}",
                f"DD_spec{self.species_id:03d}",
            ],
            var_specs_mult_unshared={"deposition": ("wet", "dry")},
        )
