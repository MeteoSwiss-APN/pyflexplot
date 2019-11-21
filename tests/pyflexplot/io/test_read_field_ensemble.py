#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import numpy as np
import pytest

from pyflexplot.io import FieldSpecs
from pyflexplot.io import FileReader

from pyflexplot.data import threshold_agreement

from utils import datadir
from io_utils import read_nc_var


class TestReadFieldEnsemble_Single:
    """Read one ensemble of 2D fields from FLEXPART NetCDF files."""

    # Dimensions shared by all tests
    dims_shared = {
        "nageclass": 0,
        "numpoint": 0,
        "time": 10,
    }

    # Variable specifications shared by all tests
    var_specs_mult_shared = {
        "integrate": False,
        "species_id": 2,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared["species_id"]

    # Ensemble member ids
    member_ids = [0, 1, 5, 10, 15, 20]

    def datafile_fmt(self, datadir):
        return f"{datadir}/grid_conc_20190727120000_{{member_id:03d}}.nc"

    def datafile(self, member_id, *, datadir=None, datafile_fmt=None):
        if datafile_fmt is None:
            datafile_fmt = self.datafile_fmt(datadir)
        return datafile_fmt.format(member_id=member_id)

    def run(
        self,
        datadir,
        *,
        cls_fld_specs,
        dims,
        var_names_ref,
        var_specs_mult_unshared,
        ens_var,
        fct_reduce_mem,
    ):
        """Run an individual test."""

        datafile_fmt = self.datafile_fmt(datadir)

        # Initialize specifications
        var_specs_raw = {
            **dims,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs = cls_fld_specs(
            var_specs_raw, member_ids=self.member_ids, ens_var=ens_var
        )
        var_specs = cls_fld_specs.cls_var_specs(**var_specs_raw)

        # Read input fields
        flex_field = FileReader(datafile_fmt).run(fld_specs)
        fld = flex_field.fld

        # Read reference fields
        fld_ref = fct_reduce_mem(
            np.nansum(
                [
                    [
                        read_nc_var(
                            self.datafile(member_id, datafile_fmt=datafile_fmt),
                            var_name,
                            var_specs,
                        )
                        for member_id in self.member_ids
                    ]
                    for var_name in var_names_ref
                ],
                axis=0,
            ),
            axis=0,
        )

        # Check array
        assert fld.shape == fld_ref.shape
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)

    def test_ens_mean_concentration(self, datadir):
        """Read concentration field."""
        self.run(
            datadir,
            cls_fld_specs=FieldSpecs.subclass("ens_mean_concentration"),
            dims={**self.dims_shared, "level": 1},
            var_names_ref=[f"spec{self.species_id:03d}"],
            var_specs_mult_unshared={},
            ens_var="mean",
            fct_reduce_mem=np.nanmean,
        )


class TestReadFieldEnsemble_Multiple:
    """Read multiple 2D field ensembles from FLEXPART NetCDF files."""

    # Dimensions arguments shared by all tests
    dims_shared = {
        "nageclass": 0,
        "numpoint": 0,
        "time_lst": [0, 3, 9],
    }

    # Variables specification arguments shared by all tests
    var_specs_mult_shared = {
        "integrate": True,
        "species_id": 1,
    }

    @property
    def species_id(self):
        return self.var_specs_mult_shared["species_id"]

    # Ensemble member ids
    member_ids = [0, 1, 5, 10, 15, 20]

    # Thresholds for ensemble threshold agreement
    agreement_threshold_concentration = 1e-7  # SR_TMP
    agreement_threshold_deposition_tot = None  # SR_TMP

    def datafile_fmt(self, datadir):
        return f"{datadir}/grid_conc_20190727120000_{{member_id:03d}}.nc"

    def datafile(self, member_id, *, datafile_fmt=None, datadir=None):
        if datafile_fmt is None:
            datafile_fmt = self.datafile_fmt(datadir)
        return datafile_fmt.format(member_id=member_id)

    def run(
        self,
        *,
        separate,
        datafile_fmt,
        cls_fld_specs,
        dims_mult,
        var_names_ref,
        var_specs_mult_unshared,
        ens_var,
        ens_var_setup,
        fct_reduce_mem,
        scale_fld_ref=1.0,
    ):
        """Run an individual test, reading one field after another."""

        # Create field specifications list
        var_specs_mult = {
            **dims_mult,
            **self.var_specs_mult_shared,
            **var_specs_mult_unshared,
        }
        fld_specs_lst = cls_fld_specs.multiple(
            var_specs_mult,
            member_ids=self.member_ids,
            ens_var=ens_var,
            ens_var_setup=ens_var_setup,
        )

        dim_names = sorted([d.replace("_lst", "") for d in dims_mult.keys()])

        if separate:
            # Process field specifications one after another
            for fld_specs in fld_specs_lst:
                self._run_core(
                    datafile_fmt,
                    dim_names,
                    var_names_ref,
                    [fld_specs],
                    fct_reduce_mem,
                    scale_fld_ref,
                )
        else:
            self._run_core(
                datafile_fmt,
                dim_names,
                var_names_ref,
                fld_specs_lst,
                fct_reduce_mem,
                scale_fld_ref,
            )

    def _run_core(
        self,
        datafile_fmt,
        dim_names,
        var_names_ref,
        fld_specs_lst,
        fct_reduce_mem,
        scale_fld_ref,
    ):

        # Read input fields
        flex_field_lst = FileReader(datafile_fmt).run(fld_specs_lst)
        fld_arr = np.array([flex_field.fld for flex_field in flex_field_lst])

        # Collect merged variables specifications
        var_specs_lst = [fs.var_specs_merged() for fs in fld_specs_lst]

        # Read reference fields
        fld_ref_lst = []
        for var_specs in var_specs_lst:
            fld_ref_mem_time = [
                [
                    read_nc_var(
                        self.datafile(member_id, datafile_fmt=datafile_fmt),
                        var_name,
                        var_specs,
                    )
                    * scale_fld_ref
                    for member_id in self.member_ids
                ]
                for var_name in var_names_ref
            ]
            fld_ref_lst.append(
                fct_reduce_mem(np.nansum(fld_ref_mem_time, axis=0), axis=0,)
            )
        fld_arr_ref = np.array(fld_ref_lst)

        assert fld_arr.shape == fld_arr_ref.shape
        assert np.isclose(np.nanmean(fld_arr), np.nanmean(fld_arr_ref))
        np.testing.assert_allclose(fld_arr, fld_arr_ref, equal_nan=True, rtol=1e-6)

    # Concentration

    def run_concentration(
        self,
        datadir,
        ens_var,
        *,
        separate=False,
        cls_fld_specs=FieldSpecs.subclass("concentration"),
        scale_fld_ref=1.0,
    ):
        """Read ensemble concentration field."""

        fct_reduce_mem = {
            "mean": np.nanmean,
            "max": np.nanmax,
            "thr_agrmt": (
                lambda arr, axis: threshold_agreement(
                    arr,
                    self.agreement_threshold_concentration,
                    axis=axis,
                    dtype=arr.dtype,
                )
            ),
        }[ens_var]
        ens_var_setup = {
            "thr_agrmt": {"thr": self.agreement_threshold_concentration},
        }.get(ens_var)

        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            cls_fld_specs=cls_fld_specs,
            dims_mult={**self.dims_shared, "level": 1},
            var_names_ref=[f"spec{self.species_id:03d}"],
            var_specs_mult_unshared={},
            ens_var=ens_var,
            ens_var_setup=ens_var_setup,
            fct_reduce_mem=fct_reduce_mem,
            scale_fld_ref=scale_fld_ref,
        )

    def test_ens_mean_concentration(self, datadir):
        self.run_concentration(datadir, "mean", separate=False, scale_fld_ref=3)

    def test_ens_threshold_agreement_concentration(self, datadir):
        self.run_concentration(
            datadir,
            "thr_agrmt",
            separate=False,
            cls_fld_specs=FieldSpecs.subclass("ens_thr_agrmt_concentration"),
            scale_fld_ref=3.0,
        )

    # Deposition

    def run_deposition_tot(self, datadir, ens_var, *, separate=False):
        """Read ensemble total deposition field."""
        fct_reduce_mem = {"mean": np.nanmean, "max": np.nanmax,}[ens_var]
        ens_var_setup = {
            # ...
        }.get(ens_var)
        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            cls_fld_specs=FieldSpecs.subclass("deposition"),
            dims_mult=self.dims_shared,
            var_names_ref=[
                f"WD_spec{self.species_id:03d}",
                f"DD_spec{self.species_id:03d}",
            ],
            var_specs_mult_unshared={"deposition": "tot"},
            ens_var=ens_var,
            ens_var_setup=ens_var_setup,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_deposition_tot_separate(self, datadir):
        self.run_deposition_tot(datadir, "mean", separate=True)

    def test_ens_mean_deposition_tot(self, datadir):
        self.run_deposition_tot(datadir, "mean", separate=False)

    def test_ens_max_deposition_tot(self, datadir):
        self.run_deposition_tot(datadir, "max")
