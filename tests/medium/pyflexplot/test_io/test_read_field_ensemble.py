#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.io``.
"""
# Standard library
import functools

# Third-party
import numpy as np

# First-party
from pyflexplot.data import threshold_agreement
from pyflexplot.io import read_files
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupCollection
from pyflexplot.specs import FldSpecs
from pyflexplot.words import WORDS
from srutils.dict import decompress_multival_dict

from io_utils import read_nc_var  # isort:skip
from utils import datadir  # noqa:F401 isort:skip


def get_var_name_ref(setup, var_names_ref):
    if setup.variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.variable == "deposition":
        for var_name in var_names_ref:
            if (setup.deposition_type, var_name[:2]) in [("dry", "DD"), ("wet", "WD")]:
                return var_name
    raise NotImplementedError(f"{setup}")


class TestReadFieldEnsemble_Single:
    """Read one ensemble of 2D fields from FLEXPART NetCDF files."""

    # Setup parameters shared by all tests
    setup_params_shared = {
        "infile": "dummy.nc",
        "integrate": False,
        "outfile": "dummy.png",
        "plot_type": "ens_mean",
        "simulation_type": "ensemble",
        "species_id": 2,
        "time": 10,
        "variable": "concentration",
    }

    @property
    def species_id(self):
        return self.setup_params_shared["species_id"]

    # Ensemble member ids
    ens_member_ids = [0, 1, 5, 10, 15, 20]

    def datafile_fmt(self, datadir):  # noqa:F811
        return f"{datadir}/flexpart_cosmo-2e_20190727120_{{ens_member:03d}}.nc"

    def datafile(self, ens_member_id, *, datadir=None, datafile_fmt=None):  # noqa:F811
        if datafile_fmt is None:
            datafile_fmt = self.datafile_fmt(datadir)
        return datafile_fmt.format(ens_member=ens_member_id)

    def run(
        self,
        datadir,  # noqa:F811
        *,
        var_names_ref,
        setup_params,
        ens_var,
        fct_reduce_mem,
    ):
        """Run an individual test."""

        datafile_fmt = self.datafile_fmt(datadir)

        # Initialize specifications
        setup = Setup(
            **{
                **self.setup_params_shared,
                **setup_params,
                "ens_member_id": self.ens_member_ids,
                "plot_type": f"ens_{ens_var}",
            }
        )
        fld_specs_lst = FldSpecs.create(setup)
        assert len(fld_specs_lst) == 1

        # Read input fields
        fields, attrs_lst = read_files(datafile_fmt, setup, WORDS, fld_specs_lst)
        assert len(fields) == 1
        assert len(attrs_lst) == 1
        fld = fields[0].fld

        setups = fld_specs_lst[0].setup.decompress()
        assert len(setups) == 1
        setup = next(iter(setups))

        # Read reference fields
        fld_ref = fct_reduce_mem(
            np.nansum(
                [
                    [
                        read_nc_var(
                            self.datafile(ens_member_id, datafile_fmt=datafile_fmt),
                            var_name,
                            setup,
                        )
                        for ens_member_id in self.ens_member_ids
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

    def test_ens_mean_concentration(self, datadir):  # noqa:F811
        """Read concentration field."""
        self.run(
            datadir,
            var_names_ref=[f"spec{self.species_id:03d}"],
            setup_params={"level": 1},
            ens_var="mean",
            fct_reduce_mem=np.nanmean,
        )


class TestReadFieldEnsemble_Multiple:
    """Read multiple 2D field ensembles from FLEXPART NetCDF files."""

    # Setup parameters arguments shared by all tests
    shared_setup_params_compressed = {
        "infile": "dummy.nc",
        "integrate": True,
        "outfile": "dummy.png",
        "simulation_type": "ensemble",
        "species_id": 1,
        "time": [0, 3, 9],
    }

    # Species ID
    species_id = shared_setup_params_compressed["species_id"]

    # Ensemble member ids
    ens_member_ids = [0, 1, 5, 10, 15, 20]

    # Thresholds for ensemble threshold agreement
    ens_thr_agrmt_thr_concentration = 1e-7  # SR_TMP
    ens_thr_agrmt_thr_deposition_tot = None  # SR_TMP

    def datafile_fmt(self, datadir):  # noqa:F811
        return f"{datadir}/flexpart_cosmo-2e_20190727120_{{ens_member:03d}}.nc"

    def datafile(self, ens_member_id, *, datafile_fmt=None, datadir=None):  # noqa:F811
        if datafile_fmt is None:
            datafile_fmt = self.datafile_fmt(datadir)
        return datafile_fmt.format(ens_member=ens_member_id)

    def run(
        self,
        *,
        separate,
        datafile_fmt,
        var_names_ref,
        setup_params,
        ens_var,
        fct_reduce_mem,
        scale_fld_ref=1.0,
    ):
        """Run an individual test, reading one field after another."""

        # Create field specifications list
        setups = []
        for shared_setup_params in decompress_multival_dict(
            self.shared_setup_params_compressed, skip=["infile"],
        ):
            shared_setup_params["time"] = [shared_setup_params["time"]]
            setup_params_i = {
                **shared_setup_params,
                **setup_params,
                "ens_member_id": self.ens_member_ids,
                "plot_type": f"ens_{ens_var}",
            }
            setups.append(Setup(**setup_params_i))
        fld_specs_lst = FldSpecs.create(SetupCollection(setups))

        run_core = functools.partial(
            self._run_core, datafile_fmt, var_names_ref, fct_reduce_mem, scale_fld_ref,
        )
        if separate:
            # Process field specifications one after another
            for fld_specs in fld_specs_lst:
                run_core([fld_specs])
        else:
            run_core(fld_specs_lst)

    def _run_core(
        self, datafile_fmt, var_names_ref, fct_reduce_mem, scale_fld_ref, fld_specs_lst,
    ):

        # Collect merged variables specifications
        compressed_setups = SetupCollection(
            [fld_specs.setup for fld_specs in fld_specs_lst],
        )
        global_setup = Setup.compress(compressed_setups)

        # Read input fields
        fields, attrs_lst = read_files(datafile_fmt, global_setup, WORDS, fld_specs_lst)
        fld_arr = np.array([field.fld for field in fields])

        # Read reference fields
        fld_ref_lst = []
        for compressed_setup in compressed_setups:
            fld_ref_mem_time = [
                [
                    read_nc_var(
                        self.datafile(ens_member_id, datafile_fmt=datafile_fmt),
                        get_var_name_ref(setup, var_names_ref),
                        setup,
                    )
                    * scale_fld_ref
                    for ens_member_id in self.ens_member_ids
                ]
                for setup in compressed_setup.decompress()
            ]
            fld_ref_lst.append(
                fct_reduce_mem(np.nansum(fld_ref_mem_time, axis=0), axis=0)
            )
        fld_arr_ref = np.array(fld_ref_lst)

        assert fld_arr.shape == fld_arr_ref.shape
        assert np.isclose(np.nanmean(fld_arr), np.nanmean(fld_arr_ref))
        np.testing.assert_allclose(fld_arr, fld_arr_ref, equal_nan=True, rtol=1e-6)

    # Concentration

    def run_concentration(
        self, datadir, ens_var, *, separate=False, scale_fld_ref=1.0,  # noqa:F811
    ):
        """Read ensemble concentration field."""

        fct_reduce_mem = {
            "mean": np.nanmean,
            "max": np.nanmax,
            "thr_agrmt": (
                lambda arr, axis: threshold_agreement(
                    arr, self.ens_thr_agrmt_thr_concentration, axis=axis,
                )
            ),
        }[ens_var]

        setup_params = {
            "level": 1,
            "variable": "concentration",
        }
        if ens_var == "thr_agrmt":
            setup_params["ens_param_thr"] = self.ens_thr_agrmt_thr_concentration

        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            var_names_ref=[f"spec{self.species_id:03d}"],
            setup_params=setup_params,
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
            scale_fld_ref=scale_fld_ref,
        )

    def test_ens_mean_concentration(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "mean", separate=False, scale_fld_ref=3)

    def test_ens_threshold_agreement_concentration(self, datadir):  # noqa:F811
        self.run_concentration(
            datadir, "thr_agrmt", separate=False, scale_fld_ref=3.0,
        )

    # Deposition

    def run_deposition_tot(self, datadir, ens_var, *, separate=False):  # noqa:F811
        """Read ensemble total deposition field."""
        fct_reduce_mem = {"mean": np.nanmean, "max": np.nanmax}[ens_var]
        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            var_names_ref=[
                f"WD_spec{self.species_id:03d}",
                f"DD_spec{self.species_id:03d}",
            ],
            setup_params={"variable": "deposition", "deposition_type": "tot"},
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_deposition_tot_separate(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean", separate=True)

    def test_ens_mean_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean", separate=False)

    def test_ens_max_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "max")
