# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.io``.
"""
# Standard library
import functools

# Third-party
import numpy as np

# First-party
from pyflexplot.data import ensemble_probability
from pyflexplot.io import read_fields
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection
from srutils.dict import decompress_multival_dict

from io_utils import read_nc_var  # isort:skip
from utils import datadir  # noqa:F401 isort:skip


def get_var_name_ref(setup, var_names_ref):
    if setup.input_variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.input_variable == "deposition":
        for var_name in var_names_ref:
            if (setup.deposition_type, var_name[:2]) in [("dry", "DD"), ("wet", "WD")]:
                return var_name
    raise NotImplementedError(f"{setup}")


class TestReadFieldEnsemble_Single:
    """Read one ensemble of 2D fields from FLEXPART NetCDF files."""

    # InputSetup parameters shared by all tests
    setup_params_shared = {
        "infile": "dummy.nc",
        "integrate": False,
        "outfile": "dummy.png",
        "ens_variable": "mean",
        "species_id": 2,
        "time": 10,
        "input_variable": "concentration",
    }

    @property
    def species_id(self):
        return self.setup_params_shared["species_id"]

    # Ensemble member ids
    ens_member_ids = [0, 1, 5, 10, 15, 20]

    def datafile_fmt(self, datadir):  # noqa:F811
        return f"{datadir}/flexpart_cosmo-2e_2019072712_{{ens_member:03d}}.nc"

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
        setup_dct = {
            **self.setup_params_shared,
            **setup_params,
            "ens_member_id": self.ens_member_ids,
        }
        # SR_TMP <
        if ens_var in ["probability", "minimum", "maximum", "mean", "median"]:
            setup_dct["ens_variable"] = ens_var
        else:
            setup_dct["plot_type"] = f"ensemble_{ens_var}"
        # SR_TMP >
        setup = InputSetup.create(setup_dct)
        var_setups_lst = InputSetupCollection([setup]).decompress_grouped_by_time()
        assert len(var_setups_lst) == 1

        # Read input fields
        fields, mdata_lst = read_fields(datafile_fmt, var_setups_lst)
        assert len(fields) == 1
        assert len(mdata_lst) == 1
        fld = fields[0].fld

        # SR_TMP <
        var_setups = next(iter(var_setups_lst))
        setups = var_setups.compress().decompress_partially(
            None, skip=["ens_member_id"]
        )
        # SR_TMP >
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
                            model="cosmo2",  # SR_TMP
                        )
                        for ens_member_id in self.ens_member_ids
                    ]
                    for var_name in var_names_ref
                ],
                axis=0,
            ),
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
            fct_reduce_mem=lambda arr: np.nanmean(arr, axis=0),
        )


class TestReadFieldEnsemble_Multiple:
    """Read multiple 2D field ensembles from FLEXPART NetCDF files."""

    # InputSetup parameters arguments shared by all tests
    shared_setup_params_compressed = {
        "infile": "dummy.nc",
        "integrate": True,
        "outfile": "dummy.png",
        "species_id": 1,
        "time": [0, 3, 9],
    }

    # Species ID
    species_id = shared_setup_params_compressed["species_id"]

    # Ensemble member ids
    ens_member_ids = [0, 1, 5, 10, 15, 20]

    # Thresholds for ensemble probability
    ens_prob_thr_concentration = 1e-7  # SR_TMP
    ens_prob_thr_tot_deposition = None  # SR_TMP

    def datafile_fmt(self, datadir):  # noqa:F811
        return f"{datadir}/flexpart_cosmo-2e_2019072712_{{ens_member:03d}}.nc"

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
            }
            # SR_TMP <
            if ens_var in ["probability", "minimum", "maximum", "mean", "median"]:
                setup_params_i["ens_variable"] = ens_var
            else:
                setup_params_i["plot_type"] = f"ensemble_{ens_var}"
            # SR_TMP >
            setups.append(InputSetup.create(setup_params_i))
        var_setups_lst = InputSetupCollection(setups).decompress_grouped_by_time()

        run_core = functools.partial(
            self._run_core, datafile_fmt, var_names_ref, fct_reduce_mem, scale_fld_ref,
        )
        if separate:
            # Process field specifications one after another
            for var_setups in var_setups_lst:
                run_core([var_setups])
        else:
            run_core(var_setups_lst)

    def _run_core(
        self,
        datafile_fmt,
        var_names_ref,
        fct_reduce_mem,
        scale_fld_ref,
        var_setups_lst,
    ):
        # Collect merged variables specifications
        # SR_TMP < TODO cleaner solution
        compressed_setups = InputSetupCollection(
            [var_setups.compress() for var_setups in var_setups_lst],
        )
        # SR_TMP >

        # Read input fields
        fields, mdata_lst = read_fields(datafile_fmt, var_setups_lst)
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
                        model="cosmo2",  # SR_TMP
                    )
                    * scale_fld_ref
                    for ens_member_id in self.ens_member_ids
                ]
                # SR_TMP <
                for setup in compressed_setup.decompress_partially(
                    None, skip=["ens_member_id"],
                )
                # SR_TMP >
            ]
            fld_ref_lst.append(fct_reduce_mem(np.nansum(fld_ref_mem_time, axis=0)))
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
            "mean": lambda arr: np.nanmean(arr, axis=0),
            "maximum": lambda arr: np.nanmax(arr, axis=0),
            "probability": (
                lambda arr: ensemble_probability(
                    arr, self.ens_prob_thr_concentration, len(self.ens_member_ids)
                )
            ),
        }[ens_var]

        setup_params = {
            "level": 1,
            "input_variable": "concentration",
        }
        if ens_var == "probability":
            setup_params["ens_param_thr"] = self.ens_prob_thr_concentration

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

    def test_ens_probability_concentration(self, datadir):  # noqa:F811
        self.run_concentration(
            datadir, "probability", separate=False, scale_fld_ref=3.0,
        )

    # Deposition

    def run_deposition_tot(self, datadir, ens_var, *, separate=False):  # noqa:F811
        """Read ensemble total deposition field."""
        fct_reduce_mem = {
            "mean": lambda arr: np.nanmean(arr, axis=0),
            "maximum": lambda arr: np.nanmax(arr, axis=0),
        }[ens_var]
        self.run(
            separate=separate,
            datafile_fmt=self.datafile_fmt(datadir),
            var_names_ref=[
                f"WD_spec{self.species_id:03d}",
                f"DD_spec{self.species_id:03d}",
            ],
            setup_params={"input_variable": "deposition", "deposition_type": "tot"},
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_deposition_tot_separate(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean", separate=True)

    def test_ens_mean_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean", separate=False)

    def test_ens_max_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "maximum")
