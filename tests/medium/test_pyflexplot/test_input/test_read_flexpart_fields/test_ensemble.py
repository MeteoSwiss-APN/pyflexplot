"""Tests for module ``pyflexplot.input.read_fields``.

These tests use ensemble data.

"""
# Standard library
from typing import Any
from typing import Dict

# Third-party
import numpy as np
import pytest

# First-party
from pyflexplot.input.data import ensemble_probability
from pyflexplot.input.read_fields import read_fields
from pyflexplot.setups.setup import PlotSetup
from pyflexplot.setups.setup import PlotSetupGroup
from srutils.dict import decompress_multival_dict
from srutils.dict import merge_dicts

# Local
from .shared import datadir_reduced as datadir  # noqa:F401
from .shared import read_flexpart_field


def get_var_name_ref(setup, var_names_ref):
    if setup.panels.input_variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.panels.input_variable == "deposition":
        for var_name in var_names_ref:
            if (setup.deposition_type_str, var_name[:2]) in [
                ("dry", "DD"),
                ("wet", "WD"),
            ]:
                return var_name
    raise NotImplementedError(f"{setup}")


class TestReadFieldEnsemble_Single:
    """Read one ensemble of 2D fields from FLEXPART NetCDF files."""

    # Setup parameters shared by all tests
    setup_params_shared: Dict[str, Any] = {
        "infile": "dummy.nc",
        "outfile": "dummy.png",
        "model": {
            "name": "COSMO-2E",
        },
        "panels": {
            "integrate": False,
            "ens_variable": "mean",
            "dimensions": {"time": 10, "species_id": 2},
            "input_variable": "concentration",
        },
    }

    species_id = setup_params_shared["panels"]["dimensions"]["species_id"]

    # Ensemble member ids
    ens_member_ids = (0, 1, 5, 10, 15, 20)

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
        cache_on,
    ):
        """Run an individual test."""
        datafile_fmt = self.datafile_fmt(datadir)

        # Initialize specifications
        setup_dct = merge_dicts(
            self.setup_params_shared,
            setup_params,
            {
                "infile": datafile_fmt,
                "model": {
                    "ens_member_id": self.ens_member_ids,
                },
            },
        )
        # SR_TMP <
        if ens_var in ["probability", "minimum", "maximum", "mean", "median"]:
            setup_dct["panels"]["ens_variable"] = ens_var
        else:
            setup_dct["panels"]["plot_type"] = f"ensemble_{ens_var}"
        # SR_TMP >
        setups = PlotSetupGroup([PlotSetup.create(setup_dct)])

        # Read input fields
        field_groups = read_fields(setups, {"add_ts0": True, "cache_on": cache_on})
        assert len(field_groups) == 1
        assert len(field_groups[0]) == 1
        fld = next(iter(field_groups[0])).fld

        # SR_TMP <
        var_setups_lst = setups.decompress_twice(
            "dimensions.time", skip=["model.ens_member_id"]
        )
        assert len(var_setups_lst) == 1
        var_setups = next(iter(var_setups_lst))
        setups = var_setups.compress().decompress_partially(
            None, skip=["model.ens_member_id"]
        )
        # SR_TMP >
        assert len(setups) == 1
        setup = next(iter(setups))

        # Read reference fields
        fld_ref = fct_reduce_mem(
            np.nansum(
                [
                    [
                        read_flexpart_field(
                            self.datafile(ens_member_id, datafile_fmt=datafile_fmt),
                            var_name,
                            setup,
                            model="COSMO-2",  # SR_TMP
                            add_ts0=True,
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

    def test_ens_mean_concentration(self, datadir, cache_on=False):  # noqa:F811
        """Read concentration field."""
        self.run(
            datadir,
            var_names_ref=[f"spec{self.species_id:03d}"],
            setup_params={"panels": {"dimensions": {"level": 1}}},
            ens_var="mean",
            fct_reduce_mem=lambda arr: np.nanmean(arr, axis=0),
            cache_on=cache_on,
        )

    @pytest.mark.skip("cache is broken")
    def test_ens_mean_concentration_cached(self, datadir):  # noqa:F811
        self.test_ens_mean_concentration(datadir, cache_on=True)


class TestReadFieldEnsemble_Multiple:
    """Read multiple 2D field ensembles from FLEXPART NetCDF files."""

    # Setup parameters arguments shared by all tests
    shared_setup_params_compressed: Dict[str, Any] = {
        "infile": "dummy.nc",
        "outfile": "dummy.png",
        "model": {
            "name": "COSMO-2E",
        },
        "panels": {
            "integrate": True,
            "dimensions": {"species_id": 1, "time": [0, 3, 9]},
        },
    }

    # Species ID
    species_id = shared_setup_params_compressed["panels"]["dimensions"]["species_id"]

    # Ensemble member ids
    ens_member_ids = (0, 1, 5, 10, 15, 20)

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
        datafile_fmt,
        var_names_ref,
        setup_params,
        ens_var,
        fct_reduce_mem,
        cache_on,
        scale_fld_ref=1.0,
    ):
        """Run an individual test, reading one field after another."""
        # Create field specifications list
        setup_lst = []
        for shared_setup_params in decompress_multival_dict(
            self.shared_setup_params_compressed, skip=["infile", "panels"]
        ):
            if "panels" not in shared_setup_params:
                shared_setup_params["panels"] = {}
            for shared_core in decompress_multival_dict(shared_setup_params["panels"]):
                setup_params_i = merge_dicts(
                    shared_setup_params,
                    setup_params,
                    {
                        "model": {
                            "ens_member_id": self.ens_member_ids,
                        },
                    },
                )
                # SR_TMP <
                if ens_var in ["probability", "minimum", "maximum", "mean", "median"]:
                    setup_params_i["panels"]["ens_variable"] = ens_var
                else:
                    setup_params_i["panels"]["plot_type"] = f"ensemble_{ens_var}"
                # SR_TMP >
                setup_lst.append(PlotSetup.create(setup_params_i))
        setups = PlotSetupGroup(setup_lst)

        # Read input fields
        field_groups = read_fields(setups, {"add_ts0": False, "cache_on": cache_on})
        fld_arr = np.array(
            [field.fld for field_group in field_groups for field in field_group]
        )

        # Read reference fields
        fld_ref_lst = []
        for sub_setups_time in setups.decompress_partially(["dimensions.time"]):
            fld_ref_mem_time = []
            for sub_setups in sub_setups_time.decompress_partially(
                None, skip=["model.ens_member_id"]
            ):
                # SR_TMP <
                assert len(sub_setups) == 1
                sub_setup = next(iter(sub_setups))
                # SR_TMP >
                fld_ref_mem_time.append([])
                flds_mem = []
                for ens_member_id in self.ens_member_ids:
                    fld = (
                        read_flexpart_field(
                            self.datafile(ens_member_id, datafile_fmt=datafile_fmt),
                            get_var_name_ref(sub_setup, var_names_ref),
                            sub_setup,
                            model="COSMO-2",  # SR_TMP
                            add_ts0=False,
                        )
                        * scale_fld_ref
                    )
                    flds_mem.append(fld)
                    fld_ref_mem_time[-1].append(fld)
            fld_ref_lst.append(fct_reduce_mem(np.nansum(fld_ref_mem_time, axis=0)))
        fld_arr_ref = np.array(fld_ref_lst)
        assert fld_arr.shape == fld_arr_ref.shape
        try:
            assert np.isclose(np.nanmean(fld_arr), np.nanmean(fld_arr_ref))
        except AssertionError as error:
            fld_rel = fld_arr / fld_arr_ref
            if np.isclose(np.nanmin(fld_rel), np.nanmax(fld_rel)):
                f = np.nanmean(fld_rel)
                raise AssertionError(
                    f"fields differ by constant factor: result = "
                    f"{f:g} * reference (1 / {1.0 / f:g}))"
                ) from error
            else:
                raise error
        np.testing.assert_allclose(fld_arr, fld_arr_ref, equal_nan=True, rtol=1e-6)

    def run_concentration(
        self, datadir, ens_var, *, cache_on=False, scale_fld_ref=1.0  # noqa:F811
    ):
        """Read ensemble concentration field."""
        datafile_fmt = self.datafile_fmt(datadir)

        fct_reduce_mem = {
            "mean": lambda arr: np.nanmean(arr, axis=0),
            "maximum": lambda arr: np.nanmax(arr, axis=0),
            "probability": (
                lambda arr: ensemble_probability(arr, self.ens_prob_thr_concentration)
            ),
        }[ens_var]

        setup_params = {
            "infile": datafile_fmt,
            "panels": {
                "dimensions": {"level": 1},
                "input_variable": "concentration",
            },
        }
        if ens_var == "probability":
            setup_params["panels"]["ens_params"] = {
                "thr": self.ens_prob_thr_concentration
            }

        self.run(
            datafile_fmt=datafile_fmt,
            var_names_ref=[f"spec{self.species_id:03d}"],
            setup_params=setup_params,
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
            cache_on=cache_on,
            scale_fld_ref=scale_fld_ref,
        )

    def test_ens_mean_concentration(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "mean", scale_fld_ref=3.0)

    def test_ens_probability_concentration(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "probability", scale_fld_ref=3.0)

    @pytest.mark.skip("cache is broken")
    def test_ens_mean_concentration_cached(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "mean", cache_on=True, scale_fld_ref=3.0)

    @pytest.mark.skip("cache is broken")
    def test_ens_probability_concentration_cached(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "probability", cache_on=True, scale_fld_ref=3.0)

    def run_deposition_tot(self, datadir, ens_var, cache_on=False):  # noqa:F811
        """Read ensemble total deposition field."""
        datafile_fmt = self.datafile_fmt(datadir)
        fct_reduce_mem = {
            "mean": lambda arr: np.nanmean(arr, axis=0),
            "maximum": lambda arr: np.nanmax(arr, axis=0),
        }[ens_var]
        self.run(
            datafile_fmt=datafile_fmt,
            var_names_ref=[
                f"WD_spec{self.species_id:03d}",
                f"DD_spec{self.species_id:03d}",
            ],
            setup_params={
                "infile": datafile_fmt,
                "panels": {
                    "input_variable": "deposition",
                    "combine_deposition_types": True,
                    "dimensions": {"deposition_type": ("dry", "wet")},
                },
            },
            ens_var=ens_var,
            cache_on=cache_on,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean")

    def test_ens_max_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "maximum")

    @pytest.mark.skip("cache is broken")
    def test_ens_mean_deposition_tot_cached(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean", cache_on=True)

    @pytest.mark.skip("cache is broken")
    def test_ens_max_deposition_tot_cached(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "maximum", cache_on=True)
