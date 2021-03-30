"""Tests for module ``pyflexplot.input.read_fields``.

These tests use ensemble data.

"""
# Standard library
from typing import Any
from typing import Dict
from typing import List

# Third-party
import numpy as np

# First-party
from pyflexplot.input.data import ensemble_probability
from pyflexplot.input.read_fields import read_fields
from pyflexplot.setups.dimensions import Dimensions
from pyflexplot.setups.plot_setup import PlotSetup
from pyflexplot.setups.plot_setup import PlotSetupGroup
from srutils.dict import decompress_multival_dict
from srutils.dict import merge_dicts

# Local
from .shared import datadir_reduced as datadir  # noqa:F401
from .shared import decompress_twice
from .shared import read_flexpart_field


def get_var_name_ref(dimensions: Dimensions, var_names_ref: str) -> str:
    variable = dimensions.variable
    assert isinstance(variable, str)
    if variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif variable.endswith("_deposition"):
        for var_name in var_names_ref:
            if (variable[:3], var_name[:2]) in [
                ("dry", "DD"),
                ("wet", "WD"),
            ]:
                return var_name
    raise NotImplementedError(f"dimensions={dimensions}\nvar_names_ref={var_names_ref}")


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
            "plot_variable": "concentration",
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
            raise NotImplementedError()  # SR_DBG
            setup_dct["panels"]["plot_type"] = f"ensemble_{ens_var}"
        # SR_TMP >
        plot_setups = PlotSetupGroup([PlotSetup.create(setup_dct)])

        # Read input fields
        field_groups = read_fields(plot_setups, {"add_ts0": True})
        assert len(field_groups) == 1
        assert len(field_groups[0]) == 1
        fld = next(iter(field_groups[0])).fld

        # SR_TMP <
        plot_setups_lst = decompress_twice(
            plot_setups, "dimensions.time", skip=["model.ens_member_id"]
        )
        assert len(plot_setups_lst) == 1
        var_setups = next(iter(plot_setups_lst))
        plot_setups = var_setups.compress().decompress(skip=["model.ens_member_id"])
        # SR_TMP >
        assert len(plot_setups) == 1
        setup = next(iter(plot_setups))
        assert len(setup.panels) == 1
        panel_setup = next(iter(setup.panels))

        # Read reference fields
        fld_ref = fct_reduce_mem(
            np.nansum(
                [
                    [
                        read_flexpart_field(
                            self.datafile(ens_member_id, datafile_fmt=datafile_fmt),
                            var_name,
                            panel_setup.dimensions,
                            integrate=panel_setup.integrate,
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

    def test_ens_mean_concentration(self, datadir):  # noqa:F811
        """Read concentration field."""
        self.run(
            datadir,
            var_names_ref=[f"spec{self.species_id:03d}"],
            setup_params={"panels": {"dimensions": {"level": 1}}},
            ens_var="mean",
            fct_reduce_mem=lambda arr: np.nanmean(arr, axis=0),
        )


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
        scale_fld_ref=1.0,
    ):
        """Run an individual test, reading one field after another."""
        # Create field specifications list
        setup_dcts: List[Dict[str, Any]] = []
        for shared_setup_params in decompress_multival_dict(
            self.shared_setup_params_compressed, skip=["infile", "panels"]
        ):
            if "panels" not in shared_setup_params:
                shared_setup_params["panels"] = {}
            setup_params_i = merge_dicts(
                shared_setup_params,
                setup_params,
                {"model": {"ens_member_id": self.ens_member_ids}},
            )
            # SR_TMP <
            if ens_var in ["probability", "minimum", "maximum", "mean", "median"]:
                setup_params_i["panels"]["ens_variable"] = ens_var
            else:
                raise NotImplementedError()  # SR_DBG
                setup_params_i["panels"]["plot_type"] = f"ensemble_{ens_var}"
            # SR_TMP >
            setup_dcts.append(setup_params_i)
        plot_setup_group = PlotSetupGroup.create(setup_dcts)
        time = plot_setup_group.collect_equal("dimensions.time")

        # Read input fields
        field_groups = read_fields(plot_setup_group, {"add_ts0": False})
        fld_arr = np.array(
            [field.fld for field_group in field_groups for field in field_group]
        )
        assert fld_arr.shape[0] == len(time)

        def read_fld_ref(plot_setup_group: PlotSetupGroup) -> np.ndarray:
            assert len(plot_setup_group) == 1
            plot_setup = next(iter(plot_setup_group))
            fld_lst: List[np.ndarray] = []
            for plot_setup_i in plot_setup.decompress(["dimensions.time"]):
                fld_mem_time: List[List[np.ndarray]] = []
                for plot_setup_ij in plot_setup_i.decompress(
                    skip=["model.ens_member_id"]
                ):
                    fld_mem_time.append([])
                    for plot_setup_ijk in plot_setup_ij.decompress(
                        ["model.ens_member_id"], internal=False
                    ):
                        assert plot_setup_ijk.model.ens_member_id is not None  # mypy
                        ens_member_id = next(iter(plot_setup_ijk.model.ens_member_id))
                        # SR_TMP <
                        assert len(plot_setup_ijk.panels) == 1
                        panel_setup = next(iter(plot_setup_ijk.panels))
                        # SR_TMP >
                        fld_mem_time_i_lst = []
                        for dimensions in panel_setup.dimensions.decompress(
                            ["variable"]
                        ):
                            fld = (
                                read_flexpart_field(
                                    self.datafile(
                                        ens_member_id, datafile_fmt=datafile_fmt
                                    ),
                                    get_var_name_ref(dimensions, var_names_ref),
                                    dimensions,
                                    integrate=panel_setup.integrate,
                                    model="COSMO-2",  # SR_TMP
                                    add_ts0=False,
                                )
                                * scale_fld_ref  # SR_TODO Why?!?!?
                            )
                            fld_mem_time_i_lst.append(fld)
                        fld_mem_time_i = np.nansum(fld_mem_time_i_lst, axis=0)
                        fld_mem_time[-1].append(fld_mem_time_i)
                fld_lst.append(fct_reduce_mem(np.nansum(fld_mem_time, axis=0)))
            return np.array(fld_lst)

        fld_arr_ref = read_fld_ref(plot_setup_group)
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

    def run_concentration(self, datadir, ens_var, *, scale_fld_ref=1.0):  # noqa:F811
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
                "plot_variable": "concentration",
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
            scale_fld_ref=scale_fld_ref,
        )

    def run_deposition_tot(self, datadir, ens_var):  # noqa:F811
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
                "panels": {"plot_variable": "tot_deposition"},
            },
            ens_var=ens_var,
            fct_reduce_mem=fct_reduce_mem,
        )

    def test_ens_mean_concentration(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "mean", scale_fld_ref=3.0)

    def test_ens_probability_concentration(self, datadir):  # noqa:F811
        self.run_concentration(datadir, "probability", scale_fld_ref=3.0)

    def test_ens_mean_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "mean")

    def test_ens_max_deposition_tot(self, datadir):  # noqa:F811
        self.run_deposition_tot(datadir, "maximum")
