"""Test module ``pyflexplot.setup``."""
# Third-party
import pytest

# First-party
from pyflexplot.setups.plot_setup import PlotSetup
from pyflexplot.setups.plot_setup import PlotSetupGroup
from srutils.dict import merge_dicts
from srutils.testing import check_is_sub_element

# Local
from .shared import DEFAULT_SETUP


class Test_Create:
    base_params = {
        "infile": "dummy.nc",
        "outfile": "dummy.png",
        "model": {"name": "COSMO-1"},
    }

    def check_hashable(self, setup):
        """Check some properties of the setup object."""
        try:
            hash(setup)
        except TypeError as e:
            raise AssertionError(f"setup object is not hashable:\n{setup}") from e

    def test_some_dimensions(self):
        params = {
            "model": {
                "ens_member_id": (0, 1, 5, 10, 15, 20),
            },
            "panels": [
                {
                    "integrate": False,
                    "ens_variable": "mean",
                    "dimensions": {"time": 10, "level": 1},
                    "plot_variable": "concentration",
                }
            ],
        }
        params = merge_dicts(self.base_params, params, overwrite_seqs=True)
        setup = PlotSetup.create(params)
        res = setup.dict()
        sol = params
        check_is_sub_element(
            name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
        )
        self.check_hashable(setup)

    @pytest.mark.parametrize(
        "dct",
        [
            {
                "panels": [{"plot_variable": ["concentration", "tot_deposition"]}]
            },  # [dct0]
            {"panels": [{"ens_variable": ["minimum", "maximum"]}]},  # [dct1]
            {"panels": [{"integrate": [True, False]}]},  # [dct2]
            {"panels": [{"combine_levels": [True, False]}]},  # [dct3]
            {"panels": [{"combine_species": [True, False]}]},  # [dct4]
            {"panels": [{"lang": ["en", "de"]}]},  # [dct5]
            {"panels": [{"domain": ["full", "ch"]}]},  # [dct6]
            {"panels": [{"domain_size_lat": [None, 100]}]},  # [dct7]
            {"panels": [{"domain_size_lon": [None, 100]}]},  # [dct8]
            {"panels": [{"dimensions_default": ["all", "first"]}]},  # [dct9]
        ],
    )
    def test_multiple_values_fail(self, dct):
        params = merge_dicts(self.base_params, dct)
        with pytest.raises(ValueError):
            PlotSetup.create(params)

    def test_multipanel(self):
        params = {
            "layout": {
                "plot_type": "multipanel",
                "multipanel_param": "ens_variable",
            },
            "panels": {"ens_variable": ["minimum", "maximum", "mean", "median"]},
        }
        params = merge_dicts(self.base_params, params, overwrite_seqs=True)
        setup = PlotSetup.create(params)
        res = setup.dict()
        sol = {
            "panels": [
                {"ens_variable": "minimum"},
                {"ens_variable": "maximum"},
                {"ens_variable": "mean"},
                {"ens_variable": "median"},
            ]
        }
        check_is_sub_element(
            name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
        )
        self.check_hashable(setup)


class Test_Derive:
    def test_panels_list(self):
        params = {"panels": [{"plot_variable": "tot_deposition"}]}
        setup = DEFAULT_SETUP.derive(params)
        sol = params
        res = setup.dict()
        check_is_sub_element(
            name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
        )

    def test_panels_dict(self):
        params = {"panels": {"plot_variable": "tot_deposition"}}
        setup = DEFAULT_SETUP.derive(params)
        sol = {"panels": [params["panels"]]}
        res = setup.dict()
        check_is_sub_element(
            name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
        )

    def test_multipanel(self):
        params = {
            "layout": {
                "plot_type": "multipanel",
                "multipanel_param": "ens_variable",
            },
            "panels": {"ens_variable": ["minimum", "maximum", "mean", "median"]},
        }
        setup = DEFAULT_SETUP.derive(params)
        sol = {
            "panels": [
                {"ens_variable": "minimum"},
                {"ens_variable": "maximum"},
                {"ens_variable": "mean"},
                {"ens_variable": "median"},
            ]
        }
        res = setup.dict()
        check_is_sub_element(
            name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
        )


class Test_Decompress:
    class Test_Base:
        @property
        def setup(self):
            params = {
                "panels": {
                    "plot_variable": "tot_deposition",
                    "combine_species": True,
                    "dimensions": {
                        "nageclass": (0,),
                        "noutrel": (0,),
                        "numpoint": (0,),
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    },
                },
            }
            return DEFAULT_SETUP.derive(params)

        def test_full_internal(self):
            """Full decompression fails with internal=True due to ens_member_id."""
            group = self.setup.decompress(internal=True)
            assert isinstance(group, PlotSetupGroup)
            res = {
                (
                    s.panels.collect_equal("dimensions").variable,
                    s.panels.collect_equal("dimensions").species_id,
                    s.panels.collect_equal("dimensions").time,
                )
                for s in group
            }
            sol = {
                (("dry_deposition", "wet_deposition"), 1, 1),
                (("dry_deposition", "wet_deposition"), 1, 2),
                (("dry_deposition", "wet_deposition"), 1, 3),
                (("dry_deposition", "wet_deposition"), 2, 1),
                (("dry_deposition", "wet_deposition"), 2, 2),
                (("dry_deposition", "wet_deposition"), 2, 3),
            }
            assert res == sol

        def test_full_external(self):
            """Decompress all params into separate setups."""
            setups = self.setup.decompress(internal=False)
            assert len(setups) == 6
            res = {
                (
                    s.panels.collect_equal("dimensions").variable,
                    s.panels.collect_equal("dimensions").species_id,
                    s.panels.collect_equal("dimensions").time,
                )
                for s in setups
            }
            sol = {
                (("dry_deposition", "wet_deposition"), 1, 1),
                (("dry_deposition", "wet_deposition"), 1, 2),
                (("dry_deposition", "wet_deposition"), 1, 3),
                (("dry_deposition", "wet_deposition"), 2, 1),
                (("dry_deposition", "wet_deposition"), 2, 2),
                (("dry_deposition", "wet_deposition"), 2, 3),
            }
            assert res == sol

        def test_partially_one(self):
            """Decompress only one select parameter."""
            setups = self.setup.decompress(["dimensions.species_id"])
            assert isinstance(setups, PlotSetupGroup)
            assert all(isinstance(setup, PlotSetup) for setup in setups)
            assert len(setups) == 2
            res = {
                (
                    s.panels.collect_equal("dimensions").variable,
                    s.panels.collect_equal("dimensions").species_id,
                    s.panels.collect_equal("dimensions").time,
                )
                for s in setups
            }
            sol = {
                (("dry_deposition", "wet_deposition"), 1, (1, 2, 3)),
                (("dry_deposition", "wet_deposition"), 2, (1, 2, 3)),
            }
            assert res == sol

        def test_select_two(self):
            """Decompress two select parameters."""
            setups = self.setup.decompress(["dimensions.time", "dimensions.species_id"])
            assert len(setups) == 6
            assert isinstance(setups, PlotSetupGroup)
            assert all(isinstance(setup, PlotSetup) for setup in setups)
            res = {
                (
                    s.panels.collect_equal("dimensions").variable,
                    s.panels.collect_equal("dimensions").species_id,
                    s.panels.collect_equal("dimensions").time,
                )
                for s in setups
            }
            sol = {
                (("dry_deposition", "wet_deposition"), 1, 1),
                (("dry_deposition", "wet_deposition"), 1, 2),
                (("dry_deposition", "wet_deposition"), 1, 3),
                (("dry_deposition", "wet_deposition"), 2, 1),
                (("dry_deposition", "wet_deposition"), 2, 2),
                (("dry_deposition", "wet_deposition"), 2, 3),
            }
            assert res == sol

        def test_select_variable_fails(self):
            """``Dimensions.variable`` cannot be selected for decompression."""
            msg = r"^cannot decompress Dimensions.variable"
            with pytest.raises(ValueError, match=msg):
                self.setup.decompress(["dimensions.variable"])

        def test_model(self):
            """Decompress model parameters."""
            setup = self.setup.derive({"model": {"ens_member_id": (0, 1, 2)}})
            setups = setup.decompress(["model.ens_member_id"], internal=False)
            assert len(setups) == 3
            assert isinstance(setups, list)
            assert all(isinstance(setup, PlotSetup) for setup in setups)
            res = {setup.model.ens_member_id for setup in setups}
            sol = {(0,), (1,), (2,)}
            assert res == sol

    class Test_Ensemble:
        @property
        def setup(self):
            params = {
                "model": {"ens_member_id": [1, 2, 3, 4, 5]},
                "panels": {
                    "plot_variable": "cloud_arrival_time",
                    "ens_variable": "median",
                    "dimensions": {
                        "time": [0, 1, 2],
                    },
                },
            }
            return DEFAULT_SETUP.derive(params)

        def test_internal_fail(self):
            with pytest.raises(ValueError, match="ens_member_id"):
                self.setup.decompress(internal=True)

        def test_external(self):
            setups = self.setup.decompress(internal=False)
            assert isinstance(setups, list)
            res = [setup.dict() for setup in setups]
            sol = [
                {
                    "model": {"ens_member_id": (ens_member_id,)},
                    "panels": [{"dimensions": {"time": time}}],
                }
                for ens_member_id in [1, 2, 3, 4, 5]
                for time in [0, 1, 2]
            ]
            check_is_sub_element(
                name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
            )

        def test_internal_skip(self):
            setups = self.setup.decompress(skip=["ens_member_id"], internal=True)
            res = setups.dicts()
            sol = [
                {
                    "model": {"ens_member_id": (1, 2, 3, 4, 5)},
                    "panels": [{"dimensions": {"time": time}}],
                }
                for time in [0, 1, 2]
            ]
            check_is_sub_element(
                name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
            )

        def test_external_skip(self):
            setups = self.setup.decompress(skip=["ens_member_id"], internal=False)
            assert isinstance(setups, list)
            res = [setup.dict() for setup in setups]
            sol = [
                {
                    "model": {"ens_member_id": (1, 2, 3, 4, 5)},
                    "panels": [{"dimensions": {"time": time}}],
                }
                for time in [0, 1, 2]
            ]
            check_is_sub_element(
                name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
            )

    class Test_MultipanelEnsVariable:
        """Decompress, but skip multipanel param."""

        params = {
            "layout": {
                "plot_type": "multipanel",
                "multipanel_param": "ens_variable",
            },
            "model": {"ens_member_id": [1, 2, 3]},
            "panels": {"ens_variable": ["minimum", "maximum", "mean", "median"]},
        }

        @property
        def setup(self) -> PlotSetup:
            return DEFAULT_SETUP.derive(self.params)

        # @pytest.mark.skip("WIP")
        def test_skip_internal(self):
            group = self.setup.decompress(skip=["ens_member_id"], internal=True)
            assert isinstance(group, PlotSetupGroup)
            assert len(group) == 1
            plot_setup = next(iter(group))
            res = plot_setup.dict()
            sol = {
                **self.params,
                "panels": [
                    {"ens_variable": var}
                    for var in self.params["panels"]["ens_variable"]
                ],
            }
            check_is_sub_element(
                name_super="result", obj_super=res, name_sub="solution", obj_sub=sol
            )
