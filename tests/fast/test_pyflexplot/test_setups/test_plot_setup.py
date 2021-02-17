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
        check_is_sub_element(obj_super=res, obj_sub=params)

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


class Test_Decompress:
    @property
    def setup(self):
        return DEFAULT_SETUP.derive(
            {
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "combine_species": True,
                        "dimensions": {
                            "nageclass": (0,),
                            "noutrel": (0,),
                            "numpoint": (0,),
                            "species_id": [1, 2],
                            "time": [1, 2, 3],
                        },
                    }
                ],
            },
        )

    def test_full_fail(self):
        """Full decompression fails with internal=True due to ens_member_id."""
        with pytest.raises(ValueError, match="ens_member_id"):
            self.setup.decompress()

    def test_full_external(self):
        """Decompress all params."""
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
