"""Tests for ``pyflexplot.setups.plot_panel_setup.PlotPanelSetupGroup``."""
# First-party
from pyflexplot.setups.plot_panel_setup import PlotPanelSetupGroup
from srutils.testing import assert_is_sub_element


class TestDecompress:
    def test(self):
        params = {
            "input_variable": "deposition",
            "combine_deposition_types": False,
            "combine_species": False,
            "dimensions": {
                "deposition_type": ["dry", "wet"],
                "nageclass": (0,),
                "noutrel": (0,),
                "numpoint": (0,),
                "species_id": [1, 2],
                "time": [1, 2, 3],
            },
        }
        group = PlotPanelSetupGroup.create([params])
        groups = group.decompress()
        dcts = [group.dicts() for group in groups]
        sol = [
            [{"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 1}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 2}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 3}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 1}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 2}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 3}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 1}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 2}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 3}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 1}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 2}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 3}}],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(skip=["time"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "deposition_type": "dry",
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "dry",
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "wet",
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "wet",
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip_all(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(skip=["deposition_type", "species_id", "time"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "deposition_type": ["dry", "wet"],
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(select=["deposition_type"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "deposition_type": "dry",
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "wet",
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_all(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(select=["deposition_type", "species_id", "time"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [{"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 1}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 2}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 3}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 1}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 2}}],
            [{"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 3}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 1}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 2}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 3}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 1}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 2}}],
            [{"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 3}}],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_skip(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(
            select=["deposition_type", "species_id", "time"], skip=["time"]
        )
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "deposition_type": "dry",
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "dry",
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "wet",
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "deposition_type": "wet",
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")
