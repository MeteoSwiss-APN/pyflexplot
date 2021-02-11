"""Tests for ``pyflexplot.setups.plot_panel_setup.PlotPanelSetupGroup``."""
# First-party
from pyflexplot.setups.plot_panel_setup import PlotPanelSetup
from pyflexplot.setups.plot_panel_setup import PlotPanelSetupGroup
from srutils.testing import assert_is_sub_element


class TestDecompress:
    params = {
        "plot_variable": "deposition",
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

    def test_internal_vs_default(self):
        """The option ``internal`` is true by default."""
        group = PlotPanelSetupGroup.create([self.params])
        group_internal = group.decompress(internal=True)
        group_default = group.decompress()
        assert group_internal == group_default

    def test_internal_vs_external(self):
        """Check types and content with option ``internal`` true or false."""
        group = PlotPanelSetupGroup.create([self.params])
        group_internal = group.decompress(internal=True)
        groups_external = group.decompress(internal=False)
        assert isinstance(group_internal, PlotPanelSetupGroup)
        assert isinstance(groups_external, list)
        assert all(isinstance(group, PlotPanelSetupGroup) for group in groups_external)
        assert all(len(group) == 1 for group in groups_external)
        dcts_internal = group_internal.dicts()
        dcts_external = [setup.dict() for group in groups_external for setup in group]
        assert dcts_internal == dcts_external

    def test_full(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        group_out = group_in.decompress()
        dcts = group_out.dicts()
        sol = [
            {"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 1}},
            {"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 2}},
            {"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 3}},
            {"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 1}},
            {"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 2}},
            {"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 3}},
            {"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 1}},
            {"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 2}},
            {"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 3}},
            {"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 1}},
            {"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 2}},
            {"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 3}},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_full_internal(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        groups_out = group_in.decompress(internal=False)
        assert isinstance(groups_out, list)
        assert all(isinstance(group, PlotPanelSetupGroup) for group in groups_out)
        group_out = group_in.decompress(internal=True)
        assert isinstance(group_out, PlotPanelSetupGroup)
        assert all(isinstance(setup, PlotPanelSetup) for setup in group_out)

    def test_skip(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        group_out = group_in.decompress(skip=["time"])
        dcts = group_out.dicts()
        sol = [
            {
                "dimensions": {
                    "deposition_type": "dry",
                    "species_id": 1,
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "dry",
                    "species_id": 2,
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "wet",
                    "species_id": 1,
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "wet",
                    "species_id": 2,
                    "time": [1, 2, 3],
                }
            },
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip_all(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        group_out = group_in.decompress(skip=["deposition_type", "species_id", "time"])
        dcts = group_out.dicts()
        sol = [
            {
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
                    "species_id": [1, 2],
                    "time": [1, 2, 3],
                }
            },
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        group_out = group_in.decompress(select=["deposition_type"])
        dcts = group_out.dicts()
        sol = [
            {
                "dimensions": {
                    "deposition_type": "dry",
                    "species_id": [1, 2],
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "wet",
                    "species_id": [1, 2],
                    "time": [1, 2, 3],
                }
            },
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_all(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        group_out = group_in.decompress(
            select=["deposition_type", "species_id", "time"]
        )
        dcts = group_out.dicts()
        sol = [
            {"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 1}},
            {"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 2}},
            {"dimensions": {"deposition_type": "dry", "species_id": 1, "time": 3}},
            {"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 1}},
            {"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 2}},
            {"dimensions": {"deposition_type": "dry", "species_id": 2, "time": 3}},
            {"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 1}},
            {"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 2}},
            {"dimensions": {"deposition_type": "wet", "species_id": 1, "time": 3}},
            {"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 1}},
            {"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 2}},
            {"dimensions": {"deposition_type": "wet", "species_id": 2, "time": 3}},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_skip(self):
        group_in = PlotPanelSetupGroup.create([self.params])
        group_out = group_in.decompress(
            select=["deposition_type", "species_id", "time"], skip=["time"]
        )
        dcts = group_out.dicts()
        sol = [
            {
                "dimensions": {
                    "deposition_type": "dry",
                    "species_id": 1,
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "dry",
                    "species_id": 2,
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "wet",
                    "species_id": 1,
                    "time": [1, 2, 3],
                }
            },
            {
                "dimensions": {
                    "deposition_type": "wet",
                    "species_id": 2,
                    "time": [1, 2, 3],
                }
            },
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")


class TestDecompressExternal:
    params = {
        "plot_variable": "concentration",
        "combine_levels": False,
        "combine_species": False,
        "dimensions": {
            "level": [4, 8],
            "nageclass": (0,),
            "noutrel": (0,),
            "numpoint": (0,),
            "species_id": [1, 2],
            "time": [1, 2, 3],
        },
    }

    def test_full(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(internal=False)
        dcts = [group.dicts() for group in groups]
        sol = [
            [{"dimensions": {"level": 4, "species_id": 1, "time": 1}}],
            [{"dimensions": {"level": 4, "species_id": 1, "time": 2}}],
            [{"dimensions": {"level": 4, "species_id": 1, "time": 3}}],
            [{"dimensions": {"level": 4, "species_id": 2, "time": 1}}],
            [{"dimensions": {"level": 4, "species_id": 2, "time": 2}}],
            [{"dimensions": {"level": 4, "species_id": 2, "time": 3}}],
            [{"dimensions": {"level": 8, "species_id": 1, "time": 1}}],
            [{"dimensions": {"level": 8, "species_id": 1, "time": 2}}],
            [{"dimensions": {"level": 8, "species_id": 1, "time": 3}}],
            [{"dimensions": {"level": 8, "species_id": 2, "time": 1}}],
            [{"dimensions": {"level": 8, "species_id": 2, "time": 2}}],
            [{"dimensions": {"level": 8, "species_id": 2, "time": 3}}],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(internal=False, skip=["time"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "level": 4,
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 4,
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 8,
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 8,
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip_all(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(internal=False, skip=["level", "species_id", "time"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "level": [4, 8],
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(internal=False, select=["level"])
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "level": 4,
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 8,
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_all(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(
            internal=False, select=["level", "species_id", "time"]
        )
        dcts = [group.dicts() for group in groups]
        sol = [
            [{"dimensions": {"level": 4, "species_id": 1, "time": 1}}],
            [{"dimensions": {"level": 4, "species_id": 1, "time": 2}}],
            [{"dimensions": {"level": 4, "species_id": 1, "time": 3}}],
            [{"dimensions": {"level": 4, "species_id": 2, "time": 1}}],
            [{"dimensions": {"level": 4, "species_id": 2, "time": 2}}],
            [{"dimensions": {"level": 4, "species_id": 2, "time": 3}}],
            [{"dimensions": {"level": 8, "species_id": 1, "time": 1}}],
            [{"dimensions": {"level": 8, "species_id": 1, "time": 2}}],
            [{"dimensions": {"level": 8, "species_id": 1, "time": 3}}],
            [{"dimensions": {"level": 8, "species_id": 2, "time": 1}}],
            [{"dimensions": {"level": 8, "species_id": 2, "time": 2}}],
            [{"dimensions": {"level": 8, "species_id": 2, "time": 3}}],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_skip(self):
        group = PlotPanelSetupGroup.create([self.params])
        groups = group.decompress(
            internal=False,
            select=["level", "species_id", "time"],
            skip=["time"],
        )
        dcts = [group.dicts() for group in groups]
        sol = [
            [
                {
                    "dimensions": {
                        "level": 4,
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 4,
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 8,
                        "species_id": 1,
                        "time": [1, 2, 3],
                    }
                }
            ],
            [
                {
                    "dimensions": {
                        "level": 8,
                        "species_id": 2,
                        "time": [1, 2, 3],
                    }
                }
            ],
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")
