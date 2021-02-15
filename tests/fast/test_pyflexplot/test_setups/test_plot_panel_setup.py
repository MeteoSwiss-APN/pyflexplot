"""Test module ``pyflexplot.setup``."""
# Standard library
from typing import Any
from typing import Dict

# First-party
from pyflexplot.setups.plot_panel_setup import PlotPanelSetup
from srutils.testing import assert_is_sub_element


class Test_CompleteDimensions:
    raw_dimensions: Dict[str, Any] = {
        "time": {"name": "time", "size": 11},
        "rlon": {"name": "rlon", "size": 40},
        "rlat": {"name": "rlat", "size": 30},
        "level": {"name": "level", "size": 3},
        "nageclass": {"name": "nageclass", "size": 1},
        "noutrel": {"name": "numpoint", "size": 1},
        "numpoint": {"name": "numpoint", "size": 2},
        "nchar": {"name": "nchar", "size": 45},
    }
    species_ids = (1, 2)

    def test_time(self):
        setup = PlotPanelSetup.create({"dimensions": {"time": "*"}})
        assert setup.dimensions.time is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.time == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    def test_level(self):
        setup = PlotPanelSetup.create({"dimensions": {"level": "*"}})
        assert setup.dimensions.level is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.level == (0, 1, 2)

    def test_species_id(self):
        setup = PlotPanelSetup.create({"dimensions": {"species_id": "*"}})
        assert setup.dimensions.species_id is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.species_id == (1, 2)

    def test_others(self):
        setup = PlotPanelSetup.create(
            {"dimensions": {"nageclass": "*", "noutrel": "*", "numpoint": "*"}}
        )
        assert setup.dimensions.nageclass is None
        assert setup.dimensions.noutrel is None
        assert setup.dimensions.numpoint is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.nageclass == 0
        assert setup.dimensions.noutrel == 0
        assert setup.dimensions.numpoint == (0, 1)

    def test_completion_mode_all(self):
        dimensions = {
            "time": "*",
            "level": "*",
            "species_id": "*",
            "nageclass": "*",
            "noutrel": "*",
            "numpoint": "*",
        }
        setup = PlotPanelSetup.create(
            {"dimensions": dimensions, "dimensions_default": "all"}
        )
        assert setup.dimensions.time is None
        assert setup.dimensions.level is None
        assert setup.dimensions.species_id is None
        assert setup.dimensions.nageclass is None
        assert setup.dimensions.noutrel is None
        assert setup.dimensions.numpoint is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.time == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert setup.dimensions.level == (0, 1, 2)
        assert setup.dimensions.species_id == (1, 2)
        assert setup.dimensions.nageclass == 0
        assert setup.dimensions.noutrel == 0
        assert setup.dimensions.numpoint == (0, 1)

    def test_completion_mode_first(self):
        dimensions = {
            "time": "*",
            "level": "*",
            "species_id": "*",
            "nageclass": "*",
            "noutrel": "*",
            "numpoint": "*",
        }
        setup = PlotPanelSetup.create(
            {"dimensions": dimensions, "dimensions_default": "first"}
        )
        assert setup.dimensions.time is None
        assert setup.dimensions.level is None
        assert setup.dimensions.species_id is None
        assert setup.dimensions.nageclass is None
        assert setup.dimensions.noutrel is None
        assert setup.dimensions.numpoint is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.time == 0
        assert setup.dimensions.level == 0
        assert setup.dimensions.species_id == 1
        assert setup.dimensions.nageclass == 0
        assert setup.dimensions.noutrel == 0
        assert setup.dimensions.numpoint == 0


class TestDecompress:
    params = {
        "plot_variable": "concentration",
        "combine_levels": False,
        "combine_species": False,
        "dimensions": {
            "level": [0, 1],
            "nageclass": (0,),
            "noutrel": (0,),
            "numpoint": (0,),
            "species_id": [1, 2],
            "time": [1, 2, 3],
        },
    }

    def test_full(self):
        setup = PlotPanelSetup.create(self.params)
        group = setup.decompress()
        dcts = group.dicts()
        sol = [
            {"dimensions": {"level": 0, "species_id": 1, "time": 1}},
            {"dimensions": {"level": 0, "species_id": 1, "time": 2}},
            {"dimensions": {"level": 0, "species_id": 1, "time": 3}},
            {"dimensions": {"level": 0, "species_id": 2, "time": 1}},
            {"dimensions": {"level": 0, "species_id": 2, "time": 2}},
            {"dimensions": {"level": 0, "species_id": 2, "time": 3}},
            {"dimensions": {"level": 1, "species_id": 1, "time": 1}},
            {"dimensions": {"level": 1, "species_id": 1, "time": 2}},
            {"dimensions": {"level": 1, "species_id": 1, "time": 3}},
            {"dimensions": {"level": 1, "species_id": 2, "time": 1}},
            {"dimensions": {"level": 1, "species_id": 2, "time": 2}},
            {"dimensions": {"level": 1, "species_id": 2, "time": 3}},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip(self):
        setup = PlotPanelSetup.create(self.params)
        group = setup.decompress(skip=["species_id"])
        dcts = group.dicts()
        sol = [
            {"dimensions": {"level": 0, "species_id": [1, 2], "time": 1}},
            {"dimensions": {"level": 0, "species_id": [1, 2], "time": 2}},
            {"dimensions": {"level": 0, "species_id": [1, 2], "time": 3}},
            {"dimensions": {"level": 1, "species_id": [1, 2], "time": 1}},
            {"dimensions": {"level": 1, "species_id": [1, 2], "time": 2}},
            {"dimensions": {"level": 1, "species_id": [1, 2], "time": 3}},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select(self):
        setup = PlotPanelSetup.create(self.params)
        group = setup.decompress(select=["species_id"])
        dcts = group.dicts()
        sol = [
            {"dimensions": {"level": [0, 1], "species_id": 1, "time": [1, 2, 3]}},
            {"dimensions": {"level": [0, 1], "species_id": 2, "time": [1, 2, 3]}},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_skip(self):
        setup = PlotPanelSetup.create(self.params)
        group = setup.decompress(select=["time", "species_id"], skip=["level", "time"])
        dcts = group.dicts()
        sol = [
            {"dimensions": {"level": [0, 1], "species_id": 1, "time": [1, 2, 3]}},
            {"dimensions": {"level": [0, 1], "species_id": 2, "time": [1, 2, 3]}},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")
