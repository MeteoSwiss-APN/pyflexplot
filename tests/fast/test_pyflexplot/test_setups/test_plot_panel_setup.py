"""Test module ``pyflexplot.setup``."""
# Standard library
from typing import Any
from typing import Dict

# Third-party
import pytest

# First-party
from pyflexplot.setups.plot_panel_setup import PlotPanelSetup
from srutils.testing import assert_is_sub_element


class Test_Create:
    def test_combine_levels_ok(self):
        params = {"combine_levels": True, "dimensions": {"level": (1, 2)}}
        setup = PlotPanelSetup.create(params)
        res = setup.dict()
        assert_is_sub_element(
            name_sub="solution", obj_sub=params, name_super="result", obj_super=res
        )

    def test_combine_levels_fail(self):
        params = {"combine_levels": False, "dimensions": {"level": (1, 2)}}
        with pytest.raises(ValueError, match="combine_levels"):
            PlotPanelSetup.create(params)

    def test_combine_species_ok(self):
        params = {"combine_species": True, "dimensions": {"species_id": (1, 2)}}
        setup = PlotPanelSetup.create(params)
        res = setup.dict()
        assert_is_sub_element(
            name_sub="solution", obj_sub=params, name_super="result", obj_super=res
        )

    def test_combine_species_fail(self):
        params = {"combine_species": False, "dimensions": {"species_id": (1, 2)}}
        with pytest.raises(ValueError, match="combine_species"):
            PlotPanelSetup.create(params)

    @pytest.mark.skip(
        "consider exception when dimensions.variable passed to PlotPanelSetup.create"
    )
    def test_dimensions_variable_fail(self):
        params = {"dimensions": {"variable": "concentration"}}
        with pytest.raises(ValueError, match="variable"):
            PlotPanelSetup.create(params)


class Test_CompleteDimensions:
    raw_dimensions: Dict[str, Any] = {
        "time": {"name": "time", "size": 11},
        "rlon": {"name": "rlon", "size": 40},
        "rlat": {"name": "rlat", "size": 30},
        "level": {"name": "level", "size": 3},
        "nageclass": {"name": "nageclass", "size": 1},
        "release": {"name": "release", "size": 2},
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
            {"dimensions": {"nageclass": "*", "release": "*"}}
        )
        assert setup.dimensions.nageclass is None
        assert setup.dimensions.release is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.nageclass == 0
        assert setup.dimensions.release is None

    def test_completion_mode_all(self):
        dimensions = {
            "time": "*",
            "level": "*",
            "species_id": "*",
            "nageclass": "*",
            "release": "*",
        }
        setup = PlotPanelSetup.create(
            {"dimensions": dimensions, "dimensions_default": "all"}
        )
        assert setup.dimensions.time is None
        assert setup.dimensions.level is None
        assert setup.dimensions.species_id is None
        assert setup.dimensions.nageclass is None
        assert setup.dimensions.release is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.time == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert setup.dimensions.level == (0, 1, 2)
        assert setup.dimensions.species_id == (1, 2)
        assert setup.dimensions.nageclass == 0
        assert setup.dimensions.release is None

    def test_completion_mode_first(self):
        dimensions = {
            "time": "*",
            "level": "*",
            "species_id": "*",
            "nageclass": "*",
            "release": "*",
        }
        setup = PlotPanelSetup.create(
            {"dimensions": dimensions, "dimensions_default": "first"}
        )
        assert setup.dimensions.time is None
        assert setup.dimensions.level is None
        assert setup.dimensions.species_id is None
        assert setup.dimensions.nageclass is None
        assert setup.dimensions.release is None
        setup = setup.complete_dimensions(self.raw_dimensions, self.species_ids)
        assert setup.dimensions.time == 0
        assert setup.dimensions.level == 0
        assert setup.dimensions.species_id == 1
        assert setup.dimensions.nageclass == 0
        assert setup.dimensions.release is None


class Test_Decompress:
    params = {
        "plot_variable": "concentration",
        "combine_levels": True,
        "combine_species": True,
        "dimensions": {
            "level": [0, 1],
            "nageclass": (0,),
            "release": (0,),
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
