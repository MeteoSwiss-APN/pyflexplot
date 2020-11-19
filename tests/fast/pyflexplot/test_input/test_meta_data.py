"""Tests for module ``pyflexplot.input.meta_data``."""
# Standard library
import datetime as dt
from typing import Any
from typing import Dict
from typing import List

# First-party
from pyflexplot.input.meta_data import MetaData
from pyflexplot.input.meta_data import ReleaseMetaData
from pyflexplot.input.meta_data import SimulationMetaData
from pyflexplot.input.meta_data import SpeciesMetaData
from pyflexplot.input.meta_data import VariableMetaData


def expand_singles(dct: Dict[Any, Any], n: int) -> List[Dict[Any, Any]]:
    """Expand dict with multi-element values into single-element dicts."""
    dcts: List[Dict[Any, Any]] = [{} for _ in range(n)]
    for key, raw_value in dct.items():
        if isinstance(raw_value, tuple):
            if len(raw_value) != n:
                raise Exception(
                    f"value of element {repr(key)} is a tuple with the wrong length"
                    f": expected {n}, got {len(raw_value)}: {raw_value}"
                )
            values = list(raw_value)
        else:
            values = [raw_value] * n
        for dct_i, value in zip(dcts, values):
            dct_i[key] = value
    return dcts


class TestMerge:
    raw_release_params = {
        "duration": dt.timedelta(seconds=28800),
        "duration_unit": "s",
        "end_rel": dt.timedelta(seconds=28800),
        "height": 100.0,
        "height_unit": "meters",
        "lat": 47.37,
        "lon": 7.97,
        "mass": (1e9, 2.88e11, 1e9, 2.88e11, 1e9, 2.88e11),
        "mass_unit": "Bq",
        "site": r"G$\mathrm{\"o}$sgen",
        "rate": (34722, 1e7, 34722, 1e7, 34722, 1e7),
        "rate_unit": "Bq s-1",
        "start_rel": dt.timedelta(0),
    }
    raw_simulation_params = {
        "base_time": dt.datetime(2019, 7, 27, 9, 0, tzinfo=dt.timezone.utc),
        "start": dt.datetime(2019, 7, 27, 12, 0, tzinfo=dt.timezone.utc),
        "end": dt.datetime(2019, 7, 28, 21, 0, tzinfo=dt.timezone.utc),
        "now": dt.datetime(2019, 7, 27, 21, 0, tzinfo=dt.timezone.utc),
        "now_rel": dt.timedelta(seconds=32400),
        "lead_time": dt.timedelta(seconds=32400),
        "reduction_start": dt.datetime(2019, 7, 27, 12, 0, tzinfo=dt.timezone.utc),
        "reduction_start_rel": dt.timedelta(seconds=0),
    }
    raw_species_params = {
        "name": ("Cs-137", "I-131a", "Cs-137", "I-131a", "Cs-137", "I-131a"),
        "half_life": (30.0, 8.04, 30.0, 8.04, 30.0, 8.04),
        "half_life_unit": ("a", "d", "a", "d", "a", "d"),
        "deposition_velocity": 0.0015,
        "deposition_velocity_unit": "m s-1",
        "sedimentation_velocity": 0.0,
        "sedimentation_velocity_unit": "m s-1",
        "washout_coefficient": 7e-05,
        "washout_coefficient_unit": "s-1",
        "washout_exponent": 0.8,
    }
    raw_variable_params = {
        "unit": "ng kg-1",
        "bottom_level": (0, 0, 500, 500, 2000, 2000),
        "top_level": (500, 500, 2000, 2000, 10000, 10000),
        "level_unit": "meters",
    }
    n = 6

    def create_merged_mdata(self) -> MetaData:
        mdata_lst: List[MetaData] = []
        for release_params, simulation_params, species_params, variable_params in zip(
            expand_singles(self.raw_release_params, self.n),
            expand_singles(self.raw_simulation_params, self.n),
            expand_singles(self.raw_species_params, self.n),
            expand_singles(self.raw_variable_params, self.n),
        ):
            mdata = MetaData(
                release=ReleaseMetaData(**release_params),
                simulation=SimulationMetaData(**simulation_params),
                species=SpeciesMetaData(**species_params),
                variable=VariableMetaData(**variable_params),
            )
            mdata_lst.append(mdata)
        assert len(mdata_lst) == self.n
        return mdata_lst[0].merge_with(mdata_lst[1:])

    def test_release(self):
        merged_mdata = self.create_merged_mdata()
        merged_release_params = merged_mdata.release.dict()
        sol_release = {
            "duration": dt.timedelta(seconds=28800),
            "duration_unit": "s",
            "end_rel": dt.timedelta(seconds=28800),
            "height": 100.0,
            "height_unit": "meters",
            "lat": 47.37,
            "lon": 7.97,
            "mass": (1e9, 2.88e11),
            "mass_unit": ("Bq", "Bq"),
            "site": r"G$\mathrm{\"o}$sgen",
            "rate": (34722, 1e7),
            "rate_unit": ("Bq s-1", "Bq s-1"),
            "start_rel": dt.timedelta(0),
        }
        assert merged_release_params == sol_release

    def test_simulation(self):
        merged_mdata = self.create_merged_mdata()
        merged_simulation_params = merged_mdata.simulation.dict()
        sol_simulation = {
            "base_time": dt.datetime(2019, 7, 27, 9, 0, tzinfo=dt.timezone.utc),
            "start": dt.datetime(2019, 7, 27, 12, 0, tzinfo=dt.timezone.utc),
            "end": dt.datetime(2019, 7, 28, 21, 0, tzinfo=dt.timezone.utc),
            "now": dt.datetime(2019, 7, 27, 21, 0, tzinfo=dt.timezone.utc),
            "now_rel": dt.timedelta(seconds=32400),
            "lead_time": dt.timedelta(seconds=32400),
            "reduction_start": dt.datetime(2019, 7, 27, 12, 0, tzinfo=dt.timezone.utc),
            "reduction_start_rel": dt.timedelta(seconds=0),
        }
        assert merged_simulation_params == sol_simulation

    def test_species(self):
        merged_mdata = self.create_merged_mdata()
        merged_species_params = merged_mdata.species.dict()
        sol_species = {
            "name": ("Cs-137", "I-131a"),
            "half_life": (30.0, 8.04),
            "half_life_unit": ("a", "d"),
            "deposition_velocity": (0.0015, 0.0015),
            "deposition_velocity_unit": ("m s-1", "m s-1"),
            "sedimentation_velocity": (0.0, 0.0),
            "sedimentation_velocity_unit": ("m s-1", "m s-1"),
            "washout_coefficient": (7e-05, 7e-05),
            "washout_coefficient_unit": ("s-1", "s-1"),
            "washout_exponent": (0.8, 0.8),
        }
        assert merged_species_params == sol_species

    def test_variable(self):
        merged_mdata = self.create_merged_mdata()
        merged_variable_params = merged_mdata.variable.dict()
        sol_variable = {
            "unit": "ng kg-1",
            "bottom_level": 0,
            "top_level": 10000,
            "level_unit": "meters",
        }
        assert merged_variable_params == sol_variable
