"""Test class ``pyflexplot.setup.CoreDimensions``."""
# First-party
from pyflexplot.setups.dimensions import CoreDimensions


class Test_Init:
    """Initialize ``CoreDimensions`` objects."""

    def test_no_args(self):
        cdims = CoreDimensions()
        res = cdims.dict()
        sol = {
            "level": None,
            "nageclass": None,
            "noutrel": None,
            "numpoint": None,
            "species_id": None,
            "time": None,
            "variable": None,
        }
        assert res == sol

    def test_all_args(self):
        params = {
            "level": 2,
            "nageclass": 0,
            "noutrel": 1,
            "numpoint": 3,
            "species_id": 2,
            "time": 0,
            "variable": "dry_deposition",
        }
        cdims = CoreDimensions(**params)
        res = cdims.dict()
        sol = params
        assert res == sol

    def test_some_args(self):
        params = {
            "noutrel": 1,
            "species_id": 2,
            "variable": "concentration",
        }
        cdims = CoreDimensions(**params)
        res = cdims.dict()
        sol = {
            "level": None,
            "nageclass": None,
            "noutrel": 1,
            "numpoint": None,
            "species_id": 2,
            "time": None,
            "variable": "concentration",
        }
        assert res == sol
