# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup``.
"""
# Standard library
from typing import Any
from typing import Dict
from typing import List

# First-party
from pyflexplot.setup import CoreInputSetup
from pyflexplot.setup import CoreInputSetupCollection
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection
from srutils.testing import check_summary_dict_is_subdict

# Local
from .shared import DEFAULT_KWARGS
from .shared import DEFAULT_SETUP


class Test_InputSetup_Create:
    def test_some_dimensions(self):
        params = {
            "infile": "dummy.nc",
            "integrate": False,
            "outfile": "dummy.png",
            "ens_variable": "mean",
            "dimensions": {"time": 10, "level": 1},
            "input_variable": "concentration",
            "ens_member_id": (0, 1, 5, 10, 15, 20),
        }
        setup = InputSetup.create(params)
        res = setup.dict()
        check_summary_dict_is_subdict(superdict=res, subdict=params)


class Test_InputSetup_Decompress:
    @property
    def setup(self):
        return DEFAULT_SETUP.derive(
            {
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "dimensions": {
                    "time": [1, 2, 3],
                    "species_id": [1, 2],
                    "nageclass": (0,),
                    "noutrel": (0,),
                    "numpoint": (0,),
                },
            },
        )

    def test_full(self):
        """Decompress all params."""
        setups = self.setup.decompress()
        assert len(setups) == 12
        assert isinstance(setups, CoreInputSetupCollection)
        assert all(isinstance(setup, CoreInputSetup) for setup in setups)
        res = {
            (s.deposition_type, s.dimensions.species_id, s.dimensions.time)
            for s in setups
        }
        sol = {
            ("dry", 1, 1),
            ("dry", 1, 2),
            ("dry", 1, 3),
            ("dry", 2, 1),
            ("dry", 2, 2),
            ("dry", 2, 3),
            ("wet", 1, 1),
            ("wet", 1, 2),
            ("wet", 1, 3),
            ("wet", 2, 1),
            ("wet", 2, 2),
            ("wet", 2, 3),
        }
        assert res == sol

    def test_full_with_partially(self):
        """Decompress all params."""
        setups = self.setup.decompress_partially(None)
        assert len(setups) == 12
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        res = {
            (s.deposition_type, s.dimensions.species_id, s.dimensions.time)
            for s in setups
        }
        sol = {
            (("dry",), 1, 1),
            (("dry",), 1, 2),
            (("dry",), 1, 3),
            (("dry",), 2, 1),
            (("dry",), 2, 2),
            (("dry",), 2, 3),
            (("wet",), 1, 1),
            (("wet",), 1, 2),
            (("wet",), 1, 3),
            (("wet",), 2, 1),
            (("wet",), 2, 2),
            (("wet",), 2, 3),
        }
        assert res == sol

    def test_partially_one(self):
        """Decompress only one select parameter."""
        setups = self.setup.decompress_partially(["dimensions.species_id"])
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        assert len(setups) == 2
        res = {
            (s.deposition_type, s.dimensions.species_id, s.dimensions.time)
            for s in setups
        }
        sol = {(("dry", "wet"), 1, (1, 2, 3)), (("dry", "wet"), 2, (1, 2, 3))}
        assert res == sol

    def test_select_two(self):
        """Decompress two select parameters."""
        setups = self.setup.decompress_partially(["dimensions.time", "deposition_type"])
        assert len(setups) == 6
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        res = {
            (s.deposition_type, s.dimensions.species_id, s.dimensions.time)
            for s in setups
        }
        sol = {
            (("dry",), (1, 2), 1),
            (("dry",), (1, 2), 2),
            (("dry",), (1, 2), 3),
            (("wet",), (1, 2), 1),
            (("wet",), (1, 2), 2),
            (("wet",), (1, 2), 3),
        }
        assert res == sol


class Test_InputSetupCollection_Create:
    def create_partial_dicts(self):
        return [
            {**DEFAULT_KWARGS, **dct}
            for dct in [
                {"infile": "foo.nc", "input_variable": "concentration", "domain": "ch"},
                {"infile": "bar.nc", "input_variable": "deposition", "lang": "de"},
                {"dimensions": {"nageclass": 1, "noutrel": 5, "numpoint": 3}},
            ]
        ]

    def create_complete_dicts(self):
        return [{**DEFAULT_SETUP.dict(), **dct} for dct in self.create_partial_dicts()]

    def create_setup_lst(self):
        return [InputSetup.create(dct) for dct in self.create_partial_dicts()]

    def create_setups(self):
        return InputSetupCollection(self.create_setup_lst())

    def test_create_empty(self):
        setups = InputSetupCollection([])
        assert len(setups) == 0

    def test_from_setups(self):
        partial_dicts = self.create_partial_dicts()
        setups = InputSetupCollection.create(partial_dicts)
        assert len(setups) == len(partial_dicts)


class Test_InputSetupCollection_Compress:
    dcts: List[Dict[str, Any]] = [
        {**DEFAULT_KWARGS, **dct}
        for dct in [
            {
                "infile": "foo.nc",
                "input_variable": "concentration",
                "dimensions": {"level": 0},
            },
            {
                "infile": "foo.nc",
                "input_variable": "concentration",
                "dimensions": {"level": 1},
            },
            {
                "infile": "foo.nc",
                "input_variable": "concentration",
                "dimensions": {"level": (1, 2)},
            },
        ]
    ]
    setups_lst = [InputSetup.create(dct) for dct in dcts]

    def test_one(self):
        res = InputSetupCollection(self.setups_lst[:1]).compress().dict()
        sol = self.setups_lst[0]
        assert res == sol

    def test_two(self):
        res = InputSetupCollection(self.setups_lst[:2]).compress().dict()
        sol = InputSetup.create(self.dcts[0]).derive({"dimensions": {"level": (0, 1)}})
        assert res == sol

    def test_three(self):
        res = InputSetupCollection(self.setups_lst[:3]).compress().dict()
        sol = InputSetup.create(self.dcts[0]).derive(
            {"dimensions": {"level": (0, 1, 2)}}
        )
        assert res == sol
