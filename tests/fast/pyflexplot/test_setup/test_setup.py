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

# Local
from .shared import DEFAULT_KWARGS
from .shared import DEFAULT_SETUP


class Test_Decompress:
    @property
    def setup(self):
        return DEFAULT_SETUP.derive(
            {
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "species_id": [1, 2],
                "time": [1, 2, 3],
                "dimensions": {
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
        res = {(s.deposition_type, s.species_id, s.time) for s in setups}
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
        res = {(s.deposition_type, s.species_id, s.time) for s in setups}
        sol = {
            (("dry",), (1,), (1,)),
            (("dry",), (1,), (2,)),
            (("dry",), (1,), (3,)),
            (("dry",), (2,), (1,)),
            (("dry",), (2,), (2,)),
            (("dry",), (2,), (3,)),
            (("wet",), (1,), (1,)),
            (("wet",), (1,), (2,)),
            (("wet",), (1,), (3,)),
            (("wet",), (2,), (1,)),
            (("wet",), (2,), (2,)),
            (("wet",), (2,), (3,)),
        }
        assert res == sol

    def test_partially_one(self):
        """Decompress only one select parameter."""
        setups = self.setup.decompress_partially(["species_id"])
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        assert len(setups) == 2
        res = {(s.deposition_type, s.species_id, s.time) for s in setups}
        sol = {(("dry", "wet"), (1,), (1, 2, 3)), (("dry", "wet"), (2,), (1, 2, 3))}
        assert res == sol

    def test_select_two(self):
        """Decompress two select parameters."""
        setups = self.setup.decompress_partially(["time", "deposition_type"])
        assert len(setups) == 6
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        res = {(s.deposition_type, s.species_id, s.time) for s in setups}
        sol = {
            (("dry",), (1, 2), (1,)),
            (("dry",), (1, 2), (2,)),
            (("dry",), (1, 2), (3,)),
            (("wet",), (1, 2), (1,)),
            (("wet",), (1, 2), (2,)),
            (("wet",), (1, 2), (3,)),
        }
        assert res == sol


class Test_InputSetupCollection:
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

    def create_setups(self):
        return [InputSetup.create(dct) for dct in self.create_partial_dicts()]

    def test_create_empty(self):
        setups = InputSetupCollection([])
        assert len(setups) == 0

    def test_from_setups(self):
        partial_dicts = self.create_partial_dicts()
        setups = InputSetupCollection.create(partial_dicts)
        assert len(setups) == len(partial_dicts)


class Test_Compress:
    dcts: List[Dict[str, Any]] = [
        {**DEFAULT_KWARGS, **dct}
        for dct in [
            {"infile": "foo.nc", "input_variable": "concentration", "level": 0},
            {"infile": "foo.nc", "input_variable": "concentration", "level": 1},
            {"infile": "foo.nc", "input_variable": "concentration", "level": (1, 2)},
        ]
    ]
    setups_lst = [InputSetup.create(dct) for dct in dcts]

    def test_one(self):
        res = InputSetupCollection(self.setups_lst[:1]).compress().dict()
        sol = self.setups_lst[0]
        assert res == sol

    def test_two(self):
        res = InputSetupCollection(self.setups_lst[:2]).compress().dict()
        sol = InputSetup.create({**self.dcts[0], "level": (0, 1)})
        assert res == sol

    def test_three(self):
        res = InputSetupCollection(self.setups_lst[:3]).compress().dict()
        sol = InputSetup.create({**self.dcts[0], "level": (0, 1, 2)})
        assert res == sol
