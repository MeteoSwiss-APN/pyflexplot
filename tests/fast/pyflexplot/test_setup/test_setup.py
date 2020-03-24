#!/usr/bin/env python
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

DEFAULT_KWARGS: Dict[str, Any] = {
    "infile": ("foo.nc",),
    "outfile": "bar.png",
}


DEFAULT_SETUP = InputSetup(
    **{
        **DEFAULT_KWARGS,
        "nageclass": None,
        "combine_species": False,
        "deposition_type": "none",
        "domain": "auto",
        "ens_member_id": None,
        "integrate": False,
        "lang": "en",
        "level": None,
        "noutrel": None,
        "plot_type": "auto",
        "numpoint": None,
        "simulation_type": "deterministic",
        "species_id": None,
        "time": None,
        "variable": "concentration",
    }
)


def test_default_setup_dict():
    """Check the default setupuration dict."""
    setup1 = InputSetup(**DEFAULT_KWARGS)
    setup2 = DEFAULT_SETUP.dict()
    assert setup1 == setup2


class Test_Decompress:
    @property
    def setup(self):
        return DEFAULT_SETUP.derive(
            {
                "variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "species_id": [1, 2],
                "time": [1, 2, 3],
                "nageclass": (0,),
                "noutrel": (0,),
                "numpoint": (0,),
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
            ("dry", (1,), (1,)),
            ("dry", (1,), (2,)),
            ("dry", (1,), (3,)),
            ("dry", (2,), (1,)),
            ("dry", (2,), (2,)),
            ("dry", (2,), (3,)),
            ("wet", (1,), (1,)),
            ("wet", (1,), (2,)),
            ("wet", (1,), (3,)),
            ("wet", (2,), (1,)),
            ("wet", (2,), (2,)),
            ("wet", (2,), (3,)),
        }
        assert res == sol

    def test_partially_one(self):
        """Decompress only one select parameter."""
        setups = self.setup.decompress_partially(["species_id"])
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        assert len(setups) == 2
        res = {(s.deposition_type, s.species_id, s.time) for s in setups}
        sol = {("tot", (1,), (1, 2, 3)), ("tot", (2,), (1, 2, 3))}
        assert res == sol

    def test_select_two(self):
        """Decompress two select parameters."""
        setups = self.setup.decompress_partially(["time", "deposition_type"])
        assert len(setups) == 6
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        res = {(s.deposition_type, s.species_id, s.time) for s in setups}
        sol = {
            ("dry", (1, 2), (1,)),
            ("dry", (1, 2), (2,)),
            ("dry", (1, 2), (3,)),
            ("wet", (1, 2), (1,)),
            ("wet", (1, 2), (2,)),
            ("wet", (1, 2), (3,)),
        }
        assert res == sol


class Test_InputSetupCollection:
    def create_partial_dicts(self):
        return [
            {**DEFAULT_KWARGS, **dct}
            for dct in [
                {"infile": "foo.nc", "variable": "concentration", "domain": "ch"},
                {"infile": "bar.nc", "variable": "deposition", "lang": "de"},
                {"nageclass": 1, "noutrel": 5, "numpoint": 3},
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
            {"infile": "foo.nc", "variable": "concentration", "level": 0},
            {"infile": "foo.nc", "variable": "concentration", "level": 1},
            {"infile": "foo.nc", "variable": "concentration", "level": (1, 2)},
        ]
    ]
    setups_lst = [InputSetup.create(dct) for dct in dcts]

    def test_one(self):
        res = InputSetup.compress(InputSetupCollection(self.setups_lst[:1])).dict()
        sol = self.setups_lst[0]
        assert res == sol

    def test_two(self):
        res = InputSetup.compress(InputSetupCollection(self.setups_lst[:2])).dict()
        sol = InputSetup.create({**self.dcts[0], "level": (0, 1)})
        assert res == sol

    def test_three(self):
        res = InputSetup.compress(InputSetupCollection(self.setups_lst[:3])).dict()
        sol = InputSetup.create({**self.dcts[0], "level": (0, 1, 2)})
        assert res == sol


class Test_CreateWildcardToNone:
    """Wildcard values passed to ``InputSetup.create`` turn into None.

    The wildcards can be used in a set file to explicitly specify that all
    available values of a respective dimension (etc.) shall be read from a
    NetCDF file, as es the case if the setup value is None (the default).

    """

    def setup_create(self, params):
        return InputSetup.create({**DEFAULT_KWARGS, **params})

    def test_species_id(self):
        setup = self.setup_create({"species_id": "*"})
        assert setup.species_id is None

    def test_time(self):
        setup = self.setup_create({"time": "*"})
        assert setup.time is None

    def test_level(self):
        setup = self.setup_create({"level": "*"})
        assert setup.level is None

    def test_others(self):
        setup = self.setup_create({"nageclass": "*", "noutrel": "*", "numpoint": "*"})
        assert setup.nageclass is None
        assert setup.noutrel is None
        assert setup.numpoint is None


class Test_ReplaceNoneByAvailable:
    meta_data: Dict[str, Any] = {
        "dimensions": {
            "time": {"name": "time", "size": 11},
            "rlon": {"name": "rlon", "size": 40},
            "rlat": {"name": "rlat", "size": 30},
            "level": {"name": "level", "size": 3},
            "nageclass": {"name": "nageclass", "size": 1},
            "noutrel": {"name": "numpoint", "size": 1},
            "numpoint": {"name": "numpoint", "size": 2},
            "nchar": {"name": "nchar", "size": 45},
        },
        "analysis": {"species_ids": (1, 2)},
    }

    def setup_create(self, params):
        return InputSetup.create({**DEFAULT_KWARGS, **params})

    def test_time(self):
        setup = self.setup_create({"time": "*"})
        assert setup.time is None
        setup.replace_nones(self.meta_data)
        assert setup.time == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    def test_level(self):
        setup = self.setup_create({"level": "*"})
        assert setup.level is None
        setup.replace_nones(self.meta_data)
        assert setup.level == (0, 1, 2)

    def test_species_id(self):
        setup = self.setup_create({"species_id": "*"})
        assert setup.species_id is None
        setup.replace_nones(self.meta_data)
        assert setup.species_id == (1, 2)

    def test_others(self):
        setup = self.setup_create({"nageclass": "*", "noutrel": "*", "numpoint": "*"})
        assert setup.nageclass is None
        assert setup.noutrel is None
        assert setup.numpoint is None
        setup.replace_nones(self.meta_data)
        assert setup.nageclass == (0,)
        assert setup.noutrel == (0,)
        assert setup.numpoint == (0, 1)
