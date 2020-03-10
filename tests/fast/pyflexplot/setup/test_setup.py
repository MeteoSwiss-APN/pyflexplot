#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup``.
"""
# First-party
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupCollection

DEFAULT_KWARGS = {
    "infiles": ("foo.nc",),
    "outfile": "bar.png",
}


def CREATE_DEFAULT_SETUP():
    return Setup(
        **{
            **DEFAULT_KWARGS,
            "age_class_idx": 0,
            "combine_species": False,
            "deposition_type": "none",
            "domain": "auto",
            "ens_member_ids": None,
            "integrate": False,
            "lang": "en",
            "level_idx": None,
            "nout_rel_idx": 0,
            "plot_type": "auto",
            "release_point_idx": 0,
            "reverse_legend": False,
            "scale_fact": None,
            "simulation_type": "deterministic",
            "species_id": 1,
            "time_idcs": (0,),
            "variable": "concentration",
        }
    )


def test_default_setup_dict():
    """Check the default setupuration dict."""
    assert Setup(**DEFAULT_KWARGS) == CREATE_DEFAULT_SETUP().dict()


class Test_Decompress:
    @property
    def setup(self):
        return CREATE_DEFAULT_SETUP().derive(
            {
                "variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "species_id": [1, 2],
                "time_idcs": [1, 2, 3],
            },
        )

    def test_all(self):
        """Decompress all params."""
        setups = self.setup.decompress()
        assert len(setups) == 12
        res = {(s.deposition_type, s.species_id, s.time_idcs) for s in setups}
        sol = {
            ("dry", 1, (1,)),
            ("dry", 1, (2,)),
            ("dry", 1, (3,)),
            ("dry", 2, (1,)),
            ("dry", 2, (2,)),
            ("dry", 2, (3,)),
            ("wet", 1, (1,)),
            ("wet", 1, (2,)),
            ("wet", 1, (3,)),
            ("wet", 2, (1,)),
            ("wet", 2, (2,)),
            ("wet", 2, (3,)),
        }
        assert res == sol

    def test_select_one(self):
        """Decompress only one select parameter."""
        setups = self.setup.decompress(["species_id"])
        assert len(setups) == 2
        res = {(s.deposition_type, s.species_id, s.time_idcs) for s in setups}
        sol = {("tot", 1, (1, 2, 3)), ("tot", 2, (1, 2, 3))}
        assert res == sol

    def test_select_two(self):
        """Decompress two select parameters."""
        setups = self.setup.decompress(["time_idcs", "deposition_type"])
        assert len(setups) == 6
        res = {(s.deposition_type, s.species_id, s.time_idcs) for s in setups}
        sol = {
            ("dry", (1, 2), (1,)),
            ("dry", (1, 2), (2,)),
            ("dry", (1, 2), (3,)),
            ("wet", (1, 2), (1,)),
            ("wet", (1, 2), (2,)),
            ("wet", (1, 2), (3,)),
        }
        assert res == sol


class Test_SetupCollection:
    def create_partial_dicts(self):
        return [
            {**DEFAULT_KWARGS, **dct}
            for dct in [
                {"infiles": ("foo.nc",), "variable": "concentration", "domain": "ch"},
                {"infiles": ("bar.nc",), "variable": "deposition", "lang": "de"},
                {"age_class_idx": 1, "nout_rel_idx": 5, "release_point_idx": 3},
            ]
        ]

    def create_complete_dicts(self):
        return [
            {**CREATE_DEFAULT_SETUP().dict(), **dct}
            for dct in self.create_partial_dicts()
        ]

    def create_setups(self):
        return [Setup(**dct) for dct in self.create_partial_dicts()]

    def test_create_empty(self):
        setups = SetupCollection([])
        assert len(setups) == 0

    def test_from_setups(self):
        partial_dicts = self.create_partial_dicts()
        setups = SetupCollection(partial_dicts)
        assert len(setups) == len(partial_dicts)


class Test_Compress:
    dcts = [
        {**DEFAULT_KWARGS, **dct}
        for dct in [
            {"infiles": ("foo.nc",), "variable": "concentration", "level_idx": 0},
            {"infiles": ("foo.nc",), "variable": "concentration", "level_idx": 1},
            {"infiles": ("foo.nc",), "variable": "concentration", "level_idx": (1, 2)},
        ]
    ]
    setups = [Setup(**dct) for dct in dcts]

    def test_one(self):
        res = Setup.compress(self.setups[:1]).dict()
        sol = self.setups[0]
        assert res == sol

    def test_two(self):
        res = Setup.compress(self.setups[:2]).dict()
        sol = Setup(**{**self.dcts[0], "level_idx": (0, 1)})
        assert res == sol

    def test_three(self):
        res = Setup.compress(self.setups[:3]).dict()
        sol = Setup(**{**self.dcts[0], "level_idx": (0, 1, 2)})
        assert res == sol
