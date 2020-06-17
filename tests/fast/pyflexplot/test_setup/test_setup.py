# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.setup``.
"""
# Standard library
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

# First-party
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection
from srutils.dict import merge_dicts
from srutils.testing import check_summary_dict_is_subdict

# Local
from .shared import DEFAULT_KWARGS
from .shared import DEFAULT_SETUP


def tuples(objs: Sequence[Any]) -> List[Tuple[Any]]:
    """Turn all elements in a sequence into one-element tuples."""
    return [(obj,) for obj in objs]


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
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
                    "nageclass": (0,),
                    "noutrel": (0,),
                    "numpoint": (0,),
                    "species_id": [1, 2],
                    "time": [1, 2, 3],
                },
            },
        )

    def test_full(self):
        """Decompress all params."""
        setups = self.setup.decompress()
        assert len(setups) == 12
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
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
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
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

    def test_partially_one(self):
        """Decompress only one select parameter."""
        setups = self.setup.decompress_partially(["dimensions.species_id"])
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        assert len(setups) == 2
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
            for s in setups
        }
        sol = {(("dry", "wet"), 1, (1, 2, 3)), (("dry", "wet"), 2, (1, 2, 3))}
        assert res == sol

    def test_select_two(self):
        """Decompress two select parameters."""
        setups = self.setup.decompress_partially(
            ["dimensions.time", "dimensions.deposition_type"]
        )
        assert len(setups) == 6
        assert isinstance(setups, InputSetupCollection)
        assert all(isinstance(setup, InputSetup) for setup in setups)
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
            for s in setups
        }
        sol = {
            ("dry", (1, 2), 1),
            ("dry", (1, 2), 2),
            ("dry", (1, 2), 3),
            ("wet", (1, 2), 1),
            ("wet", (1, 2), 2),
            ("wet", (1, 2), 3),
        }
        assert res == sol


class Test_InputSetupCollection_Create:
    def create_partial_dicts(self):
        return [
            merge_dicts(DEFAULT_KWARGS, dct)
            for dct in [
                {"infile": "foo.nc", "input_variable": "concentration", "domain": "ch"},
                {"infile": "bar.nc", "input_variable": "deposition", "lang": "de"},
                {"dimensions": {"nageclass": 1, "noutrel": 5, "numpoint": 3}},
            ]
        ]

    def create_complete_dicts(self):
        return [
            merge_dicts(DEFAULT_SETUP.dict(), dct)
            for dct in self.create_partial_dicts()
        ]

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
        merge_dicts(DEFAULT_KWARGS, dct)
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


class Test_InputSetupCollection_Group:
    """Group setups by the values of one or more params.

    'Regular' params apply to all variables, while 'special' params are
    variable-specific (e.g., deposition type).

    """

    infile_lst = ["foo.nc", "bar.nc"]
    combine_species_lst = [True, False]
    time_lst = [0, -1, (0, 5, 10)]

    combine_levels_lst = [True, False]
    default_combine_levels = DEFAULT_SETUP.core.combine_levels

    deposition_type_lst = ["dry", "wet", ("dry", "wet")]
    default_deposition_type = DEFAULT_SETUP.core.dimensions.deposition_type

    n_infile = len(infile_lst)
    n_combine_species = len(combine_species_lst)
    n_time = len(time_lst)
    n_base = n_infile * n_combine_species * n_time

    n_combine_levels = len(combine_levels_lst)
    n_concentration = n_combine_levels

    n_deposition_type = len(deposition_type_lst)
    n_deposition = n_deposition_type

    n_setups = n_base * (n_concentration + n_deposition)

    def get_setup_dcts(self):
        base_dcts = [
            {
                "infile": infile,
                "outfile": infile.replace(".nc", ".png"),
                "combine_species": combine_species,
                "dimensions": {"species_id": [1, 2], "time": time},
            }
            for infile in self.infile_lst
            for combine_species in self.combine_species_lst
            for time in self.time_lst
        ]
        assert len(base_dcts) == self.n_base
        concentration_dcts = [
            {
                "input_variable": "concentration",
                "combine_levels": combine_levels,
                "dimensions": {"level": [0, 1, 2]},
            }
            for combine_levels in self.combine_levels_lst
        ]
        assert len(concentration_dcts) == self.n_concentration
        deposition_dcts = [
            {
                "input_variable": "deposition",
                "combine_deposition_types": True,
                "dimensions": {"deposition_type": deposition_type},
            }
            for deposition_type in self.deposition_type_lst
        ]
        assert len(deposition_dcts) == self.n_deposition
        return [
            merge_dicts(base_dct, dct)
            for base_dct in base_dcts
            for dct in concentration_dcts + deposition_dcts
        ]

    def get_setups(self):
        return InputSetupCollection.create(self.get_setup_dcts())

    def test_reference_setups(self):
        """Sanity check of the test reference setups."""
        setups = self.get_setups()
        assert isinstance(setups, InputSetupCollection)
        assert len(setups) == self.n_setups

    def test_one_regular(self):
        """One regular param, passed as a string."""
        setups = self.get_setups()

        grouped = setups.group("infile")
        assert len(grouped) == self.n_infile
        assert set(grouped) == set(self.infile_lst)

        grouped = setups.group("combine_species")
        assert len(grouped) == self.n_combine_species
        assert set(grouped) == set(self.combine_species_lst)

        grouped = setups.group("dimensions.time")
        assert len(grouped) == self.n_time
        assert set(grouped) == set(self.time_lst)

    def test_one_special(self):
        """One special param, passed as a string."""
        setups = self.get_setups()

        grouped = setups.group("combine_levels")
        assert len(grouped) == self.n_combine_levels
        assert set(grouped) == set(self.combine_levels_lst)

        grouped = setups.group("dimensions.deposition_type")
        assert len(grouped) == self.n_deposition_type + 1
        sol = set([self.default_deposition_type] + self.deposition_type_lst)
        assert set(grouped) == sol

    def test_one_seq_regular(self):
        """One regular param, passed as a sequence."""
        setups = self.get_setups()

        grouped = setups.group(["infile"])
        assert len(grouped) == self.n_infile
        assert set(grouped) == set(tuples(self.infile_lst))

        grouped = setups.group(["combine_species"])
        assert len(grouped) == self.n_combine_species
        assert set(grouped) == set(tuples(self.combine_species_lst))

        grouped = setups.group(["dimensions.time"])
        assert len(grouped) == self.n_time
        assert set(grouped) == set(tuples(self.time_lst))

    def test_one_seq_special(self):
        """One special param, passed as a sequence."""
        setups = self.get_setups()

        grouped = setups.group(["combine_levels"])
        assert len(grouped) == self.n_combine_levels
        assert set(grouped) == set(tuples(self.combine_levels_lst))

        grouped = setups.group(["dimensions.deposition_type"])
        assert len(grouped) == self.n_deposition_type + 1
        sol = set(tuples([self.default_deposition_type] + self.deposition_type_lst))
        assert set(grouped) == sol

    def test_two_regular(self):
        """Two regular params."""
        setups = self.get_setups()

        grouped = setups.group(["infile", "combine_species"])
        assert len(grouped) == self.n_infile * self.n_combine_species
        assert set(grouped) == set(product(self.infile_lst, self.combine_species_lst))

        grouped = setups.group(["dimensions.time", "combine_species"])
        assert len(grouped) == self.n_time * self.n_combine_species
        assert set(grouped) == set(product(self.time_lst, self.combine_species_lst))

    def test_two_special(self):
        """Two params, among them special params."""
        setups = self.get_setups()

        grouped = setups.group(["infile", "combine_levels"])
        assert len(grouped) == self.n_infile * self.n_combine_species
        assert set(grouped) == set(product(self.infile_lst, self.combine_levels_lst))

        grouped = setups.group(["dimensions.time", "dimensions.deposition_type"])
        assert len(grouped) == self.n_time * (self.n_deposition_type + 1)
        sol = set(
            product(
                self.time_lst, [self.default_deposition_type] + self.deposition_type_lst
            )
        )
        assert set(grouped) == sol

    def test_two_exclusive(self):
        """Two mutually special exclusive params."""
        setups = self.get_setups()
        grouped = setups.group(["combine_levels", "dimensions.deposition_type"])
        assert len(grouped) == self.n_combine_levels + self.n_deposition_type
        sol = set(
            list(product(self.combine_levels_lst, [self.default_deposition_type]))
            + list(product([self.default_combine_levels], self.deposition_type_lst))
        )
        assert set(grouped) == sol

    def test_three_regular(self):
        """Three regular params."""
        setups = self.get_setups()

        grouped = setups.group(["infile", "combine_species", "dimensions.time"])
        assert len(grouped) == self.n_infile * self.n_combine_species * self.n_time
        sol = set(product(self.infile_lst, self.combine_species_lst, self.time_lst))
        assert set(grouped) == sol

    def test_three_special(self):
        """Three params, among them special params."""
        setups = self.get_setups()

        grouped = setups.group(["infile", "dimensions.time", "combine_levels"])
        assert len(grouped) == self.n_infile * self.n_time * self.n_combine_levels
        sol = set(product(self.infile_lst, self.time_lst, self.combine_species_lst))
        assert set(grouped) == sol

        grouped = setups.group(
            ["dimensions.time", "combine_species", "dimensions.deposition_type"]
        )
        sol = self.n_time * self.n_combine_species * (self.n_deposition_type + 1)
        assert len(grouped) == sol
        sol = set(
            product(
                self.time_lst,
                self.combine_species_lst,
                [self.default_deposition_type] + self.deposition_type_lst,
            )
        )
        assert set(grouped) == sol

    def test_three_exclusive(self):
        """Three params, among them mutually exclusive special params."""
        setups = self.get_setups()

        grouped = setups.group(
            ["infile", "dimensions.deposition_type", "combine_levels"]
        )
        sol = self.n_infile * (self.n_combine_levels + self.n_deposition_type)
        assert len(grouped) == sol
        exclusive_combos = list(
            product([self.default_deposition_type], self.combine_levels_lst)
        ) + list(product(self.deposition_type_lst, [self.default_combine_levels]))
        sol = set(
            [
                (infile, combine_levels, deposition_type)
                for infile in self.infile_lst
                for (combine_levels, deposition_type) in exclusive_combos
            ]
        )
        assert set(grouped) == sol
