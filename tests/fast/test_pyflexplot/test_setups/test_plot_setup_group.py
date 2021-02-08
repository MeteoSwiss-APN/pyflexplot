"""Tests for class ``pyflexplot.setup.SetupGroup``."""
# Standard library
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

# Third-party
import pytest

# First-party
from pyflexplot.setups.setup import PlotSetup
from pyflexplot.setups.setup import PlotSetupGroup
from pyflexplot.setups.setup_file import SetupFile
from srutils.dict import merge_dicts
from srutils.testing import check_is_sub_element

# Local
from .shared import DEFAULT_SETUP
from .shared import OPTIONAL_RAW_DEFAULT_PARAMS


def tuples(objs: Sequence[Any]) -> List[Tuple[Any]]:
    """Turn all elements in a sequence into one-element tuples."""
    return [(obj,) for obj in objs]


class Test_Copy:
    def test_preserve_outfiles(self):
        params = {
            "infile": "foo.nc",
            "model": {"name": "IFS-HRES"},
            "outfile": ("foo.png", "bar.pdf"),
        }
        setup = PlotSetup.create(params)
        setups = PlotSetupGroup([setup])
        copied_setups = setups.copy()
        assert len(copied_setups) == len(setups)
        assert copied_setups == setups


class Test_FromRawParams:
    def test_empty(self):
        with pytest.raises(ValueError):
            PlotSetupGroup.create([])

    def test_one_variable(self):
        raw_params = {
            "infile": "foo.nc",
            "outfile": "foo.png",
            "input_variable": "concentration",
            "species_id": [1, 2],
            "combine_species": False,
        }
        setups = PlotSetupGroup.create(SetupFile.prepare_raw_params(raw_params))
        res = setups.dicts()
        sol = [
            {
                "infile": "foo.nc",
                "outfile": "foo.png",
                "panels": {
                    "input_variable": "concentration",
                    "combine_species": False,
                    "dimensions": {
                        "species_id": (1, 2),
                    },
                },
            },
        ]
        check_is_sub_element(sol, res, "solution", "result")


class Test_SetupGroup_Create:
    def create_partial_dicts(self):
        base = {
            "infile": "foo.nc",
            "outfile": "bar.png",
            "model": {"name": "COSMO-baz"},
        }
        return [
            {
                **base,
                "panels": {"input_variable": "concentration", "domain": "ch"},
            },
            {
                **base,
                "panels": {"input_variable": "deposition", "lang": "de"},
            },
            {
                **base,
                "panels": {"dimensions": {"nageclass": 1, "noutrel": 5, "numpoint": 3}},
            },
        ]

    def create_complete_dicts(self):
        return [
            merge_dicts(OPTIONAL_RAW_DEFAULT_PARAMS, dct)
            for dct in self.create_partial_dicts()
        ]

    def create_setup_lst(self):
        return [PlotSetup.create(dct) for dct in self.create_partial_dicts()]

    def create_setup_group(self):
        return PlotSetupGroup(self.create_setup_lst())

    def test_create_empty_setup_group(self):
        with pytest.raises(ValueError):
            PlotSetupGroup([])

    def test_from_setups(self):
        partial_dicts = self.create_partial_dicts()
        setups = PlotSetupGroup.create(partial_dicts)
        assert len(setups) == len(partial_dicts)


class Test_SetupGroup_Compress:
    dcts: List[Dict[str, Any]] = [
        {
            "infile": "foo.nc",
            "outfile": "bar.png",
            "model": {
                "name": "COSMO-1",
            },
            "panels": {
                "input_variable": "concentration",
                "dimensions": {"level": 0},
            },
        },
        {
            "infile": "foo.nc",
            "outfile": "bar.png",
            "model": {
                "name": "COSMO-1",
            },
            "panels": {
                "input_variable": "concentration",
                "dimensions": {"level": 1},
            },
        },
        {
            "infile": "foo.nc",
            "outfile": "bar.png",
            "model": {
                "name": "COSMO-1",
            },
            "panels": {
                "input_variable": "concentration",
                "dimensions": {"level": (1, 2)},
            },
        },
    ]
    setups_lst = [PlotSetup.create(dct) for dct in dcts]

    def test_one(self):
        res = PlotSetupGroup(self.setups_lst[:1]).compress().dict()
        sol = self.setups_lst[0]
        assert res == sol

    def test_two(self):
        res = PlotSetupGroup(self.setups_lst[:2]).compress().dict()
        sol = PlotSetup.create(self.dcts[0]).derive(
            {"panels": {"dimensions": {"level": (0, 1)}}}
        )
        assert res == sol

    def test_three(self):
        res = PlotSetupGroup(self.setups_lst[:3]).compress().dict()
        sol = PlotSetup.create(self.dcts[0]).derive(
            {"panels": {"dimensions": {"level": (0, 1, 2)}}}
        )
        assert res == sol


class Test_SetupGroup_Group:
    """Group setups by the values of one or more params.

    'Regular' params apply to all variables, while 'special' params are
    variable-specific (e.g., deposition type).

    """

    outfile_lst = ["foo.png", "bar.pdf"]
    combine_species_lst = [True, False]
    time_lst = [0, -1, (0, 5, 10)]

    combine_levels_lst = [True, False]
    default_combine_levels = DEFAULT_SETUP.panels.collect_equal("combine_levels")

    deposition_type_lst = ["dry", "wet", ("dry", "wet")]
    default_deposition_type = DEFAULT_SETUP.panels.collect_equal(
        "dimensions"
    ).deposition_type

    n_outfile = len(outfile_lst)
    n_combine_species = len(combine_species_lst)
    n_time = len(time_lst)
    n_base = n_outfile * n_combine_species * n_time

    n_combine_levels = len(combine_levels_lst)
    n_concentration = n_combine_levels

    n_deposition_type = len(deposition_type_lst)
    n_deposition = n_deposition_type

    n_setups = n_base * (n_concentration + n_deposition)

    def get_setup_dcts(self):
        base_dcts = [
            {
                "infile": "foo.nc",
                "outfile": outfile,
                "model": {
                    "name": "COSMO-1",
                },
                "panels": {
                    "combine_species": combine_species,
                    "dimensions": {"species_id": [1, 2], "time": time},
                },
            }
            for outfile in self.outfile_lst
            for combine_species in self.combine_species_lst
            for time in self.time_lst
        ]
        assert len(base_dcts) == self.n_base
        concentration_dcts = [
            {
                "panels": {
                    "input_variable": "concentration",
                    "combine_levels": combine_levels,
                    "dimensions": {"level": [0, 1, 2]},
                },
            }
            for combine_levels in self.combine_levels_lst
        ]
        assert len(concentration_dcts) == self.n_concentration
        deposition_dcts = [
            {
                "panels": {
                    "input_variable": "deposition",
                    "combine_deposition_types": True,
                    "dimensions": {"deposition_type": deposition_type},
                },
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
        return PlotSetupGroup.create(self.get_setup_dcts())

    def test_reference_setups(self):
        """Sanity check of the test reference setups."""
        setups = self.get_setups()
        assert isinstance(setups, PlotSetupGroup)
        assert len(setups) == self.n_setups

    def test_one_regular(self):
        """One regular param, passed as a string."""
        setups = self.get_setups()

        grouped = setups.group("outfile")
        assert len(grouped) == self.n_outfile
        assert set(grouped) == set(self.outfile_lst)

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

        grouped = setups.group(["outfile"])
        assert len(grouped) == self.n_outfile
        assert set(grouped) == set(tuples(self.outfile_lst))

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

        grouped = setups.group(["outfile", "combine_species"])
        assert len(grouped) == self.n_outfile * self.n_combine_species
        assert set(grouped) == set(product(self.outfile_lst, self.combine_species_lst))

        grouped = setups.group(["dimensions.time", "combine_species"])
        assert len(grouped) == self.n_time * self.n_combine_species
        assert set(grouped) == set(product(self.time_lst, self.combine_species_lst))

    def test_two_special(self):
        """Two params, among them special params."""
        setups = self.get_setups()

        grouped = setups.group(["outfile", "combine_levels"])
        assert len(grouped) == self.n_outfile * self.n_combine_species
        assert set(grouped) == set(product(self.outfile_lst, self.combine_levels_lst))

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

        grouped = setups.group(["outfile", "combine_species", "dimensions.time"])
        assert len(grouped) == self.n_outfile * self.n_combine_species * self.n_time
        sol = set(product(self.outfile_lst, self.combine_species_lst, self.time_lst))
        assert set(grouped) == sol

    def test_three_special(self):
        """Three params, among them special params."""
        setups = self.get_setups()

        grouped = setups.group(["outfile", "dimensions.time", "combine_levels"])
        assert len(grouped) == self.n_outfile * self.n_time * self.n_combine_levels
        sol = set(product(self.outfile_lst, self.time_lst, self.combine_species_lst))
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
            ["outfile", "dimensions.deposition_type", "combine_levels"]
        )
        sol = self.n_outfile * (self.n_combine_levels + self.n_deposition_type)
        assert len(grouped) == sol
        exclusive_combos = list(
            product([self.default_deposition_type], self.combine_levels_lst)
        ) + list(product(self.deposition_type_lst, [self.default_combine_levels]))
        sol = set(
            [
                (outfile, combine_levels, deposition_type)
                for outfile in self.outfile_lst
                for (combine_levels, deposition_type) in exclusive_combos
            ]
        )
        assert set(grouped) == sol
