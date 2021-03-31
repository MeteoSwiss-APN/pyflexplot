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
from pyflexplot.setups.plot_setup import PlotSetup
from pyflexplot.setups.plot_setup import PlotSetupGroup
from pyflexplot.setups.setup_file import prepare_raw_params
from srutils.dict import merge_dicts
from srutils.testing import assert_is_sub_element

# Local
from .shared import DEFAULT_SETUP


def tuples(objs: Sequence[Any]) -> List[Tuple[Any]]:
    """Turn all elements in a sequence into one-element tuples."""
    return [(obj,) for obj in objs]


class Test_Copy:
    def test_preserve_outfiles(self):
        params = {
            "files": {
                "input": "foo.nc",
                "output": ("foo.png", "bar.pdf"),
            },
            "model": {"name": "IFS-HRES"},
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
            "files": {
                "input": "foo.nc",
                "output": "foo.png",
            },
            "plot_variable": "concentration",
            "species_id": [1, 2],
            "combine_species": False,
        }
        params_lst = prepare_raw_params(raw_params)
        assert len(params_lst) == 1
        setups = PlotSetupGroup.create(params_lst)
        res = setups.dicts()
        sol = [
            {
                "files": {
                    "input": "foo.nc",
                    "output": "foo.png",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "combine_species": False,
                        "dimensions": {
                            "species_id": species_id,
                        },
                    }
                ],
            }
            for species_id in [1, 2]
        ]
        assert_is_sub_element(
            name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
        )


class Test_Create:
    class Test_Simple:
        """Test simple setup dicts which result in one setup object each."""

        panels_lst: List[List[Dict[str, Any]]] = [
            [{"plot_variable": "concentration", "domain": "ch"}],
            [{"plot_variable": "tot_deposition", "lang": "de"}],
            [{"dimensions": {"nageclass": 1, "noutrel": 5, "numpoint": 3}}],
        ]
        dicts: List[Dict[str, Any]] = [
            merge_dicts(
                {
                    "files": {
                        "input": "foo.nc",
                        "output": "bar.png",
                    },
                    "model": {"name": "COSMO-baz"},
                },
                {"panels": panels},
                overwrite_seqs=True,
            )
            for panels in panels_lst
        ]

        @property
        def setup_lst(self):
            return [PlotSetup.create(dct) for dct in self.dicts]

        @property
        def setup_group(self):
            return PlotSetupGroup(self.setup_lst)

        def test_empty_fail(self):
            with pytest.raises(ValueError):
                PlotSetupGroup([])

        def test_from_objs(self):
            group = PlotSetupGroup.create(self.setup_lst)
            res = group.dicts()
            sol = self.dicts
            assert_is_sub_element(
                name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
            )

        def test_from_dicts(self):
            group = PlotSetupGroup.create(self.dicts)
            res = group.dicts()
            sol = self.dicts
            assert_is_sub_element(
                name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
            )

    class Test_Complex:
        """Test more complex setup dicts resulting in multiple setups each."""

        base_params = {
            "files": {
                "input": "foo.nc",
                "output": "bar.png",
            },
            "model": {"name": "COSMO-baz"},
        }

        def test_combine_levels_true(self):
            params = {
                **self.base_params,
                "panels": {
                    "plot_variable": "affected_area",
                    "combine_levels": True,
                    "dimensions": {
                        "level": (0, 1, 2),
                    },
                },
            }
            group = PlotSetupGroup.create(params)
            res = group.dicts()
            sol = [{**params, "panels": [params["panels"]]}]
            assert_is_sub_element(
                name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
            )

        def test_combine_levels_false(self):
            params = {
                **self.base_params,
                "panels": {
                    "plot_variable": "affected_area",
                    "combine_levels": False,
                    "dimensions": {
                        "level": (0, 1, 2),
                    },
                },
            }
            group = PlotSetupGroup.create(params)
            res = group.dicts()
            sol = [
                merge_dicts(
                    params,
                    {
                        "panels": [
                            merge_dicts(
                                params["panels"],
                                {
                                    "plot_variable": "affected_area",
                                    "combine_levels": False,
                                    "dimensions": {
                                        "level": value,
                                        "variable": (
                                            "concentration",
                                            "dry_deposition",
                                            "wet_deposition",
                                        ),
                                    },
                                },
                                overwrite_seqs=True,
                            )
                        ]
                    },
                    overwrite_seqs=True,
                )
                for value in [0, 1, 2]
            ]
            assert_is_sub_element(
                name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
            )

        def test_combine_species_true(self):
            params = {
                **self.base_params,
                "panels": {
                    "plot_variable": "tot_deposition",
                    "combine_species": True,
                    "dimensions": {
                        "species_id": (0, 1, 2),
                    },
                },
            }
            group = PlotSetupGroup.create(params)
            res = group.dicts()
            sol = [{**params, "panels": [params["panels"]]}]
            assert_is_sub_element(
                name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
            )

        def test_combine_species_false(self):
            params = {
                **self.base_params,
                "panels": {
                    "plot_variable": "tot_deposition",
                    "combine_species": False,
                    "dimensions": {
                        "species_id": (0, 1, 2),
                    },
                },
            }
            group = PlotSetupGroup.create(params)
            res = group.dicts()
            sol = [
                merge_dicts(
                    params,
                    {
                        "panels": [
                            merge_dicts(
                                params["panels"],
                                {
                                    "plot_variable": "tot_deposition",
                                    "combine_species": False,
                                    "dimensions": {
                                        "species_id": value,
                                        "variable": (
                                            "dry_deposition",
                                            "wet_deposition",
                                        ),
                                    },
                                },
                                overwrite_seqs=True,
                            )
                        ]
                    },
                    overwrite_seqs=True,
                )
                for value in [0, 1, 2]
            ]
            assert_is_sub_element(
                name_sub="solution", obj_sub=sol, name_super="result", obj_super=res
            )


class Test_Compress:
    dcts: List[Dict[str, Any]] = [
        {
            "files": {
                "input": "foo.nc",
                "output": "bar.png",
            },
            "model": {
                "name": "COSMO-1",
            },
            "panels": [
                {
                    "plot_variable": "concentration",
                    "combine_levels": True,
                    "dimensions": {"level": 0},
                }
            ],
        },
        {
            "files": {
                "input": "foo.nc",
                "output": "bar.png",
            },
            "model": {
                "name": "COSMO-1",
            },
            "panels": [
                {
                    "plot_variable": "concentration",
                    "combine_levels": True,
                    "dimensions": {"level": 1},
                }
            ],
        },
        {
            "files": {
                "input": "foo.nc",
                "output": "bar.png",
            },
            "model": {
                "name": "COSMO-1",
            },
            "panels": [
                {
                    "plot_variable": "concentration",
                    "combine_levels": True,
                    "dimensions": {"level": (1, 2)},
                }
            ],
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
            {"panels": [{"dimensions": {"level": (0, 1)}}]}
        )
        assert res == sol

    def test_three(self):
        res = PlotSetupGroup(self.setups_lst[:3]).compress().dict()
        sol = PlotSetup.create(self.dcts[0]).derive(
            {"panels": [{"dimensions": {"level": (0, 1, 2)}}]}
        )
        assert res == sol


class Test_Group:
    """Group setups by the values of one or more params.

    'Regular' params apply to all variables, while 'special' params are
    variable-specific (e.g., deposition type).

    """

    outfile_lst = ["foo.png", "bar.pdf"]
    combine_species_lst = [True, False]
    time_lst = [0, -1, (0, 5, 10)]

    combine_levels_lst = [True, False]
    default_combine_levels = DEFAULT_SETUP.panels.collect_equal("combine_levels")

    n_outfile = len(outfile_lst)
    n_combine_species = len(combine_species_lst)
    n_time = len(time_lst)
    n_base = n_outfile * n_combine_species * n_time

    n_combine_levels = len(combine_levels_lst)
    n_concentration = n_combine_levels

    n_setups = n_base * n_concentration

    def get_setup_dcts(self):
        base_dcts = [
            {
                "files": {
                    "input": "foo.nc",
                    "output": outfile,
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "combine_species": combine_species,
                        "dimensions": {
                            "species_id": [1, 2] if combine_species else 1,
                            "time": time,
                        },
                    }
                ],
            }
            for outfile in self.outfile_lst
            for combine_species in self.combine_species_lst
            for time in self.time_lst
        ]
        assert len(base_dcts) == self.n_base
        concentration_dcts = [
            {
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "combine_levels": combine_levels,
                        "dimensions": {
                            "level": [0, 1, 2] if combine_levels else 0,
                        },
                    }
                ],
            }
            for combine_levels in self.combine_levels_lst
        ]
        assert len(concentration_dcts) == self.n_concentration
        return [
            merge_dicts(base_dct, dct)
            for base_dct in base_dcts
            for dct in concentration_dcts
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

        grouped = setups.group("files.output")
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

    def test_one_seq_regular(self):
        """One regular param, passed as a sequence."""
        setups = self.get_setups()

        grouped = setups.group(["files.output"])
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

    def test_two_regular(self):
        """Two regular params."""
        setups = self.get_setups()

        grouped = setups.group(["files.output", "combine_species"])
        assert len(grouped) == self.n_outfile * self.n_combine_species
        assert set(grouped) == set(product(self.outfile_lst, self.combine_species_lst))

        grouped = setups.group(["dimensions.time", "combine_species"])
        assert len(grouped) == self.n_time * self.n_combine_species
        assert set(grouped) == set(product(self.time_lst, self.combine_species_lst))

    def test_two_special(self):
        """Two params, among them special params."""
        setups = self.get_setups()

        grouped = setups.group(["files.output", "combine_levels"])
        assert len(grouped) == self.n_outfile * self.n_combine_species
        assert set(grouped) == set(product(self.outfile_lst, self.combine_levels_lst))

    def test_two_exclusive(self):
        """Two mutually special exclusive params."""
        setups = self.get_setups()
        grouped = setups.group(["files.output", "combine_levels"])
        assert len(grouped) == self.n_outfile * self.n_combine_levels
        sol = set(
            [
                (outfile, combine_levels)
                for outfile in self.outfile_lst
                for combine_levels in self.combine_levels_lst
            ]
        )
        assert set(grouped) == sol


class Test_Collect:
    base = {
        "files": {
            "input": "foo.nc",
            "output": "foo.png",
        },
        "model": {"name": "foo"},
    }
    md = lambda *dicts: merge_dicts(*dicts, overwrite_seqs=True)  # noqa
    params_lst = [
        md(
            base,
            {
                "layout": {"scale_fact": 1},
                "panels": [{"dimensions": {"species_id": 1}}],
            },
        ),
        md(
            base,
            {
                "layout": {"scale_fact": 2},
                "panels": [{"dimensions": {"species_id": 1}}],
            },
        ),
        md(
            base,
            {
                "layout": {"scale_fact": 1},
                "panels": [{"dimensions": {"species_id": 2}}],
            },
        ),
        md(
            base,
            {
                "layout": {"scale_fact": 3},
                "panels": [{"dimensions": {"species_id": 2}}],
            },
        ),
        md(
            base,
            {
                "layout": {"scale_fact": 1},
                "panels": [{"dimensions": {"species_id": 3}}],
            },
        ),
    ]

    def test_scale_fact(self):
        group = PlotSetupGroup.create(self.params_lst)
        vals = group.collect("layout.scale_fact")
        assert vals == [1, 2, 1, 3, 1]

    def test_scale_fact_flat(self):
        group = PlotSetupGroup.create(self.params_lst)
        vals = group.collect("layout.scale_fact", flatten=True)
        assert vals == [1, 2, 1, 3, 1]

    def test_scale_fact_flat_unique(self):
        group = PlotSetupGroup.create(self.params_lst)
        vals = group.collect("layout.scale_fact", flatten=True, unique=True)
        assert vals == [1, 2, 3]

    def test_species_id(self):
        group = PlotSetupGroup.create(self.params_lst)
        vals = group.collect("dimensions.species_id")
        assert vals == [[1], [1], [2], [2], [3]]

    def test_species_id_flat(self):
        group = PlotSetupGroup.create(self.params_lst)
        vals = group.collect("dimensions.species_id", flatten=True)
        assert vals == [1, 1, 2, 2, 3]

    def test_species_id_flat_unique(self):
        group = PlotSetupGroup.create(self.params_lst)
        vals = group.collect("dimensions.species_id", flatten=True, unique=True)
        assert vals == [1, 2, 3]
