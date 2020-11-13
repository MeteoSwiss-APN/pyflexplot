"""Test function ``pyflexplot.input.read_fields`` in dry-run mode."""
# Standard library
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence

# Third-party
import pytest

# First-party
from pyflexplot.data import Field
from pyflexplot.input import read_fields
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupCollection
from srutils.dict import merge_dicts
from srutils.testing import check_summary_dict_element_is_subelement

# Local  isort:skip
from .shared import datadir_reduced as datadir  # noqa:F401 isort:skip


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"
datafilename3 = "flexpart_ifs_20200317000000.nc"


@overload
def fields_to_setup_dcts(obj: Field, params=...) -> Dict[str, Any]:
    ...


@overload
def fields_to_setup_dcts(obj: Sequence[Field], params=...) -> Sequence[Dict[str, Any]]:
    ...


@overload
def fields_to_setup_dcts(obj: Sequence[Sequence], params=...) -> Sequence[Sequence]:
    ...


def fields_to_setup_dcts(obj, params: Optional[List[str]] = None):
    """Extract input setups from one or more ``Field`` objects.

    Multiple ``Field`` objects may be passed as arbitrarily nested sequences,
    in which case the nesting is retained in the result.

    """
    if isinstance(obj, Sequence):
        result = []
        for sub_obj in obj:
            result.append(fields_to_setup_dcts(sub_obj, params))
        return result
    else:
        assert isinstance(obj, Field)
        dct_all = obj.var_setups.compress().dict()
        if params is None:
            return dct_all
        else:
            dct_sel = {}
            for param in params:
                if not param.startswith("dimensions."):
                    dct_sel[param] = dct_all[param]
                else:
                    if "dimensions" not in dct_sel:
                        dct_sel["dimensions"] = {}
                    param = param.split(".", 1)[-1]
                    dct_sel["dimensions"][param] = dct_all["dimensions"][param]
            return dct_sel


def _test_setups_core(
    setups: SetupCollection, params: List[str], sol: List[List[Dict[str, Any]]]
):
    infile = setups.collect_equal("infile")
    field_lst_lst = read_fields(infile, setups, dry_run=True)
    res = fields_to_setup_dcts(field_lst_lst, params)
    check_summary_dict_element_is_subelement(obj_super=res, obj_sub=sol)


@dataclass
class ConfSingleSetup:
    setup_dct: Dict[str, Any]
    sol: List[List[Dict[str, Any]]]


@dataclass
class ConfMultipleSetups:
    setup_dct_lst: List[Dict[str, Any]]
    sol: List[List[Dict[str, Any]]]


def _test_single_setup_core(config, params):
    setup = Setup.create(config.setup_dct)
    setups = SetupCollection([setup])
    _test_setups_core(setups, params, config.sol)


def _test_multiple_setups_core(config, params):
    setup_lst = [Setup.create(setup_dct) for setup_dct in config.setup_dct_lst]
    setups = SetupCollection(setup_lst)
    _test_setups_core(setups, params, config.sol)


# test_single_setup_concentration
@pytest.mark.parametrize(
    "config",
    [
        ConfSingleSetup(  # [conf0]
            setup_dct={"dimensions": {"species_id": 1, "level": 0, "time": 0}},
            sol=[[{"dimensions": {"species_id": 1, "time": 0}}]],
        ),
        ConfSingleSetup(  # [conf1]
            setup_dct={"dimensions": {"species_id": 1, "level": 0, "time": (0, 3, 6)}},
            sol=[
                [{"dimensions": {"species_id": 1, "time": 0}}],
                [{"dimensions": {"species_id": 1, "time": 3}}],
                [{"dimensions": {"species_id": 1, "time": 6}}],
            ],
        ),
        ConfSingleSetup(  # [conf2]
            setup_dct={
                "dimensions": {"species_id": (1, 2), "level": 0, "time": 0},
                "combine_species": True,
            },
            sol=[[{"dimensions": {"species_id": (1, 2), "time": 0}}]],
        ),
        ConfSingleSetup(  # [conf3]
            setup_dct={
                "dimensions": {"species_id": (1, 2), "level": 0, "time": (0, 3, 6)},
                "combine_species": True,
            },
            sol=[
                [{"dimensions": {"species_id": (1, 2), "time": 0}}],
                [{"dimensions": {"species_id": (1, 2), "time": 3}}],
                [{"dimensions": {"species_id": (1, 2), "time": 6}}],
            ],
        ),
        ConfSingleSetup(  # [conf4]
            setup_dct={
                "dimensions": {"species_id": (1, 2), "level": 0, "time": 0},
                "combine_species": False,
            },
            sol=[
                [{"dimensions": {"species_id": 1, "time": 0}}],
                [{"dimensions": {"species_id": 2, "time": 0}}],
            ],
        ),
        ConfSingleSetup(  # [conf5]
            setup_dct={
                "dimensions": {"species_id": (1, 2), "level": 0, "time": (0, 3, 6)},
                "combine_species": False,
            },
            sol=[
                [{"dimensions": {"species_id": 1, "time": 0}}],
                [{"dimensions": {"species_id": 1, "time": 3}}],
                [{"dimensions": {"species_id": 1, "time": 6}}],
                [{"dimensions": {"species_id": 2, "time": 0}}],
                [{"dimensions": {"species_id": 2, "time": 3}}],
                [{"dimensions": {"species_id": 2, "time": 6}}],
            ],
        ),
        ConfSingleSetup(  # [conf6]
            setup_dct={
                "dimensions": {"species_id": 1, "level": (0, 1), "time": 0},
                "combine_levels": True,
            },
            sol=[[{"dimensions": {"species_id": 1, "level": (0, 1), "time": 0}}]],
        ),
        ConfSingleSetup(  # [conf7]
            setup_dct={
                "dimensions": {"species_id": 1, "level": (0, 1), "time": 0},
                "combine_levels": False,
            },
            sol=[
                [{"dimensions": {"species_id": 1, "level": 0, "time": 0}}],
                [{"dimensions": {"species_id": 1, "level": 1, "time": 0}}],
            ],
        ),
        ConfSingleSetup(  # [conf8]
            setup_dct={
                "dimensions": {
                    "species_id": (1, 2),
                    "level": (0, 1),
                    "time": (0, 3, 6),
                },
                "combine_species": False,
                "combine_levels": False,
            },
            sol=[
                [{"dimensions": {"level": 0, "species_id": 1, "time": 0}}],
                [{"dimensions": {"level": 0, "species_id": 1, "time": 3}}],
                [{"dimensions": {"level": 0, "species_id": 1, "time": 6}}],
                [{"dimensions": {"level": 0, "species_id": 2, "time": 0}}],
                [{"dimensions": {"level": 0, "species_id": 2, "time": 3}}],
                [{"dimensions": {"level": 0, "species_id": 2, "time": 6}}],
                [{"dimensions": {"level": 1, "species_id": 1, "time": 0}}],
                [{"dimensions": {"level": 1, "species_id": 1, "time": 3}}],
                [{"dimensions": {"level": 1, "species_id": 1, "time": 6}}],
                [{"dimensions": {"level": 1, "species_id": 2, "time": 0}}],
                [{"dimensions": {"level": 1, "species_id": 2, "time": 3}}],
                [{"dimensions": {"level": 1, "species_id": 2, "time": 6}}],
            ],
        ),
    ],
)
def test_single_setup_concentration(datadir: str, config: ConfSingleSetup):
    params = ["dimensions.species_id", "dimensions.level", "dimensions.time"]
    config.setup_dct.update(
        {
            "infile": f"{datadir}/{datafilename1}",
            "outfile": "foo.png",
            "model": "COSMO-1",
            "input_variable": "concentration",
        }
    )
    _test_single_setup_core(config, params)


# test_single_setup_deposition
@pytest.mark.parametrize(
    "config",
    [
        ConfSingleSetup(  # [conf0]
            setup_dct={
                "dimensions": {"species_id": 1, "time": 0, "deposition_type": "dry"},
            },
            sol=[
                [
                    {
                        "dimensions": {
                            "species_id": 1,
                            "time": 0,
                            "deposition_type": "dry",
                        },
                    }
                ]
            ],
        ),
        ConfSingleSetup(  # [conf1]
            setup_dct={
                "combine_deposition_types": False,
                "dimensions": {
                    "species_id": 1,
                    "time": 0,
                    "deposition_type": ("dry", "wet"),
                },
            },
            sol=[
                [
                    {
                        "dimensions": {
                            "species_id": 1,
                            "time": 0,
                            "deposition_type": "dry",
                        },
                    }
                ],
                [
                    {
                        "dimensions": {
                            "species_id": 1,
                            "time": 0,
                            "deposition_type": "wet",
                        },
                    }
                ],
            ],
        ),
        ConfSingleSetup(  # [conf2]
            setup_dct={
                "combine_deposition_types": True,
                "dimensions": {
                    "species_id": 1,
                    "time": 0,
                    "deposition_type": ("dry", "wet"),
                },
            },
            sol=[
                [
                    {
                        "dimensions": {
                            "species_id": 1,
                            "time": 0,
                            "deposition_type": ("dry", "wet"),
                        },
                    }
                ],
            ],
        ),
        ConfSingleSetup(  # [conf3]
            setup_dct={
                "combine_deposition_types": False,
                "combine_species": True,
                "dimensions": {
                    "species_id": (1, 2),
                    "time": (0, 3, 6),
                    "deposition_type": ("dry", "wet"),
                },
            },
            sol=[
                [
                    {
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": 0,
                            "deposition_type": "dry",
                        },
                    }
                ],
                [
                    {
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": 3,
                            "deposition_type": "dry",
                        },
                    }
                ],
                [
                    {
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": 6,
                            "deposition_type": "dry",
                        },
                    }
                ],
                [
                    {
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": 0,
                            "deposition_type": "wet",
                        },
                    }
                ],
                [
                    {
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": 3,
                            "deposition_type": "wet",
                        },
                    }
                ],
                [
                    {
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": 6,
                            "deposition_type": "wet",
                        },
                    }
                ],
            ],
        ),
    ],
)
def test_single_setup_deposition(datadir: str, config: ConfSingleSetup):
    params = ["dimensions.deposition_type", "dimensions.species_id", "dimensions.time"]
    config.setup_dct.update(
        {
            "infile": f"{datadir}/{datafilename1}",
            "outfile": "foo.png",
            "model": "COSMO-1",
            "input_variable": "deposition",
        }
    )
    _test_single_setup_core(config, params)


# test_multiple_setups
@pytest.mark.parametrize(
    "config",
    [
        ConfMultipleSetups(  # [conf0]
            setup_dct_lst=[
                {
                    "input_variable": "concentration",
                    "dimensions": {"species_id": 1, "level": 0, "time": 0},
                },
                {
                    "input_variable": "deposition",
                    "dimensions": {
                        "species_id": 1,
                        "time": 0,
                        "deposition_type": "dry",
                    },
                },
            ],
            sol=[
                [{"input_variable": "concentration"}],
                [{"input_variable": "deposition"}],
            ],
        ),
        ConfMultipleSetups(  # [conf1]
            setup_dct_lst=[
                {
                    "input_variable": "concentration",
                    "combine_levels": False,
                    "dimensions": {
                        "species_id": (1, 2),
                        "level": (0, 1),
                        "time": (0, 3),
                    },
                    "combine_species": True,
                },
                {
                    "input_variable": "deposition",
                    "combine_deposition_types": True,
                    "combine_species": False,
                    "dimensions": {
                        "species_id": (1, 2),
                        "time": (3, 6),
                        "deposition_type": ("dry", "wet"),
                    },
                },
            ],
            sol=[
                [merge_dicts(dct, {"input_variable": "concentration"})]
                for dct in [
                    {"dimensions": {"level": 0, "species_id": (1, 2), "time": 0}},
                    {"dimensions": {"level": 0, "species_id": (1, 2), "time": 3}},
                    {"dimensions": {"level": 1, "species_id": (1, 2), "time": 0}},
                    {"dimensions": {"level": 1, "species_id": (1, 2), "time": 3}},
                ]
            ]
            + [
                [
                    {
                        "input_variable": "deposition",
                        "dimensions": merge_dicts(
                            dims, {"deposition_type": ("dry", "wet")}
                        ),
                    }
                ]
                for dims in [
                    {"species_id": 1, "time": 3},
                    {"species_id": 1, "time": 6},
                    {"species_id": 2, "time": 3},
                    {"species_id": 2, "time": 6},
                ]
            ],
        ),
        ConfMultipleSetups(  # [conf2]
            setup_dct_lst=[
                {
                    "input_variable": "concentration",
                    "combine_levels": True,
                    "combine_species": False,
                    "dimensions": {"species_id": (1, 2), "level": (0, 1), "time": 0},
                },
                {
                    "input_variable": "concentration",
                    "combine_levels": False,
                    "combine_species": True,
                    "dimensions": {
                        "species_id": (1, 2),
                        "level": (0, 1),
                        "time": (0, 3),
                    },
                },
            ],
            sol=[
                [{"dimensions": {"level": (0, 1), "species_id": 1, "time": 0}}],
                [{"dimensions": {"level": (0, 1), "species_id": 2, "time": 0}}],
                [{"dimensions": {"level": 0, "species_id": (1, 2), "time": 0}}],
                [{"dimensions": {"level": 0, "species_id": (1, 2), "time": 3}}],
                [{"dimensions": {"level": 1, "species_id": (1, 2), "time": 0}}],
                [{"dimensions": {"level": 1, "species_id": (1, 2), "time": 3}}],
            ],
        ),
    ],
)
def test_multiple_setups(datadir: str, config: ConfMultipleSetups):
    params = [
        "input_variable",
        "dimensions.deposition_type",
        "dimensions.level",
        "dimensions.species_id",
        "dimensions.time",
    ]
    for setup_dct in config.setup_dct_lst:
        setup_dct.update(
            {
                "infile": f"{datadir}/{datafilename1}",
                "outfile": "foo.png",
                "model": "COSMO-1",
            }
        )
    _test_multiple_setups_core(config, params)
