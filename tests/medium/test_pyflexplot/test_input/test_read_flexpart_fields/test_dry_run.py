"""Tests for module ``pyflexplot.input.read_fields``.

These test the dry-run mode.

"""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence

# Third-party
import pytest

# First-party
from pyflexplot.input.field import Field
from pyflexplot.input.field import FieldGroup
from pyflexplot.input.read_fields import read_fields
from pyflexplot.setups.dimensions import is_dimensions_param
from pyflexplot.setups.model_setup import is_model_setup_param
from pyflexplot.setups.plot_panel_setup import is_plot_panel_setup_param
from pyflexplot.setups.plot_setup import is_plot_setup_param
from pyflexplot.setups.plot_setup import PlotSetupGroup
from srutils.dict import merge_dicts
from srutils.testing import assert_is_sub_element

# Local
from .shared import datadir_reduced as datadir  # noqa:F401

datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"
datafilename3 = "flexpart_ifs_20200317000000.nc"


@overload
def field_groups_to_setup_dcts(
    obj: Field, params: Optional[List[str]] = ...
) -> Dict[str, Any]:
    ...


@overload
def field_groups_to_setup_dcts(
    obj: FieldGroup, params: Optional[List[str]] = ...
) -> Sequence[Dict[str, Any]]:
    ...


@overload
def field_groups_to_setup_dcts(
    obj: Sequence[FieldGroup], params: Optional[List[str]] = ...
) -> Sequence[Sequence[Dict[str, Any]]]:
    ...


def field_groups_to_setup_dcts(obj, params=None):
    """Extract input setups from one or more ``Field`` objects.

    Multiple ``Field`` objects may be passed as arbitrarily nested sequences,
    in which case the nesting is retained in the result.

    """
    if isinstance(obj, Field):
        field = obj
        if params is None:
            return field.dict()
        panel_dct = {}
        for param in params:
            if is_plot_panel_setup_param(param):
                panel_dct[param] = getattr(field.panel_setup, param)
            elif is_dimensions_param(param):
                param = param.replace("dimensions.", "")
                if "dimensions" not in panel_dct:
                    panel_dct["dimensions"] = {}
                panel_dct["dimensions"][param] = getattr(
                    field.panel_setup.dimensions, param
                )
        return panel_dct
    elif isinstance(obj, FieldGroup):
        field_group = obj
        if params is None:
            return field_group.dicts()
        dct_sel = {}
        for param in params:
            if is_plot_setup_param(param):
                dct_sel[param] = getattr(field_group.plot_setup, param)
            elif is_model_setup_param(param):
                param = param.replace("model.", "")
                if "model" not in dct_sel:
                    dct_sel["model"] = {}
                dct_sel["model"][param] = getattr(field_group.plot_setup.model, param)
        field_dcts = []
        for field in field_group:
            field_dct_i = field_groups_to_setup_dcts(field, params)
            if field_dct_i:
                field_dcts.append(field_dct_i)
        if field_dcts:
            dct_sel["panels"] = field_dcts
        return dct_sel
    elif isinstance(obj, Sequence):
        return [field_groups_to_setup_dcts(sub_obj, params) for sub_obj in obj]


def _test_setups_core(
    setups: PlotSetupGroup, params: List[str], sol: List[List[Dict[str, Any]]]
):
    field_groups = read_fields(setups, {"dry_run": True})
    res = field_groups_to_setup_dcts(field_groups, params)
    assert_is_sub_element(
        obj_super=res, name_super="result", obj_sub=sol, name_sub="solution"
    )


@dc.dataclass
class ConfSingleSetup:
    setup_dct: Dict[str, Any]
    sol: List[Dict[str, Any]]


@dc.dataclass
class ConfMultipleSetups:
    setup_dct_lst: List[Dict[str, Any]]
    sol: List[Dict[str, Any]]


def _test_single_setup_core(config, params):
    setups = PlotSetupGroup.create(config.setup_dct)
    _test_setups_core(setups, params, config.sol)


def _test_multiple_setups_core(config, params):
    setups = PlotSetupGroup.create(config.setup_dct_lst)
    _test_setups_core(setups, params, config.sol)


# test_single_setup_concentration
@pytest.mark.parametrize(
    "config",
    [
        ConfSingleSetup(  # [conf0]
            setup_dct={
                "panels": [
                    {
                        "dimensions": {"species_id": 1, "level": 0, "time": 0},
                    }
                ]
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": 1, "time": 0}}]},
            ],
        ),
        ConfSingleSetup(  # [conf1]
            setup_dct={
                "panels": [
                    {
                        "dimensions": {
                            "species_id": 1,
                            "level": 0,
                            "time": (0, 3, 6),
                        }
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": 1, "time": 0}}]},
                {"panels": [{"dimensions": {"species_id": 1, "time": 3}}]},
                {"panels": [{"dimensions": {"species_id": 1, "time": 6}}]},
            ],
        ),
        ConfSingleSetup(  # [conf2]
            setup_dct={
                "panels": [
                    {
                        "combine_species": True,
                        "dimensions": {
                            "species_id": (1, 2),
                            "level": 0,
                            "time": 0,
                        },
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": (1, 2), "time": 0}}]},
            ],
        ),
        ConfSingleSetup(  # [conf3]
            setup_dct={
                "panels": [
                    {
                        "combine_species": True,
                        "dimensions": {
                            "species_id": (1, 2),
                            "level": 0,
                            "time": (0, 3, 6),
                        },
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": (1, 2), "time": 0}}]},
                {"panels": [{"dimensions": {"species_id": (1, 2), "time": 3}}]},
                {"panels": [{"dimensions": {"species_id": (1, 2), "time": 6}}]},
            ],
        ),
        ConfSingleSetup(  # [conf4]
            setup_dct={
                "panels": [
                    {
                        "combine_species": False,
                        "dimensions": {
                            "species_id": (1, 2),
                            "level": 0,
                            "time": 0,
                        },
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": 1, "time": 0}}]},
                {"panels": [{"dimensions": {"species_id": 2, "time": 0}}]},
            ],
        ),
        ConfSingleSetup(  # [conf5]
            setup_dct={
                "panels": [
                    {
                        "combine_species": False,
                        "dimensions": {
                            "species_id": (1, 2),
                            "level": 0,
                            "time": (0, 3, 6),
                        },
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": 1, "time": 0}}]},
                {"panels": [{"dimensions": {"species_id": 1, "time": 3}}]},
                {"panels": [{"dimensions": {"species_id": 1, "time": 6}}]},
                {"panels": [{"dimensions": {"species_id": 2, "time": 0}}]},
                {"panels": [{"dimensions": {"species_id": 2, "time": 3}}]},
                {"panels": [{"dimensions": {"species_id": 2, "time": 6}}]},
            ],
        ),
        ConfSingleSetup(  # [conf6]
            setup_dct={
                "panels": [
                    {
                        "dimensions": {"species_id": 1, "level": (0, 1), "time": 0},
                        "combine_levels": True,
                    }
                ],
            },
            sol=[
                {
                    "panels": [
                        {
                            "dimensions": {
                                "species_id": 1,
                                "level": (0, 1),
                                "time": 0,
                            }
                        }
                    ],
                }
            ],
        ),
        ConfSingleSetup(  # [conf7]
            setup_dct={
                "panels": [
                    {
                        "combine_levels": False,
                        "dimensions": {
                            "species_id": 1,
                            "level": (0, 1),
                            "time": 0,
                        },
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"species_id": 1, "level": 0, "time": 0}}]},
                {"panels": [{"dimensions": {"species_id": 1, "level": 1, "time": 0}}]},
            ],
        ),
        ConfSingleSetup(  # [conf8]
            setup_dct={
                "panels": [
                    {
                        "combine_levels": False,
                        "combine_species": False,
                        "dimensions": {
                            "species_id": (1, 2),
                            "level": (0, 1),
                            "time": (0, 3, 6),
                        },
                    }
                ],
            },
            sol=[
                {"panels": [{"dimensions": {"level": 0, "species_id": 1, "time": 0}}]},
                {"panels": [{"dimensions": {"level": 0, "species_id": 1, "time": 3}}]},
                {"panels": [{"dimensions": {"level": 0, "species_id": 1, "time": 6}}]},
                {"panels": [{"dimensions": {"level": 0, "species_id": 2, "time": 0}}]},
                {"panels": [{"dimensions": {"level": 0, "species_id": 2, "time": 3}}]},
                {"panels": [{"dimensions": {"level": 0, "species_id": 2, "time": 6}}]},
                {"panels": [{"dimensions": {"level": 1, "species_id": 1, "time": 0}}]},
                {"panels": [{"dimensions": {"level": 1, "species_id": 1, "time": 3}}]},
                {"panels": [{"dimensions": {"level": 1, "species_id": 1, "time": 6}}]},
                {"panels": [{"dimensions": {"level": 1, "species_id": 2, "time": 0}}]},
                {"panels": [{"dimensions": {"level": 1, "species_id": 2, "time": 3}}]},
                {"panels": [{"dimensions": {"level": 1, "species_id": 2, "time": 6}}]},
            ],
        ),
    ],
)
def test_single_setup_concentration(datadir: str, config: ConfSingleSetup):
    params = ["dimensions.species_id", "dimensions.level", "dimensions.time"]
    config.setup_dct = merge_dicts(
        config.setup_dct,
        {
            "infile": f"{datadir}/{datafilename1}",
            "outfile": "foo.png",
            "model": {
                "name": "COSMO-1",
            },
            "panels": [
                {
                    "plot_variable": "concentration",
                }
            ],
        },
    )
    _test_single_setup_core(config, params)


# test_single_setup_deposition
@pytest.mark.parametrize(
    "config",
    [
        ConfSingleSetup(  # [conf0]
            setup_dct={
                "panels": [
                    {
                        "plot_variable": "dry_deposition",
                        "dimensions": {
                            "species_id": 1,
                            "time": 0,
                        },
                    }
                ],
            },
            sol=[
                {
                    "panels": [
                        {
                            "plot_variable": "dry_deposition",
                            "dimensions": {
                                "species_id": 1,
                                "time": 0,
                                "variable": "dry_deposition",
                            },
                        }
                    ],
                },
            ],
        ),
        ConfSingleSetup(  # [conf1]
            setup_dct={
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "combine_species": False,
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": (0, 1),
                        },
                    }
                ],
            },
            sol=[
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": 1,
                                "time": 0,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": 1,
                                "time": 1,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": 2,
                                "time": 0,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": 2,
                                "time": 1,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
            ],
        ),
        ConfSingleSetup(  # [conf2]
            setup_dct={
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "combine_species": True,
                        "dimensions": {
                            "species_id": (1, 2),
                            "time": (0, 3, 6),
                        },
                    }
                ],
            },
            sol=[
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": (1, 2),
                                "time": 0,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": (1, 2),
                                "time": 3,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": {
                                "species_id": (1, 2),
                                "time": 6,
                                "variable": ("dry_deposition", "wet_deposition"),
                            },
                        }
                    ],
                },
            ],
        ),
    ],
)
def test_single_setup_deposition(datadir: str, config: ConfSingleSetup):
    params = [
        "plot_variable",
        "dimensions.species_id",
        "dimensions.time",
        "dimensions.variable",
    ]
    config.setup_dct = merge_dicts(
        config.setup_dct,
        {
            "infile": f"{datadir}/{datafilename1}",
            "outfile": "foo.png",
            "model": {"name": "COSMO-1"},
        },
    )
    _test_single_setup_core(config, params)


# test_multiple_setups
@pytest.mark.parametrize(
    "config",
    [
        ConfMultipleSetups(  # [conf0]
            setup_dct_lst=[
                {
                    "panels": [
                        {
                            "plot_variable": "concentration",
                            "dimensions": {"species_id": 1, "level": 0, "time": 0},
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "dry_deposition",
                            "dimensions": {
                                "species_id": 1,
                                "time": 0,
                            },
                        }
                    ],
                },
            ],
            sol=[
                {"panels": [{"plot_variable": "concentration"}]},
                {"panels": [{"plot_variable": "dry_deposition"}]},
            ],
        ),
        ConfMultipleSetups(  # [conf1]
            setup_dct_lst=[
                {
                    "panels": [
                        {
                            "plot_variable": "concentration",
                            "combine_levels": False,
                            "dimensions": {
                                "species_id": (1, 2),
                                "level": (0, 1),
                                "time": (0, 3),
                            },
                            "combine_species": True,
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "combine_species": False,
                            "dimensions": {
                                "species_id": (1, 2),
                                "time": (3, 6),
                            },
                        }
                    ],
                },
            ],
            sol=[
                merge_dicts(dct, {"panels": [{"plot_variable": "concentration"}]})
                for dct in [
                    {
                        "panels": [
                            {
                                "dimensions": {
                                    "level": 0,
                                    "species_id": (1, 2),
                                    "time": 0,
                                }
                            }
                        ],
                    },
                    {
                        "panels": [
                            {
                                "dimensions": {
                                    "level": 0,
                                    "species_id": (1, 2),
                                    "time": 3,
                                }
                            }
                        ],
                    },
                    {
                        "panels": [
                            {
                                "dimensions": {
                                    "level": 1,
                                    "species_id": (1, 2),
                                    "time": 0,
                                }
                            }
                        ],
                    },
                    {
                        "panels": [
                            {
                                "dimensions": {
                                    "level": 1,
                                    "species_id": (1, 2),
                                    "time": 3,
                                }
                            }
                        ],
                    },
                ]
            ]
            + [
                {
                    "panels": [
                        {
                            "plot_variable": "tot_deposition",
                            "dimensions": dims,
                        }
                    ],
                }
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
                    "panels": [
                        {
                            "plot_variable": "concentration",
                            "combine_levels": True,
                            "combine_species": False,
                            "dimensions": {
                                "species_id": (1, 2),
                                "level": (0, 1),
                                "time": 0,
                            },
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "plot_variable": "concentration",
                            "combine_levels": False,
                            "combine_species": True,
                            "dimensions": {
                                "species_id": (1, 2),
                                "level": (0, 1),
                                "time": (0, 3),
                            },
                        }
                    ],
                },
            ],
            sol=[
                {
                    "panels": [
                        {
                            "dimensions": {
                                "level": (0, 1),
                                "species_id": 1,
                                "time": 0,
                            }
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "dimensions": {
                                "level": (0, 1),
                                "species_id": 2,
                                "time": 0,
                            }
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "dimensions": {
                                "level": 0,
                                "species_id": (1, 2),
                                "time": 0,
                            }
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "dimensions": {
                                "level": 0,
                                "species_id": (1, 2),
                                "time": 3,
                            }
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "dimensions": {
                                "level": 1,
                                "species_id": (1, 2),
                                "time": 0,
                            }
                        }
                    ],
                },
                {
                    "panels": [
                        {
                            "dimensions": {
                                "level": 1,
                                "species_id": (1, 2),
                                "time": 3,
                            }
                        }
                    ],
                },
            ],
        ),
    ],
)
def test_multiple_setups(datadir: str, config: ConfMultipleSetups):
    params = [
        "plot_variable",
        "dimensions.level",
        "dimensions.species_id",
        "dimensions.time",
    ]
    for setup_dct in config.setup_dct_lst:
        setup_dct.update(
            {
                "infile": f"{datadir}/{datafilename1}",
                "outfile": "foo.png",
                "model": {
                    "name": "COSMO-1",
                },
            }
        )
    _test_multiple_setups_core(config, params)
