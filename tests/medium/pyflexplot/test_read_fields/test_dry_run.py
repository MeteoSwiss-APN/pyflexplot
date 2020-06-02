# -*- coding: utf-8 -*-
"""
Tests for function ``pyflexplot.input.read_fields`` in dry-run mode.
"""
# Standard library
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import overload

# Third-party
import pytest

# First-party
from pyflexplot.data import Field
from pyflexplot.input import read_fields
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection
from srutils.testing import check_summary_dict_element_is_subelement

# Local  isort:skip
from shared import datadir  # noqa:F401 isort:skip


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"
datafilename3 = "flexpart_ifs_20200317000000.nc"


@overload
def fields_to_setup_dcts(obj: Field) -> Dict[str, Any]:
    ...


@overload
def fields_to_setup_dcts(obj: Sequence[Field]) -> Sequence[Dict[str, Any]]:
    ...


@overload
def fields_to_setup_dcts(obj: Sequence[Sequence]) -> Sequence[Sequence]:
    ...


def fields_to_setup_dcts(obj):
    """Turn one or more fields into a dict each based on their input setups.

    Multiple fields may be passed in nested sequences, in which case the nesting
    structure is retained.

    """
    if isinstance(obj, Sequence):
        return [fields_to_setup_dcts(sub_obj) for sub_obj in obj]
    else:
        assert isinstance(obj, Field)
        return obj.var_setups.compress().dict()


@dataclass
class ConfSingleSetup:
    setup_dct: Dict[str, Any]
    sol: List[List[Dict[str, Any]]]


# test_single_setup
@pytest.mark.parametrize(
    "conf",
    [
        ConfSingleSetup(  # [conf0]
            setup_dct={
                "input_variable": "concentration",
                "species_id": (1,),
                "time": (0,),
            },
            sol=[[{"species_id": (1,), "time": (0,)}]],
        ),
        ConfSingleSetup(  # [conf1]
            setup_dct={
                "input_variable": "concentration",
                "species_id": (1, 2),
                "combine_species": True,
                "time": (0,),
            },
            sol=[[{"species_id": (1, 2), "time": (0,)}]],
        ),
        # ConfSingleSetup(  # [conf2]
        #     setup_dct={
        #         "input_variable": "concentration",
        #         "species_id": (1, 2),
        #         "combine_species": False,
        #         "time": (0,),
        #     },
        #     sol=[
        #         [{"species_id": (1,), "time": (0,)}],
        #         [{"species_id": (2,), "time": (0,)}],
        #     ],
        # ),
    ],
)
def test_single_setup(datadir: str, conf: ConfSingleSetup):
    datafile = f"{datadir}/{datafilename1}"
    setup = InputSetup(infile=datafile, outfile="foo.png", **conf.setup_dct)
    setup_lst = [setup]
    setups = InputSetupCollection(setup_lst)
    field_lst_lst, mdata_lst_lst = read_fields(datafile, setups, dry_run=True)
    res = fields_to_setup_dcts(field_lst_lst)
    check_summary_dict_element_is_subelement(obj_super=res, obj_sub=conf.sol)
