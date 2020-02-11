#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import numpy as np
import pytest

from pydantic.dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pyflexplot.io import FileReader
from pyflexplot.field_specs import FieldSpecs
from pyflexplot.var_specs import MultiVarSpecs

from utils import datadir
from io_utils import read_nc_var


def get_var_name_ref(var_specs, var_names_ref):
    if var_specs.issubcls("concentration"):
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif var_specs.issubcls("deposition"):
        for var_name in var_names_ref:
            if (var_specs.deposition, var_name[:2]) in [("dry", "DD"), ("wet", "WD")]:
                return var_name
    raise NotImplementedError(f"{var_specs}")


# @dataclass(frozen=True)
@dataclass
class Conf:
    datafilename: str
    name: str
    var_names_ref: List[str]
    var_specs_dct: Dict[str, Any]
    scale_fld_ref: Optional[float] = 1.0


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"


@pytest.mark.parametrize(
    "conf",
    [
        Conf(
            datafilename=datafilename1,
            name="concentration",
            var_names_ref=[f"spec002"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": 3,
                "level": 1,
                "integrate": False,
                "species_id": 2,
            },
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"DD_spec002"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": 3,
                "integrate": False,
                "species_id": 2,
                "deposition": "dry",
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec002"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": 3,
                "integrate": False,
                "species_id": 2,
                "deposition": "wet",
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec002", f"DD_spec002",],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": 3,
                "integrate": False,
                "species_id": 2,
                "deposition": ("wet", "dry"),
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename2,
            name="concentration",
            var_names_ref=[f"spec001"],
            var_specs_dct={
                "nageclass": 0,
                "noutrel": 0,
                "time": 3,
                "level": 1,
                "integrate": False,
                "species_id": 1,
            },
        ),
        Conf(
            datafilename=datafilename2,
            name="deposition",
            var_names_ref=[f"WD_spec001", f"DD_spec001",],
            var_specs_dct={
                "nageclass": 0,
                "noutrel": 0,
                "time": 3,
                "integrate": False,
                "species_id": 1,
                "deposition": ("wet", "dry"),
            },
            scale_fld_ref=1 / 3,
        ),
    ],
)
def test_single(datadir, conf):
    """Read a single field."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Initialize variable specifications
    multi_var_specs_lst = MultiVarSpecs.create(
        conf.name, conf.var_specs_dct, lang=None, words=None,
    )
    assert len(multi_var_specs_lst) == 1
    multi_var_specs = next(iter(multi_var_specs_lst))

    # Initialize field specifications
    fld_specs = FieldSpecs(conf.name, multi_var_specs)

    # Read input field
    flex_field = FileReader(datafile).run(fld_specs)
    fld = flex_field.fld

    # Read reference field
    fld_ref = (
        np.nansum(
            [
                read_nc_var(
                    datafile,
                    get_var_name_ref(var_specs, conf.var_names_ref),
                    var_specs,
                )
                for var_specs in multi_var_specs
            ],
            axis=0,
        )
        * conf.scale_fld_ref
    )

    # Check array
    assert fld.shape == fld_ref.shape
    np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)


@pytest.mark.parametrize(
    "conf",
    [
        Conf(
            datafilename=datafilename1,
            name="concentration",
            var_names_ref=[f"spec002"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": [0, 3, 9],
                "level": [0, 2],
                "integrate": True,
                "species_id": 2,
            },
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"DD_spec002"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": [0, 3, 9],
                "integrate": True,
                "species_id": 2,
                "deposition": "dry",
            },
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec002"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": [0, 3, 9],
                "integrate": True,
                "species_id": 2,
                "deposition": "wet",
            },
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            var_specs_dct={
                "nageclass": 0,
                "numpoint": 0,
                "time": [0, 3, 9],
                "integrate": True,
                "species_id": 1,
                "deposition": ("wet", "dry"),
            },
        ),
        Conf(
            datafilename=datafilename2,
            name="concentration",
            var_names_ref=[f"spec001"],
            var_specs_dct={
                "nageclass": 0,
                "noutrel": 0,
                "time": [0, 3, 9],
                "level": [0, 2],
                "integrate": True,
                "species_id": 1,
            },
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename2,
            name="deposition",
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            var_specs_dct={
                "nageclass": 0,
                "noutrel": 0,
                "time": [0, 3, 9],
                "integrate": True,
                "species_id": 1,
                "deposition": ("wet", "dry"),
            },
        ),
    ],
)
def test_multiple(datadir, conf):
    """Read multiple fields."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Create field specifications list
    multi_var_specs_lst = MultiVarSpecs.create(
        conf.name, conf.var_specs_dct, lang=None, words=None,
    )
    fld_specs_lst = [
        FieldSpecs(conf.name, multi_var_specs)
        for multi_var_specs in multi_var_specs_lst
    ]

    # Process field specifications one after another
    for fld_specs in fld_specs_lst:

        # Read input fields
        flex_field_lst = FileReader(datafile).run([fld_specs])
        fld = np.array([flex_field.fld for flex_field in flex_field_lst])
        assert fld.shape[0] == 1
        fld = fld[0]

        # Read reference fields
        fld_ref = None
        for var_specs in fld_specs.multi_var_specs:
            flds_ref_i = [
                read_nc_var(
                    datafile,
                    get_var_name_ref(var_specs, conf.var_names_ref),
                    var_specs,
                )
            ]
            fld_ref_i = np.nansum(flds_ref_i, axis=0)
            if fld_ref is None:
                fld_ref = fld_ref_i
            else:
                fld_ref += fld_ref_i
        fld_ref *= conf.scale_fld_ref

        assert fld.shape == fld_ref.shape
        assert np.isclose(np.nanmean(fld), np.nanmean(fld_ref), rtol=1e-6)
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)
