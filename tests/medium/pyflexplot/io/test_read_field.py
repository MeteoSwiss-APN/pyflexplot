#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
# Standard library
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Third-party
import numpy as np
import pytest

# First-party
from pyflexplot.field_specs import FieldSpecs
from pyflexplot.io import FileReader
from pyflexplot.setup import Setup
from pyflexplot.var_specs import MultiVarSpecs

from io_utils import read_nc_var  # isort:skip
from utils import datadir  # noqa:F401 isort:skip


def get_var_name_ref(var_specs, var_names_ref):
    if var_specs._setup.variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif var_specs._setup.variable == "deposition":
        for var_name in var_names_ref:
            if (var_specs.deposition, var_name[:2]) in [("dry", "DD"), ("wet", "WD")]:
                return var_name
    raise NotImplementedError(f"{var_specs}")


@dataclass
class Conf:
    datafilename: str
    name: str
    var_names_ref: List[str]
    var_specs_dct: Dict[str, Any]
    setup: Setup
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
                "species_id": 2,
                "level": 1,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                species_id=2,
                level_idx=1,
                integrate=False,
                time_idx=3,
            ),
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"DD_spec002"],
            var_specs_dct={
                "deposition": "dry",
                "species_id": 2,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="dry",
                species_id=2,
                integrate=False,
                time_idx=3,
            ),
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec002"],
            var_specs_dct={
                "deposition": "wet",
                "species_id": 2,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="wet",
                species_id=2,
                integrate=False,
                time_idx=3,
            ),
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec002", f"DD_spec002"],
            var_specs_dct={
                "deposition": ("wet", "dry"),
                "species_id": 2,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=2,
                integrate=False,
                time_idx=3,
            ),
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename2,
            name="concentration",
            var_names_ref=[f"spec001"],
            var_specs_dct={
                "species_id": 1,
                "level": 1,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                level_idx=1,
                species_id=1,
                integrate=False,
                time_idx=3,
            ),
        ),
        Conf(
            datafilename=datafilename2,
            name="deposition",
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            var_specs_dct={
                "deposition": ("wet", "dry"),
                "species_id": 1,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=1,
                integrate=False,
                time_idx=3,
            ),
            scale_fld_ref=1 / 3,
        ),
    ],
)
def test_single(datadir, conf):  # noqa:F811
    """Read a single field."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Initialize variable specifications
    multi_var_specs_lst = MultiVarSpecs.create(
        conf.setup, conf.var_specs_dct, lang=None, words=None,
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
                "level": [0, 2],
                "species_id": 2,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                # + level_idx=(0, 2),  # SR_TODO
                species_id=2,
                integrate=True,
                # + time_idx=(0, 3, 9),  # SR_TODO
            ),
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"DD_spec002"],
            var_specs_dct={
                "deposition": "dry",
                "species_id": 2,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="dry",
                species_id=2,
                integrate=True,
                # + time_idx=(0, 3, 9),  # SR_TODO
            ),
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec002"],
            var_specs_dct={
                "deposition": "wet",
                "species_id": 2,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="wet",
                species_id=2,
                integrate=True,
                # + time_idx=(0, 3, 9),  # SR_TODO
            ),
        ),
        Conf(
            datafilename=datafilename1,
            name="deposition",
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            var_specs_dct={
                "deposition": ("wet", "dry"),
                "species_id": 1,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "numpoint": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=1,
                integrate=True,
                # + time_idx=(0, 3, 9),  # SR_TODO
            ),
        ),
        Conf(
            datafilename=datafilename2,
            name="concentration",
            var_names_ref=[f"spec001"],
            var_specs_dct={
                "level": [0, 2],
                "species_id": 1,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                # + level_idx=(0, 2),  # SR_TODO
                species_id=1,
                integrate=True,
                # + time_idx=(0, 3, 9),  # SR_TODO
            ),
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename2,
            name="deposition",
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            var_specs_dct={
                "deposition": ("wet", "dry"),
                "species_id": 1,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
            },
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=1,
                integrate=True,
                # + time_idx=(0, 3, 9),  # SR_TODO
            ),
        ),
    ],
)
def test_multiple(datadir, conf):  # noqa:F811
    """Read multiple fields."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Create field specifications list
    multi_var_specs_lst = MultiVarSpecs.create(
        conf.setup, conf.var_specs_dct, lang=None, words=None,
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
