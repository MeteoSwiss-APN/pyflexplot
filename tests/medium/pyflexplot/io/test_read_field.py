#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.io``.
"""
# Standard library
from dataclasses import dataclass
from dataclasses import field
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


def get_var_name_ref(setup, var_names_ref):
    if setup.variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.variable == "deposition":
        for var_name in var_names_ref:
            if (setup.deposition_type, var_name[:2]) in [("dry", "DD"), ("wet", "WD")]:
                return var_name
    raise NotImplementedError(f"{setup}")


@dataclass
class Conf:
    datafilename: str
    var_names_ref: List[str]
    setup: Setup
    derived_setup_params: List[Dict[str, Any]] = field(default_factory=list)
    scale_fld_ref: Optional[float] = 1.0


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"


@pytest.mark.parametrize(
    "conf",
    [
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                species_id=2,
                level_idx=1,
                integrate=False,
                time_idcs=[3],
            ),
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"DD_spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="dry",
                species_id=2,
                integrate=False,
                time_idcs=[3],
            ),
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"WD_spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="wet",
                species_id=2,
                integrate=False,
                time_idcs=[3],
            ),
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"WD_spec002", f"DD_spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=2,
                integrate=False,
                time_idcs=[3],
            ),
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename2,
            var_names_ref=[f"spec001"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                level_idx=1,
                species_id=1,
                integrate=False,
                time_idcs=[3],
            ),
        ),
        Conf(
            datafilename=datafilename2,
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=1,
                integrate=False,
                time_idcs=[3],
            ),
            scale_fld_ref=1 / 3,
        ),
    ],
)
def test_single(datadir, conf):  # noqa:F811
    """Read a single field."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Initialize variable specifications
    multi_var_specs_lst = MultiVarSpecs.create(conf.setup)
    assert len(multi_var_specs_lst) == 1
    multi_var_specs = next(iter(multi_var_specs_lst))
    setups = multi_var_specs.setup.decompress()

    # Initialize field specifications
    fld_specs = FieldSpecs(multi_var_specs)

    # Read input field
    flex_field = FileReader(datafile).run(fld_specs)
    fld = flex_field.fld

    # Read reference field
    fld_ref = (
        np.nansum(
            [
                read_nc_var(
                    datafile, get_var_name_ref(setup, conf.var_names_ref), setup,
                )
                for setup in setups
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
            var_names_ref=[f"spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                level_idx=[0, 2],
                species_id=2,
                integrate=True,
                time_idcs=[0, 3],
            ),
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                level_idx=[0, 2],
                species_id=2,
                integrate=True,
                time_idcs=[0, 3],
            ),
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"DD_spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="dry",
                species_id=2,
                integrate=True,
                time_idcs=[0, 3, 9],
            ),
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"WD_spec002"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="wet",
                species_id=2,
                integrate=True,
                time_idcs=[0, 3, 9],
            ),
        ),
        Conf(
            datafilename=datafilename1,
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type="tot",
                species_id=1,
                integrate=True,
                time_idcs=[0, 3, 9],
            ),
        ),
        Conf(
            datafilename=datafilename2,
            var_names_ref=[f"spec001"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="concentration",
                level_idx=0,
                species_id=1,
                integrate=True,
                time_idcs=[0, 3, 9],
            ),
            derived_setup_params=[{"level_idx": 2}],
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename2,
            var_names_ref=[f"WD_spec001", f"DD_spec001"],
            setup=Setup(
                infiles=["dummy.nc"],
                outfile="dummy.png",
                variable="deposition",
                deposition_type=["wet", "dry"],
                species_id=1,
                integrate=True,
                time_idcs=[0, 3, 9],
            ),
        ),
    ],
)
def test_multiple(datadir, conf):  # noqa:F811
    """Read multiple fields."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Create setups
    setups = conf.setup.decompress()
    for setup in setups.copy():
        setups.extend(setup.derive(conf.derived_setup_params))

    # Create field specifications list
    multi_var_specs_lst = MultiVarSpecs.create(setups)
    fld_specs_lst = [
        FieldSpecs(multi_var_specs) for multi_var_specs in multi_var_specs_lst
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
                    get_var_name_ref(var_specs._setup, conf.var_names_ref),
                    var_specs._setup,  # SR_TMP
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
