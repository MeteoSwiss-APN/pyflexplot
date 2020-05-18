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
import pytest  # type: ignore

# First-party
from pyflexplot.io import read_files
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection

from io_utils import read_nc_var  # isort:skip
from utils import datadir  # noqa:F401 isort:skip


def get_var_name_ref(setup, var_names_ref):
    if setup.input_variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.input_variable == "deposition":
        species_id = setup.species_id
        if isinstance(species_id, tuple):
            assert len(species_id) == 1
            species_id = next(iter(species_id))
        var_name = f"{setup.deposition_type[0].upper()}D_spec{species_id:03d}"
        if var_name in var_names_ref:
            return var_name
    raise NotImplementedError(f"{setup}")


@dataclass
class Conf:
    datafilename: str
    model: str
    var_names_ref: List[str]
    setup_dct: Dict[str, Any]
    derived_setup_params: List[Dict[str, Any]] = field(default_factory=list)
    scale_fld_ref: Optional[float] = 1.0

    @property
    def setup(self):
        return InputSetup.create(self.setup_dct)


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"
datafilename3 = "flexpart_ifs_20200317000000.nc"


@pytest.mark.parametrize(
    "conf",
    [
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "species_id": 2,
                "level": 1,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "dry",
                "species_id": 2,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "wet",
                "species_id": 2,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "tot",
                "species_id": 2,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "level": 1,
                "species_id": 1,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
        Conf(
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "tot",
                "species_id": 1,
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["DD_spec001", "DD_spec002", "WD_spec001", "WD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "tot",
                "species_id": [1, 2],
                "integrate": False,
                "time": 3,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(
            datafilename=datafilename3,
            model="ifs",
            var_names_ref=["spec001_mr"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "species_id": 1,
                "level": 1,
                "integrate": False,
                "time": 10,
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
    ],
)
def test_single(datadir, conf):  # noqa:F811
    """Read a single field."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Initialize field specifications
    var_setups_lst = InputSetupCollection([conf.setup]).decompress_grouped_by_time()
    assert len(var_setups_lst) == 1

    # Read input field
    fields, mdata_lst = read_files(datafile, var_setups_lst)
    assert len(fields) == 1
    fld = fields[0].fld

    # Initialize individual setup objects
    assert len(var_setups_lst) == 1
    var_setups = next(iter(var_setups_lst))

    # Read reference field
    fld_ref = (
        np.nansum(
            [
                read_nc_var(
                    datafile,
                    get_var_name_ref(setup, conf.var_names_ref),
                    setup,
                    conf.model,
                )
                for setup in var_setups
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
            model="cosmo1",
            var_names_ref=["spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "level": [0, 2],
                "species_id": 2,
                "integrate": True,
                "time": [0, 3],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "level": [0, 2],
                "species_id": 2,
                "integrate": True,
                "time": [0, 3],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "dry",
                "species_id": 2,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "wet",
                "species_id": 2,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "tot",
                "species_id": 1,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
        Conf(
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "level": 0,
                "species_id": 1,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
            derived_setup_params=[{"level": 2}],
            scale_fld_ref=3.0,
        ),
        Conf(
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["wet", "dry"],
                "species_id": 1,
                "integrate": True,
                "time": [0, 3, 9],
                "nageclass": 0,
                "noutrel": 0,
                "numpoint": 0,
            },
        ),
    ],
)
def test_multiple(datadir, conf):  # noqa:F811
    """Read multiple fields."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Create setups
    setups = list(conf.setup.decompress_partially(None))
    for setup in setups.copy():
        setups.extend(setup.derive(conf.derived_setup_params))

    # Create field specifications list
    var_setups_lst = InputSetupCollection(setups).decompress_grouped_by_time()

    # Process field specifications one after another
    for var_setups in var_setups_lst:

        # Read input fields
        fields, mdata_lst = read_files(datafile, [var_setups])
        assert len(fields) == 1
        assert len(mdata_lst) == 1
        fld = np.array([field.fld for field in fields])
        assert fld.shape[0] == 1
        fld = fld[0]

        # Read reference fields
        fld_ref = None
        for var_setup in var_setups:
            flds_ref_i = [
                read_nc_var(
                    datafile,
                    get_var_name_ref(var_setup, conf.var_names_ref),
                    var_setup,
                    conf.model,
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
