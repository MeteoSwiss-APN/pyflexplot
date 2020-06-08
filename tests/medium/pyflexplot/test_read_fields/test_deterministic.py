# -*- coding: utf-8 -*-
"""
Tests for function ``pyflexplot.input.read_fields`` for deterministic data.
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
from pyflexplot.input import read_fields
from pyflexplot.setup import InputSetup
from pyflexplot.setup import InputSetupCollection

# Local  isort:skip
from shared import read_nc_var  # isort:skip
from shared import datadir_reduced as datadir  # noqa:F401 isort:skip


def get_var_name_ref(setup, var_names_ref):
    if setup.input_variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.input_variable == "deposition":
        species_id = setup.dimensions.species_id
        if isinstance(species_id, tuple):
            assert len(species_id) == 1
            species_id = next(iter(species_id))
        var_name = f"{setup.deposition_type_str[0].upper()}D_spec{species_id:03d}"
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


# test_single
@pytest.mark.parametrize(
    "conf",
    [
        Conf(  # [conf0]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "integrate": False,
                "dimensions": {
                    "level": 1,
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
        ),
        Conf(  # [conf1]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "dry",
                "integrate": False,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(  # [conf2]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "wet",
                "integrate": False,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(  # [conf3]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "combine_deposition_types": True,
                "integrate": False,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(  # [conf4]
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "integrate": False,
                "dimensions": {
                    "level": 1,
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": 3,
                },
            },
        ),
        Conf(  # [conf5]
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "combine_deposition_types": True,
                "integrate": False,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(  # [conf6]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["DD_spec001", "DD_spec002", "WD_spec001", "WD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "combine_deposition_types": True,
                "combine_species": True,
                "integrate": False,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": [1, 2],
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Conf(  # [conf7]
            datafilename=datafilename3,
            model="ifs",
            var_names_ref=["spec001_mr"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "integrate": False,
                "dimensions": {
                    "level": 1,
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": 10,
                },
            },
        ),
    ],
)
def test_single(datadir, conf):  # noqa:F811
    """Read a single field."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Initialize field specifications
    setups = InputSetupCollection([conf.setup])

    # Read input field
    field_lst_lst, mdata_lst_lst = read_fields(datafile, setups)
    assert len(field_lst_lst) == 1
    assert len(field_lst_lst[0]) == 1
    fld = field_lst_lst[0][0].fld

    # Initialize individual setup objects
    var_setups_lst = setups.decompress_twice("dimensions.time", skip=["ens_member_id"])
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


# test_multiple
@pytest.mark.parametrize(
    "conf",
    [
        Conf(  # [conf0]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "integrate": True,
                "dimensions": {
                    "level": [0, 2],
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": [0, 3],
                },
            },
            scale_fld_ref=3.0,
        ),
        Conf(  # [conf1]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "integrate": True,
                "dimensions": {
                    "level": [0, 2],
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": [0, 3],
                },
            },
            scale_fld_ref=3.0,
        ),
        Conf(  # [conf2]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "dry",
                "integrate": True,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": [0, 3, 9],
                },
            },
        ),
        Conf(  # [conf3]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": "wet",
                "integrate": True,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": [0, 3, 9],
                },
            },
        ),
        Conf(  # [conf4]
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "combine_deposition_types": True,
                "integrate": True,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": [0, 3, 9],
                },
            },
        ),
        Conf(  # [conf5]
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "concentration",
                "integrate": True,
                "dimensions": {
                    "level": 0,
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": [0, 3, 9],
                },
            },
            derived_setup_params=[{"dimensions": {"level": 2}}],
            scale_fld_ref=3.0,
        ),
        Conf(  # [conf6]
            datafilename=datafilename2,
            model="cosmo1",
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["wet", "dry"],
                "integrate": True,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": [0, 3, 9],
                },
            },
        ),
    ],
)
def test_multiple(datadir, conf):  # noqa:F811
    """Read multiple fields."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Create setups
    setup_lst = list(conf.setup.decompress_partially(None))
    for setup in setup_lst.copy():
        setup_lst.extend(setup.derive(conf.derived_setup_params))
    setups = InputSetupCollection(setup_lst)

    # Process field specifications one after another
    var_setups: InputSetupCollection
    for var_setups in setups.decompress_twice(
        "dimensions.time", skip=["ens_member_id"]
    ):

        # Read input fields
        var_setups_dicts_pre = var_setups.dicts()
        field_lst_lst, mdata_lst_lst = read_fields(datafile, var_setups)
        assert var_setups.dicts() == var_setups_dicts_pre
        assert len(field_lst_lst) == 1
        assert len(mdata_lst_lst) == 1
        assert len(field_lst_lst[0]) == 1
        assert len(mdata_lst_lst[0]) == 1
        fld = np.array(
            [field.fld for field_lst in field_lst_lst for field in field_lst]
        )
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


# test_single_add_ts0
@pytest.mark.skip("test_single_add_ts0: TODO fix/implement")
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
                "integrate": False,
                "dimensions": {
                    "level": 1,
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": None,
                },
            },
        ),
        Conf(
            datafilename=datafilename1,
            model="cosmo1",
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "infile": "dummy.nc",
                "outfile": "dummy.png",
                "input_variable": "deposition",
                "deposition_type": ["dry", "wet"],
                "combine_deposition_types": True,
                "integrate": False,
                "dimensions": {
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": None,
                },
            },
            scale_fld_ref=1 / 3,
        ),
    ],
)
def test_single_add_ts0(datadir, conf):
    """Insert an additional time step 0 in the beginning, with empty fields."""

    datafile = f"{datadir}/{conf.datafilename}"

    # Initialize field specifications
    setups = InputSetupCollection([conf.setup])

    # Read fields with and without added time step 0
    field_raw_lst_lst, _ = read_fields(datafile, setups, add_ts0=False)
    field_ts0_lst_lst, _ = read_fields(datafile, setups, add_ts0=True)
    field_raw_lst_lst, _ = read_fields(datafile, setups, add_ts0=False)
    field_ts0_lst_lst, _ = read_fields(datafile, setups, add_ts0=True)
    assert len(field_ts0_lst_lst) == len(field_raw_lst_lst) + 1
    assert all(len(field_lst) == 1 for field_lst in field_ts0_lst_lst)
    assert all(len(field_lst) == 1 for field_lst in field_raw_lst_lst)
    assert (field_ts0_lst_lst[0][0].fld == 0.0).all()
    assert (field_ts0_lst_lst[1][0].fld == field_raw_lst_lst[0][0].fld).all()
