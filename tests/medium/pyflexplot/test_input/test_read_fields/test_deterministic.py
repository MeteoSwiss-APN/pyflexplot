"""Tests for module ``pyflexplot.input.read_fields``.

These tests use deterministic data.

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
from pyflexplot.input.read_fields import FieldInputOrganizer
from pyflexplot.input.read_fields import read_fields
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupCollection

# Local  isort:skip
from ..shared import read_nc_var  # isort:skip
from ..shared import datadir_reduced as datadir  # noqa:F401 isort:skip


def get_var_name_ref(setup, var_names_ref):
    if setup.core.input_variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif setup.core.input_variable == "deposition":
        species_id = setup.core.dimensions.species_id
        if isinstance(species_id, tuple):
            assert len(species_id) == 1
            species_id = next(iter(species_id))
        var_name = f"{setup.deposition_type_str[0].upper()}D_spec{species_id:03d}"
        if var_name in var_names_ref:
            return var_name
    raise NotImplementedError(f"{setup}")


@dataclass
class Config:
    var_names_ref: List[str]
    setup_dct: Dict[str, Any]
    derived_setup_params: List[Dict[str, Any]] = field(default_factory=list)
    scale_fld_ref: Optional[float] = 1.0

    @property
    def setup(self):
        return Setup.create(self.setup_dct)


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"
datafilename3 = "flexpart_ifs_20200317000000.nc"
datafilename4 = "flexpart_cosmo-1e-ctrl_2020102105.nc"


# test_single
@pytest.mark.parametrize(
    "config",
    [
        Config(  # [config0]
            var_names_ref=["spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
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
        Config(  # [config1]
            var_names_ref=["DD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "integrate": False,
                "dimensions": {
                    "deposition_type": "dry",
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config2]
            var_names_ref=["WD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "integrate": False,
                "dimensions": {
                    "deposition_type": "wet",
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config3]
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "combine_deposition_types": True,
                "integrate": False,
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config4]
            var_names_ref=["spec001"],
            setup_dct={
                "infile": datafilename2,
                "outfile": "dummy.png",
                "model": "COSMO-1",
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
        Config(  # [config5]
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": datafilename2,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "combine_deposition_types": True,
                "integrate": False,
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config6]
            var_names_ref=["DD_spec001", "DD_spec002", "WD_spec001", "WD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "combine_deposition_types": True,
                "combine_species": True,
                "integrate": False,
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": [1, 2],
                    "time": 3,
                },
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config7]
            var_names_ref=["spec001_mr"],
            setup_dct={
                "infile": datafilename3,
                "outfile": "dummy.png",
                "model": "IFS-HRES",
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
        Config(  # [config8]
            var_names_ref=["spec001"],
            setup_dct={
                "infile": datafilename4,
                "outfile": "dummy.png",
                "model": "COSMO-1E",
                "input_variable": "concentration",
                "integrate": False,
                "dimensions": {
                    "level": 0,
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
def test_single(datadir, config):  # noqa:F811
    """Read a single field."""

    datafile = f"{datadir}/{config.setup_dct['infile']}"

    # Initialize field specifications
    setups = SetupCollection([config.setup])

    # Read input field
    field_lst_lst = read_fields(datafile, setups)
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
                    get_var_name_ref(setup, config.var_names_ref),
                    setup,
                    config.setup_dct["model"],
                )
                for setup in var_setups
            ],
            axis=0,
        )
        * config.scale_fld_ref
    )

    # Check array
    assert fld.shape == fld_ref.shape
    np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)


# test_multiple
@pytest.mark.parametrize(
    "config",
    [
        Config(  # [config0]
            var_names_ref=["spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
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
        Config(  # [config1]
            var_names_ref=["spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
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
        Config(  # [config2]
            var_names_ref=["DD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "integrate": True,
                "dimensions": {
                    "deposition_type": "dry",
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": [0, 3, 9],
                },
            },
        ),
        Config(  # [config3]
            var_names_ref=["WD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "integrate": True,
                "dimensions": {
                    "deposition_type": "wet",
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 2,
                    "time": [0, 3, 9],
                },
            },
        ),
        Config(  # [config4]
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "combine_deposition_types": True,
                "integrate": True,
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
                    "nageclass": 0,
                    "noutrel": 0,
                    "numpoint": 0,
                    "species_id": 1,
                    "time": [0, 3, 9],
                },
            },
        ),
        Config(  # [config5]
            var_names_ref=["spec001"],
            setup_dct={
                "infile": datafilename2,
                "outfile": "dummy.png",
                "model": "COSMO-1",
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
        Config(  # [config6]
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "infile": datafilename2,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "integrate": True,
                "dimensions": {
                    "deposition_type": ["wet", "dry"],
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
def test_multiple(datadir, config):  # noqa:F811
    """Read multiple fields."""

    datafile = f"{datadir}/{config.setup_dct['infile']}"

    # Create setups
    setup_lst = list(config.setup.decompress_partially(None))
    for setup in setup_lst.copy():
        setup_lst.extend(setup.derive(config.derived_setup_params))
    setups = SetupCollection(setup_lst)

    # Process field specifications one after another
    var_setups: SetupCollection
    for var_setups in setups.decompress_twice(
        "dimensions.time", skip=["ens_member_id"]
    ):

        # Read input fields
        var_setups_dicts_pre = var_setups.dicts()
        field_lst_lst = read_fields(datafile, var_setups)
        assert var_setups.dicts() == var_setups_dicts_pre
        assert len(field_lst_lst) == 1
        assert len(field_lst_lst[0]) == 1
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
                    get_var_name_ref(var_setup, config.var_names_ref),
                    var_setup,
                    config.setup_dct["model"],
                )
            ]
            fld_ref_i = np.nansum(flds_ref_i, axis=0)
            if fld_ref is None:
                fld_ref = fld_ref_i
            else:
                fld_ref += fld_ref_i
        fld_ref *= config.scale_fld_ref

        assert fld.shape == fld_ref.shape
        assert np.isclose(np.nanmean(fld), np.nanmean(fld_ref), rtol=1e-6)
        np.testing.assert_allclose(fld, fld_ref, equal_nan=True, rtol=1e-6)


# test_single_add_ts0
@pytest.mark.parametrize(
    "config",
    [
        Config(
            var_names_ref=["spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
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
        Config(
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "infile": datafilename1,
                "outfile": "dummy.png",
                "model": "COSMO-1",
                "input_variable": "deposition",
                "combine_deposition_types": True,
                "integrate": False,
                "dimensions": {
                    "deposition_type": ["dry", "wet"],
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
def test_single_add_ts0(datadir, config):
    """Insert an additional time step 0 in the beginning, with empty fields."""
    datafile = f"{datadir}/{config.setup_dct['infile']}"

    # Initialize field specifications
    setups = SetupCollection([config.setup])

    # Read fields with and without added time step 0
    # Note: Check relies on ordered time steps, which is incidental
    reader_raw = FieldInputOrganizer(datafile, add_ts0=False)
    reader_ts0 = FieldInputOrganizer(datafile, add_ts0=True)
    field_raw_lst_lst = reader_raw.run(setups)
    field_ts0_lst_lst = reader_ts0.run(setups)
    assert len(field_ts0_lst_lst) == len(field_raw_lst_lst) + 1
    assert all(len(field_lst) == 1 for field_lst in field_ts0_lst_lst)
    assert all(len(field_lst) == 1 for field_lst in field_raw_lst_lst)
    assert (field_ts0_lst_lst[0][0].fld == 0.0).all()
    assert (field_ts0_lst_lst[1][0].fld == field_raw_lst_lst[0][0].fld).all()


def test_missing_deposition(datadir):  # noqa:F811
    """Read deposition field from file that does not contain one."""

    setup_dct = {
        "infile": datafilename4,
        "outfile": "dummy.png",
        "model": "COSMO-1E",
        "input_variable": "deposition",
        "integrate": True,
        "combine_deposition_types": True,
        "dimensions": {
            "deposition_type": ("dry", "wet"),
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": 1,
            "time": 10,
        },
    }
    setup = Setup.create(setup_dct)

    datafile = f"{datadir}/{setup_dct['infile']}"

    # Initialize field specifications
    setups = SetupCollection([setup])

    # Read input field
    field_lst_lst = read_fields(datafile, setups, missing_ok=True)
    assert len(field_lst_lst) == 1
    assert len(field_lst_lst[0]) == 1
    fld = field_lst_lst[0][0].fld

    assert (fld == 0.0).all()


def test_affected_area(datadir):  # noqa:F811
    """Read affected area field, combining concentration and deposition."""

    setup_dct = {
        "infile": datafilename4,
        "outfile": "dummy.png",
        "model": "COSMO-1E",
        "input_variable": "affected_area",
        "integrate": True,
        "combine_deposition_types": True,
        "dimensions": {
            "deposition_type": ("dry", "wet"),
            "level": 0,
            "nageclass": 0,
            "noutrel": 0,
            "numpoint": 0,
            "species_id": 1,
            "time": -1,
        },
    }
    setup = Setup.create(setup_dct)

    datafile = f"{datadir}/{setup_dct['infile']}"

    # Initialize field specifications
    setups = SetupCollection([setup])

    # Read input field
    field_lst_lst = read_fields(datafile, setups, missing_ok=True)
    assert len(field_lst_lst) == 1
    assert len(field_lst_lst[0]) == 1
    fld = field_lst_lst[0][0].fld

    assert ((fld == 0) | (fld == 1)).all()