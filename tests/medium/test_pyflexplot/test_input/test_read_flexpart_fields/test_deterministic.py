"""Tests for module ``pyflexplot.input.read_fields``.

These tests use deterministic data.

"""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Third-party
import numpy as np
import pytest  # type: ignore

# First-party
from pyflexplot.input.read_fields import read_fields
from pyflexplot.setups.dimensions import Dimensions
from pyflexplot.setups.plot_setup import PlotSetup
from pyflexplot.setups.plot_setup import PlotSetupGroup

# Local
from .shared import datadir_reduced as datadir  # noqa:F401
from .shared import decompress_twice
from .shared import read_flexpart_field


def get_var_name_ref(dimensions: Dimensions, var_names_ref: str):
    variable = dimensions.variable
    assert isinstance(variable, str)
    if variable == "concentration":
        assert len(var_names_ref) == 1
        return next(iter(var_names_ref))
    elif variable.endswith("_deposition"):
        species_id = dimensions.species_id
        if isinstance(species_id, tuple):
            assert len(species_id) == 1
            species_id = next(iter(species_id))
        var_name = f"{variable[0].upper()}D_spec{species_id:03d}"
        if var_name in var_names_ref:
            return var_name
    raise NotImplementedError(f"dimensions={dimensions}\nvar_names_ref={var_names_ref}")


@dc.dataclass
class Config:
    var_names_ref: List[str]
    setup_dct: Dict[str, Any]
    n_panels: int = 1
    derived_setup_params: List[Dict[str, Any]] = dc.field(default_factory=list)
    scale_fld_ref: Optional[float] = 1.0

    @property
    def setup(self):
        return PlotSetup.create(self.setup_dct)


datafilename1 = "flexpart_cosmo-1_2019052800.nc"
datafilename2 = "flexpart_cosmo-1_2019093012.nc"
datafilename3 = "flexpart_ifs-hres_20200317000000.nc"
datafilename4 = "flexpart_cosmo-1e-ctrl_2020102105.nc"
datafilename5 = "flexpart_ifs-hres-eu_1023_20201113120000.nc"


# test_single
@pytest.mark.parametrize(
    "config",
    [
        Config(  # [config0]
            var_names_ref=["spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": False,
                        "dimensions": {
                            "level": 1,
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": 3,
                        },
                    }
                ],
            },
        ),
        Config(  # [config1]
            var_names_ref=["DD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "dry_deposition",
                        "integrate": False,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": 3,
                            "variable": "dry_deposition",
                        },
                    }
                ],
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config2]
            var_names_ref=["WD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "wet_deposition",
                        "integrate": False,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": 3,
                            "variable": "wet_deposition",
                        },
                    }
                ],
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config3]
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "integrate": False,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": 3,
                            "variable": ["dry_deposition", "wet_deposition"],
                        },
                    }
                ],
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config4]
            var_names_ref=["spec001"],
            setup_dct={
                "files": {
                    "input": datafilename2,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": False,
                        "dimensions": {
                            "level": 1,
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": 3,
                        },
                    }
                ],
            },
        ),
        Config(  # [config5]
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "files": {
                    "input": datafilename2,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "integrate": False,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": 3,
                            "variable": ["dry_deposition", "wet_deposition"],
                        },
                    }
                ],
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config6]
            var_names_ref=["DD_spec001", "DD_spec002", "WD_spec001", "WD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "combine_species": True,
                        "integrate": False,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": [1, 2],
                            "time": 3,
                            "variable": ["dry_deposition", "wet_deposition"],
                        },
                    }
                ],
            },
            scale_fld_ref=1 / 3,
        ),
        Config(  # [config7]
            var_names_ref=["spec001_mr"],
            setup_dct={
                "files": {
                    "input": datafilename3,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "IFS-HRES",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": False,
                        "dimensions": {
                            "level": 1,
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": 10,
                        },
                    }
                ],
            },
        ),
        Config(  # [config8]
            var_names_ref=["spec001"],
            setup_dct={
                "files": {
                    "input": datafilename4,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1E",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": False,
                        "dimensions": {
                            "level": 0,
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": 10,
                        },
                    }
                ],
            },
        ),
    ],
)
def test_single(datadir, config):  # noqa:F811
    """Read a single field."""

    datafile = f"{datadir}/{config.setup_dct['files']['input']}"

    # Initialize field specifications
    setups = PlotSetupGroup([config.setup])

    # Read input field
    field_groups = read_fields(setups, {"add_ts0": True}, _override_infile=datafile)
    assert len(field_groups) == 1
    assert len(field_groups[0]) == 1
    fld = next(iter(field_groups[0])).fld

    # Initialize individual setup objects
    var_setups_lst = decompress_twice(
        setups, "dimensions.time", skip=["model.ens_member_id"]
    )
    assert len(var_setups_lst) == 1
    var_setups = next(iter(var_setups_lst))

    # Read reference field
    fld_ref = (
        np.nansum(
            [
                read_flexpart_field(
                    datafile,
                    get_var_name_ref(sub_dimensions, config.var_names_ref),
                    sub_dimensions,
                    integrate=panel.integrate,
                    model=config.setup_dct["model"]["name"],
                    add_ts0=True,
                )
                for setup in var_setups
                for panel in setup.panels
                for sub_dimensions in panel.dimensions.decompress()
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
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": True,
                        "combine_levels": True,
                        "dimensions": {
                            "level": [0, 2],
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": [0, 3],
                        },
                    }
                ],
            },
            scale_fld_ref=3.0,
        ),
        Config(  # [config1]
            var_names_ref=["spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": True,
                        "combine_levels": True,
                        "dimensions": {
                            "level": [0, 2],
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": [0, 3],
                        },
                    }
                ],
            },
            scale_fld_ref=3.0,
        ),
        Config(  # [config2]
            var_names_ref=["DD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "dry_deposition",
                        "integrate": True,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": [0, 3, 9],
                            "variable": "dry_deposition",
                        },
                    }
                ],
            },
        ),
        Config(  # [config3]
            var_names_ref=["WD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "wet_deposition",
                        "integrate": True,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": [0, 3, 9],
                            "variable": "wet_deposition",
                        },
                    }
                ],
            },
        ),
        Config(  # [config4]
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "integrate": True,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": [0, 3, 9],
                            "variable": ["dry_deposition", "wet_deposition"],
                        },
                    }
                ],
            },
        ),
        Config(  # [config5]
            var_names_ref=["spec001"],
            setup_dct={
                "files": {
                    "input": datafilename2,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": True,
                        "dimensions": {
                            "level": 0,
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": [0, 3, 9],
                        },
                    }
                ],
            },
            derived_setup_params=[{"panels": [{"dimensions": {"level": 2}}]}],
            scale_fld_ref=3.0,
        ),
        Config(  # [config6]
            var_names_ref=["WD_spec001", "DD_spec001"],
            setup_dct={
                "files": {
                    "input": datafilename2,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "integrate": True,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 1,
                            "time": [0, 3, 9],
                            "variable": ["wet_deposition", "dry_deposition"],
                        },
                    }
                ],
            },
        ),
        Config(  # [config7]
            var_names_ref=["spec001"],
            setup_dct={
                "files": {
                    "input": datafilename4,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1E",
                },
                "layout": {
                    "plot_type": "multipanel",
                    "multipanel_param": "time",
                },
                "panels": {
                    "plot_variable": "concentration",
                    "integrate": False,
                    "dimensions": {
                        "level": 0,
                        "nageclass": 0,
                        "release": 0,
                        "species_id": 1,
                        "time": [2, 4, 6, 8],
                    },
                },
            },
            n_panels=4,
        ),
    ],
)
def test_multiple(datadir, config):  # noqa:F811
    """Read multiple fields."""

    datafile = f"{datadir}/{config.setup_dct['files']['input']}"

    # Create setups
    setup_lst = list(config.setup.decompress(internal=False))
    for setup in setup_lst.copy():
        setup_lst.extend(setup.derive(config.derived_setup_params))
    setups = PlotSetupGroup(setup_lst)

    # Process field specifications one after another
    var_setups: PlotSetupGroup
    for var_setups in decompress_twice(
        setups, "dimensions.time", skip=["model.ens_member_id"]
    ):

        # Read input fields
        var_setups_dicts_pre = var_setups.dicts()
        field_groups = read_fields(
            var_setups, {"add_ts0": False}, _override_infile=datafile
        )
        assert var_setups.dicts() == var_setups_dicts_pre
        assert len(field_groups) == 1
        assert len(field_groups[0]) == config.n_panels
        fld = np.array(
            [field.fld for field_group in field_groups for field in field_group]
        )
        assert fld.shape[0] == config.n_panels

        # Read reference fields
        fld_ref_lst = []
        for var_setup in var_setups:
            for panel in var_setup.panels:
                fld_ref_lst.append(None)
                for dimensions in panel.dimensions.decompress():
                    flds_ref_i = [
                        read_flexpart_field(
                            datafile,
                            get_var_name_ref(dimensions, config.var_names_ref),
                            dimensions,
                            integrate=panel.integrate,
                            model=config.setup_dct["model"]["name"],
                            add_ts0=False,
                        )
                    ]
                    fld_ref_i = np.nansum(flds_ref_i, axis=0)
                    if fld_ref_lst[-1] is None:
                        fld_ref_lst[-1] = fld_ref_i
                    else:
                        fld_ref_lst[-1] += fld_ref_i
        fld_ref = np.array(fld_ref_lst)
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
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "concentration",
                        "integrate": False,
                        "dimensions": {
                            "level": 1,
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": None,
                        },
                    }
                ],
            },
        ),
        Config(
            var_names_ref=["WD_spec002", "DD_spec002"],
            setup_dct={
                "files": {
                    "input": datafilename1,
                    "output": "dummy.png",
                },
                "model": {
                    "name": "COSMO-1",
                },
                "panels": [
                    {
                        "plot_variable": "tot_deposition",
                        "integrate": False,
                        "dimensions": {
                            "nageclass": 0,
                            "release": 0,
                            "species_id": 2,
                            "time": None,
                            "variable": ["dry_deposition", "wet_deposition"],
                        },
                    }
                ],
            },
            scale_fld_ref=1 / 3,
        ),
    ],
)
def test_single_add_ts0(datadir, config):
    """Insert an additional time step 0 in the beginning, with empty fields."""
    datafile = f"{datadir}/{config.setup_dct['files']['input']}"

    # Initialize field specifications
    setups = PlotSetupGroup([config.setup])

    # Read fields with and without added time step 0
    # Note: Check relies on ordered time steps, which is incidental
    field_groups_raw = read_fields(
        setups, {"add_ts0": False}, _override_infile=datafile
    )
    field_groups_ts0 = read_fields(setups, {"add_ts0": True}, _override_infile=datafile)
    assert len(field_groups_ts0) == len(field_groups_raw) + 1
    assert all(len(field_group) == 1 for field_group in field_groups_ts0)
    assert all(len(field_group) == 1 for field_group in field_groups_raw)
    flds_ts0 = [next(iter(field_group)).fld for field_group in field_groups_ts0]
    fld_raw = next(iter(field_groups_raw[0])).fld
    assert (flds_ts0[0] == 0.0).all()
    assert (flds_ts0[1] == fld_raw).all()


def test_missing_deposition_cosmo(datadir):  # noqa:F811
    """Read deposition field from file that does not contain one."""
    setup_dct = {
        "files": {
            "input": datafilename4,
            "output": "dummy.png",
        },
        "model": {
            "name": "COSMO-1E",
        },
        "panels": [
            {
                "plot_variable": "tot_deposition",
                "integrate": True,
                "dimensions": {
                    "nageclass": 0,
                    "release": 0,
                    "species_id": 1,
                    "time": 10,
                    "variable": ("dry_deposition", "wet_deposition"),
                },
            }
        ],
    }
    setup = PlotSetup.create(setup_dct)
    datafile = f"{datadir}/{setup_dct['files']['input']}"

    # Initialize field specifications
    setups = PlotSetupGroup([setup])

    # Read input field
    field_groups = read_fields(setups, {"missing_ok": True}, _override_infile=datafile)
    assert len(field_groups) == 1
    assert len(field_groups[0]) == 1
    fld = next(iter(field_groups[0])).fld

    assert (fld == 0.0).all()


def test_missing_deposition_ifs(datadir):  # noqa:F811
    """Read deposition field from file that does not contain one."""
    setup_dct = {
        "files": {
            "input": datafilename5,
            "output": "dummy.png",
        },
        "model": {
            "name": "IFS-HRES-EU",
        },
        "panels": [
            {
                "plot_variable": "tot_deposition",
                "integrate": True,
                "dimensions": {
                    "nageclass": 0,
                    "release": 0,
                    "species_id": 1,
                    "time": 10,
                    "variable": ("dry_deposition", "wet_deposition"),
                },
            }
        ],
    }
    setup = PlotSetup.create(setup_dct)
    datafile = f"{datadir}/{setup_dct['files']['input']}"

    # Initialize field specifications
    setups = PlotSetupGroup([setup])

    # Read input field
    field_groups = read_fields(setups, {"missing_ok": True}, _override_infile=datafile)
    assert len(field_groups) == 1
    assert len(field_groups[0]) == 1
    fld = next(iter(field_groups[0])).fld

    assert (fld == 0.0).all()


def test_affected_area(datadir):  # noqa:F811
    """Read affected area field, combining concentration and deposition."""
    setup_dct = {
        "files": {
            "input": datafilename4,
            "output": "dummy.png",
        },
        "model": {
            "name": "COSMO-1E",
        },
        "panels": [
            {
                "plot_variable": "affected_area",
                "integrate": True,
                "dimensions": {
                    "level": 0,
                    "nageclass": 0,
                    "release": 0,
                    "species_id": 1,
                    "time": -1,
                    "variable": ("concentration", "dry_deposition", "wet_deposition"),
                },
            }
        ],
    }
    setup = PlotSetup.create(setup_dct)
    datafile = f"{datadir}/{setup_dct['files']['input']}"

    # Initialize field specifications
    setups = PlotSetupGroup([setup])

    # Read input field
    field_groups = read_fields(setups, {"missing_ok": True}, _override_infile=datafile)
    assert len(field_groups) == 1
    assert len(field_groups[0]) == 1
    fld = next(iter(field_groups[0])).fld

    assert ((fld == 0) | (fld == 1)).all()
