"""Utilities for tests for module ``pyflexplot.input``."""

# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from pyflexplot.setup import CoreSetup

# Local
from ..shared import datadir_flexpart_artificial as datadir_artificial  # noqa:F401
from ..shared import datadir_flexpart_reduced as datadir_reduced  # noqa:F401


def read_flexpart_field(path, var_name, setup, *, model, add_ts0):
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        idcs = []
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon", "latitude", "longitude"]:
                idx = slice(None)
            elif dim_name == "time":
                # Read all timesteps until the selected one
                if isinstance(setup, CoreSetup):
                    idx = slice(setup.core.dimensions.time + (0 if add_ts0 else 1))
                else:
                    assert isinstance(setup.core.dimensions.time, int)
                    idx = slice(setup.core.dimensions.time + (0 if add_ts0 else 1))
            elif dim_name in ["level", "height"]:
                if isinstance(setup, CoreSetup) or setup.core.dimensions.level is None:
                    idx = setup.core.dimensions.level
                else:
                    assert isinstance(setup.core.dimensions.level, int)
                    idx = setup.core.dimensions.level
            elif dim_name in ["nageclass", "numpoint", "noutrel", "pointspec"]:
                idx = 0
            else:
                raise NotImplementedError(f"dimension '{dim_name}'")
            idcs.append(idx)

        # Read field
        fld = var[idcs]
        assert len(fld.shape) == 3

        # Fix some issues with the input data
        fix_flexpart_field(fld, model)

        # Handle time integration of data
        if setup.core.input_variable == "concentration":
            if setup.core.integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif setup.core.input_variable == "deposition":
            if not setup.core.integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(f"variable '{setup.core.input_variable}'")
        fld = fld[-1]

        return fld


def fix_flexpart_field(fld, model):
    """Fix field read directly from NetCDF file."""
    if model.startswith(("COSMO-", "IFS-")):
        fld[:] *= 1.0e-12
    else:
        raise NotImplementedError("model", model)
