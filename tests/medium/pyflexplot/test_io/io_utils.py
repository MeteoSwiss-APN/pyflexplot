#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.io``.
"""
# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from pyflexplot.setup import CoreSetup


# SR_TMP <<<
def fix_nc_fld(fld, model):
    """Fix field read directly from NetCDF file."""
    if model in ["cosmo1", "cosmo2"]:
        fld[:] *= 1e-12
    elif model == "ifs":
        pass
    else:
        raise NotImplementedError("model", model)


def read_nc_var(path, var_name, setup, model):
    # + assert isinstance(setup, CoreSetup)  # SR_TODO (if that's indeed the goal)
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        idcs = []
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon", "latitude", "longitude"]:
                idx = slice(None)
            elif dim_name == "time":
                # Read all timesteps until the selected one
                # SR_TMP <
                if isinstance(setup, CoreSetup):
                    idx = slice(setup.time + 1)
                else:
                    assert len(setup.time) == 1
                    idx = slice(setup.time[0] + 1)
                # SR_TMP >
            elif dim_name in ["level", "height"]:
                # SR_TMP <
                if isinstance(setup, CoreSetup) or setup.level is None:
                    idx = setup.level
                else:
                    assert len(setup.level) == 1
                    idx = next(iter(setup.level))
                # SR_TMP >
            elif dim_name in ["nageclass", "numpoint", "noutrel", "pointspec"]:
                idx = 0  # SR_HC
            else:
                raise NotImplementedError(f"dimension '{dim_name}'")
            idcs.append(idx)

        # Read field
        fld = var[idcs]
        assert len(fld.shape) == 3

        # Fix some issues with the input data
        fix_nc_fld(fld, model)  # SR_TMP

        # Reduce time dimension
        if setup.variable == "concentration":
            if setup.integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif setup.variable == "deposition":
            if not setup.integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(f"variable '{setup.variable}'")
        fld = fld[-1]

        return fld
