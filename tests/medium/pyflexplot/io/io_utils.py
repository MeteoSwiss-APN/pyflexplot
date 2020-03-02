#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.io``.
"""
# Third-party
import netCDF4 as nc4
import numpy as np


# SR_TMP <<<
def fix_nc_fld(fld):
    """Fix field read directly from NetCDF file."""
    fld[:] *= 1e-12


def read_nc_var(path, var_name, setup):
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        idcs = []
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon"]:
                idx = slice(None)
            elif dim_name == "time":
                # Read all timesteps until the selected one
                assert len(setup.time_idcs) == 1
                idx = slice(setup.time_idcs[0] + 1)
            elif dim_name == "level":
                idx = setup.level_idx
            elif dim_name in ["nageclass", "numpoint", "noutrel"]:
                idx = 0  # SR_HC
            else:
                raise NotImplementedError(f"dimension '{dim_name}'")
            idcs.append(idx)

        # Read field
        fld = var[idcs]
        assert len(fld.shape) == 3

        # Fix some issues with the input data
        fix_nc_fld(fld)  # SR_TMP

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
