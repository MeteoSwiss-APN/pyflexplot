#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
# Third-party
import netCDF4 as nc4
import numpy as np


# SR_TMP <<<
def fix_nc_fld(fld):
    """Fix field read directly from NetCDF file."""
    fld[:] *= 1e-12


def read_nc_var(path, var_name, var_specs):
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        inds = []
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon"]:
                ind = slice(*getattr(var_specs, dim_name, [None]))
            elif dim_name == "time":
                # Read all timesteps until the selected one
                ind = slice(getattr(var_specs, dim_name) + 1)
            else:
                ind = getattr(var_specs, dim_name, slice(None))
            inds.append(ind)

        # Read field
        fld = var[inds]
        assert len(fld.shape) == 3

        # Fix some issues with the input data
        fix_nc_fld(fld)  # SR_TMP

        # Reduce time dimension
        if var_specs.issubcls("concentration"):
            if var_specs.integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif var_specs.issubcls("deposition"):
            if not var_specs.integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(f"var specs of type '{type(var_specs).__name__}'")
        fld = fld[-1]

        return fld
