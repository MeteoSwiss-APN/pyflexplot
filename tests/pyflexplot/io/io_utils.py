#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.io``."""
import netCDF4 as nc4
import numpy as np

from pyflexplot.io import FieldSpecs


# SR_TMP <<<
def fix_nc_fld(fld):
    """Fix field read directly from NetCDF file."""
    fld[:] *= 1e-12


def read_nc_var(path, var_name, var_specs):
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        inds = []
        for name in var.dimensions:
            if name in ["rlat", "rlon"]:
                ind = slice(*getattr(var_specs, name, [None]))
            elif name == "time":
                # Read all timesteps until the selected one
                ind = slice(getattr(var_specs, name) + 1)
            else:
                ind = getattr(var_specs, name, slice(None))
            inds.append(ind)

        # Read field
        fld = var[inds]
        assert len(fld.shape) == 3

        # Fix some issues with the input data
        fix_nc_fld(fld)  # SR_TMP

        # Reduce time dimension
        if isinstance(var_specs, FieldSpecs.subclass("concentration").cls_var_specs):
            if var_specs.integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif isinstance(var_specs, FieldSpecs.subclass("deposition").cls_var_specs):
            if not var_specs.integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(f"var specs of type '{type(var_specs).__name__}'")
        fld = fld[-1]

        return fld
