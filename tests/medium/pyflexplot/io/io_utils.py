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


def read_nc_var(path, var_name, setup, var_specs_dct):
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # print()
        # print(setup)
        # print(var_specs_dct)
        # print()

        assert var_specs_dct["rlat"] == (None,), f"{var_specs_dct['rlat']}"
        assert var_specs_dct["rlon"] == (None,), f"{var_specs_dct['rlon']}"
        # assert len(setup.time_idcs) == 1, \
        #     (f"{setup.time_idcs} != {var_specs_dct['time']}"
        if setup.variable == "concentration":
            assert (
                var_specs_dct["level"] == setup.level_idx
            ), f"### {var_specs_dct['level']} != {setup.level_idx}"
        var_specs_dct = {
            "rlat": (None,),
            "rlon": (None,),
            "species_id": setup.species_id,
            "integrate": setup.integrate,
            "deposition": setup.deposition_type,
            "level": var_specs_dct.get("level"),
            # "level": setup.level_idx,
            "time": var_specs_dct["time"],
            # "time": #next(iter(setup.time_idcs)),
            "nageclass": 0,  # SR_TMP
            "numpoint": 0,  # SR_TMP
            "noutrel": 0,  # SR_TMP
        }

        # Collect dimension indices
        idcs = []
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon"]:
                idx = slice(*var_specs_dct.get(dim_name, [None]))
            elif dim_name == "time":
                # Read all timesteps until the selected one
                idx = slice(var_specs_dct.get(dim_name) + 1)
            else:
                idx = var_specs_dct.get(dim_name, slice(None))
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
