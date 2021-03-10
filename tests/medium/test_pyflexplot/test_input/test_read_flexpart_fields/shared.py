"""Utilities for tests for module ``pyflexplot.input``."""
# Standard library
from typing import Collection
from typing import List
from typing import Optional
from typing import Union

# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from pyflexplot.setups.dimensions import Dimensions
from pyflexplot.setups.plot_setup import PlotSetupGroup

# Local
from ..shared import datadir_flexpart_artificial as datadir_artificial  # noqa:F401
from ..shared import datadir_flexpart_reduced as datadir_reduced  # noqa:F401


# SR_TMP <<< TODO eliminate!!
def decompress_twice(
    setups: PlotSetupGroup, outer: str, skip: Optional[Collection[str]] = None
) -> List[PlotSetupGroup]:
    skip = list(skip or []) + ["model.ens_member_id"]
    sub_setups_lst: List[PlotSetupGroup] = []
    for setup in setups:
        for sub_setup in setup.decompress([outer], skip):
            sub_sub_setups = sub_setup.decompress(skip=skip)
            sub_setups_lst.append(sub_sub_setups)
    return sub_setups_lst


def read_flexpart_field(
    path: str,
    var_name: str,
    dimensions: Dimensions,
    *,
    integrate: bool,
    model: str,
    add_ts0: bool,
) -> np.ndarray:
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        idcs: List[Union[slice, int]] = []
        idx: Union[slice, int]
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon", "latitude", "longitude"]:
                idx = slice(None)
            elif dim_name == "time":
                assert isinstance(dimensions.time, int)
                idx = slice(dimensions.time + (0 if add_ts0 else 1))
            elif dim_name in ["level", "height"]:
                assert isinstance(dimensions.level, int)
                idx = dimensions.level
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
        if dimensions.variable == "concentration":
            if integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif dimensions.variable.endswith("_deposition"):
            if not integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(f"variable '{dimensions.variable}'")
        fld = fld[-1]

        return fld


def fix_flexpart_field(fld, model):
    """Fix field read directly from NetCDF file."""
    if model.startswith(("COSMO-", "IFS-")):
        fld[:] *= 1.0e-12
    else:
        raise NotImplementedError("model", model)
