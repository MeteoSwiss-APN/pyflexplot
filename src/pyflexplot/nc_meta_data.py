# -*- coding: utf-8 -*-
"""
Input file meta data.
"""
# Standard library
import re
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# Third-party
import netCDF4 as nc4


def read_meta_data(file_handle: nc4.Dataset) -> Dict[str, Dict[str, Any]]:
    """Read meta data (variables, dimensions, attributes) from a NetCDF file.

    Args:
        file_handle: Open NetCDF file handle.

    """
    # Select global NetCDF attributes
    attrs_select: List[str] = [
        "dxout",
        "dyout",
        "ibdate",
        "ibtime",
        "iedate",
        "ietime",
        "loutstep",
    ]
    attrs_try_select: List[str] = []
    ncattrs: Dict[str, Any] = {}
    for attr in attrs_select:
        ncattrs[attr] = file_handle.getncattr(attr)
    for attr in attrs_try_select:
        try:
            ncattrs[attr] = file_handle.getncattr(attr)
        except AttributeError:
            continue

    # Dimensions
    dimensions: Dict[str, Any] = {}
    for dim_handle in file_handle.dimensions.values():
        dimensions[dim_handle.name] = {
            "name": dim_handle.name,
            "size": dim_handle.size,
        }

    # Variables
    variables: Dict[str, Any] = {}
    for var_handle in file_handle.variables.values():
        variables[var_handle.name] = {
            "name": var_handle.name,
            "dimensions": var_handle.dimensions,
            "shape": tuple(
                [dimensions[dim_name]["size"] for dim_name in var_handle.dimensions]
            ),
            "ncattrs": {
                attr: var_handle.getncattr(attr) for attr in var_handle.ncattrs()
            },
        }

    # Derive some custom attributes
    model: str = determine_model(ncattrs)
    derived: Dict[str, Any] = {
        "model": model,
        "release_site": determine_release_site(file_handle),
        "rotated_pole": model.startswith("COSMO"),
        "species_ids": determine_species_ids(model, variables),
        "time_steps": determine_time_steps(ncattrs),
    }

    return {
        "ncattrs": ncattrs,
        "dimensions": dimensions,
        "variables": variables,
        "derived": derived,
    }


def determine_model(ncattrs: Dict[str, Any]) -> str:
    """Determine the model from global NetCDF attributes.

    For lack of an explicit model type attribute, use the grid resolution as a
    proxy.

    """
    dxout = ncattrs["dxout"]
    choices = {
        type(dxout)(0.25): "IFS",
        type(dxout)(0.10): "IFS-HRES",
        type(dxout)(0.02): "COSMO-2",
        type(dxout)(0.01): "COSMO-1",
    }
    try:
        return choices[dxout]
    except KeyError:
        raise Exception("no model defined for dxout", dxout, choices)


def determine_release_site(file_handle: nc4.Dataset) -> str:
    var = file_handle.variables["RELCOM"]
    # SR_TMP <
    assert len(var) == 1
    idx = 0
    # SR_TMP >
    return var[idx][~var[idx].mask].tostring().decode("utf-8").rstrip()


def determine_species_ids(model: str, variables: Dict[str, Any]) -> Tuple[int, ...]:
    """Determine the species ids from the variables."""
    if model in ["COSMO-1", "COSMO-2", "IFS", "IFS-HRES"]:
        rx = re.compile(r"\A[WD]D_spec(?P<species_id>[0-9][0-9][0-9])\Z")
        species_ids = set()
        for var_name in variables.keys():
            match = rx.match(var_name)
            if match:
                species_id = int(match.group("species_id"))
                species_ids.add(species_id)
        return tuple(sorted(species_ids))
    else:
        raise ValueError("unexpected model", model)


def determine_time_steps(ncattrs: Dict[str, Any]) -> Tuple[int, ...]:
    fmt_in = "%Y%m%d%H%M%S"
    fmt_out = "%Y%m%d%H%M"
    start_date: str = ncattrs["ibdate"]
    start_time: str = ncattrs["ibtime"]
    assert len(start_time) == 6
    start: datetime = datetime.strptime(start_date + start_time, fmt_in)
    end_date: str = ncattrs["iedate"]
    end_time: str = ncattrs["ietime"]
    assert len(end_time) == 6
    end: datetime = datetime.strptime(end_date + end_time, fmt_in)
    assert end > start
    delta: timedelta = timedelta(seconds=int(ncattrs["loutstep"]))
    time_steps: List[datetime] = [start]
    while time_steps[-1] < end:
        time_steps.append(time_steps[-1] + delta)
    return tuple(map(lambda ts: int(ts.strftime(fmt_out)), time_steps))
