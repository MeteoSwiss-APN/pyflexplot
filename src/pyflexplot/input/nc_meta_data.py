"""Input file meta data.

Note that these raw meta data should eventually be merged with those in module
``pyflexplot.meta_data`` because the two very different data structures serve
very similar purposes in parallel.

"""
# Standard library
import re
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple

# Third-party
import netCDF4 as nc4

# Local
from ..utils.datetime import init_datetime


def read_nc_meta_data(file_handle: nc4.Dataset) -> Dict[str, Dict[str, Any]]:
    """Read meta data (variables, dimensions, attributes) from a NetCDF file.

    Args:
        file_handle: Open NetCDF file handle.

    """
    # Select global NetCDF attributes
    attrs_select: List[str] = [
        # "dxout",
        # "dyout",
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
    derived: Dict[str, Any] = {
        "release_site": determine_release_site(file_handle),
        "rotated_pole": determine_rotated_pole(dimensions),
        "species_ids": derive_species_ids(variables.keys()),
        "time_steps": collect_time_steps(ncattrs),
    }

    return {
        "ncattrs": ncattrs,
        "dimensions": dimensions,
        "variables": variables,
        "derived": derived,
    }


def determine_release_site(file_handle: nc4.Dataset) -> str:
    var = file_handle.variables["RELCOM"]
    if len(var) == 0:
        raise Exception("no release sites")
    elif len(var) == 1:
        idx = 0
    else:
        raise NotImplementedError("multiple release sites")
    return var[idx][~var[idx].mask].tobytes().decode("utf-8").rstrip()


def determine_rotated_pole(dimensions: Dict[str, Any]) -> bool:
    if "rlat" in dimensions and "rlon" in dimensions:
        return True
    elif "latitude" in dimensions and "longitude" in dimensions:
        return False
    else:
        raise NotImplementedError("unexpected dimensions: {list(dimensions)}")


def collect_time_steps(ncattrs: Dict[str, Any]) -> Tuple[int, ...]:
    fmt_out = "%Y%m%d%H%M"
    start_date: str = ncattrs["ibdate"]
    start_time: str = ncattrs["ibtime"]
    assert len(start_time) == 6
    start: datetime = init_datetime(start_date + start_time)
    end_date: str = ncattrs["iedate"]
    end_time: str = ncattrs["ietime"]
    assert len(end_time) == 6
    end: datetime = init_datetime(end_date + end_time)
    assert end > start
    delta: timedelta = timedelta(seconds=int(ncattrs["loutstep"]))
    time_steps: List[datetime] = [start]
    while time_steps[-1] < end:
        time_steps.append(time_steps[-1] + delta)
    return tuple(map(lambda ts: int(ts.strftime(fmt_out)), time_steps))


def derive_species_ids(variable_names: Collection[str]) -> Tuple[int, ...]:
    """Derive the species ids from the NetCDF variable names."""
    rx = re.compile(r"\A([WD]D_)?spec(?P<species_id>[0-9][0-9][0-9])(_mr)?\Z")
    species_ids = set()
    for var_name in variable_names:
        match = rx.match(var_name)
        if match:
            species_id = int(match.group("species_id"))
            species_ids.add(species_id)
    if not species_ids:
        raise Exception("could not identify species ids")
    return tuple(sorted(species_ids))


def derive_variable_name(
    model: str, input_variable: str, species_id: int, deposition_type: str
) -> str:
    """Derive the NetCDF variable name given some attributes."""
    cosmo_models = ["COSMO-2", "COSMO-1", "COSMO-2E", "COSMO-1E"]
    ifs_models = ["IFS-HRES", "IFS-HRES-EU"]
    if input_variable == "concentration":
        if model in cosmo_models:
            return f"spec{species_id:03d}"
        elif model in ifs_models:
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif input_variable == "deposition":
        prefix = {"wet": "WD", "dry": "DD"}[deposition_type]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError("unknown variable", input_variable)