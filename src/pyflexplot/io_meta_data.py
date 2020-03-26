# -*- coding: utf-8 -*-
"""
Input file meta data.
"""
# Standard library
import re

# Third-party
import netCDF4 as nc4


def read_meta_data(file_handle: nc4.Dataset):
    """Read meta data (variables, dimensions, attributes) from a NetCDF file.

    Args:
        file_handle: Open NetCDF file handle.

    """
    # Global NetCDF attributes
    attrs_select = ["dxout", "dyout"]
    ncattrs = {
        attr: file_handle.getncattr(attr)
        for attr in file_handle.ncattrs()
        if attr in attrs_select
    }

    # Dimensions
    dimensions = {}
    for dim_handle in file_handle.dimensions.values():
        dimensions[dim_handle.name] = {
            "name": dim_handle.name,
            "size": dim_handle.size,
        }

    # Variables
    variables = {}
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

    # Analyze the file
    model = determine_model(ncattrs)
    species_ids = determine_species_ids(model, variables)
    analysis = {
        "model": model,
        "rotated_pole": model.startswith("cosmo"),
        "species_ids": species_ids,
    }

    meta_data = {
        "ncattrs": ncattrs,
        "dimensions": dimensions,
        "variables": variables,
        "analysis": analysis,
    }
    return meta_data


def determine_model(ncattrs):
    """Determine the model from global NetCDF attributes.

    For lack of an explicit model type attribute, use the grid resolution as a
    proxy.

    """
    dxout = ncattrs["dxout"]
    choices = {
        type(dxout)(0.25): "ifs",
        type(dxout)(0.02): "cosmo2",
        type(dxout)(0.01): "cosmo1",
    }
    try:
        return choices[dxout]
    except KeyError:
        raise Exception("no model defined for dxout", dxout, choices)


def determine_species_ids(model, variables):
    """Determine the species ids from the variables."""
    if model in ["cosmo1", "cosmo2", "ifs"]:
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
