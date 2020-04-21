# -*- coding: utf-8 -*-
"""
Crop fields in a NetCDF file.
"""
# Standard library
import dataclasses
import functools
import sys
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

# Third-party
import click
import netCDF4 as nc4

__version__ = "0.1.0"


@dataclass
class Setup:
    lat_name: str
    lon_name: str
    lat_slice: Tuple[Optional[int], Optional[int], Optional[int]]
    lon_slice: Tuple[Optional[int], Optional[int], Optional[int]]

    def dict(self):
        return dataclasses.asdict(self)


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)  # type: ignore

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"],)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "in_file_path", metavar="infile", nargs=1, type=click.Path(exists=True),
)
@click.argument(
    "out_file_path", metavar="outfile", nargs=1, type=click.Path(),
)
@click.option(
    "--lat",
    "lat_slice",
    help="Latitude index slice arguments. STOP is exclusive.",
    type=(int, int, int),
    metavar="START STOP STEP",
    default=(None, None, None),
)
@click.option(
    "--lon",
    "lon_slice",
    help="Longitude index slice arguments. STOP is exclusive.",
    type=(int, int, int),
    metavar="START STOP STEP",
    default=(None, None, None),
)
@click.option(
    "--lat-name", help="Name of latitude dimension.", default="lat",
)
@click.option(
    "--lon-name", help="Name of longitude dimension.", default="lon",
)
def main(in_file_path, out_file_path, **kwargs_setup):
    setup = Setup(**kwargs_setup)

    for name, value in {
        "in_file_path": in_file_path,
        "out_file_path": out_file_path,
        **setup.dict(),
    }.items():
        click.echo(f"{name:<15} : {value}")

    with nc4.Dataset(in_file_path, "r") as fi, nc4.Dataset(out_file_path, "w") as fo:
        transfer_ncattrs(fi, fo)
        transfer_dimensions(fi, fo, setup)
        transfer_variables(fi, fo, setup)


def len_slice(s):
    """Count number of elements selected by slice."""
    return len(range(s.start, s.stop, s.step))


def transfer_ncattrs(fi, fo):
    """Transfer global attributes from in- to outfile."""
    ncattrs = {ncattr: fi.getncattr(ncattr) for ncattr in fi.ncattrs()}
    fo.setncatts(ncattrs)


def transfer_dimensions(fi, fo, setup):
    """Transfer all dimensions from in- to outfile."""
    for dim in fi.dimensions.values():
        transfer_dimension(fi, fo, dim, setup)


def transfer_dimension(fi, fo, dim, setup):
    """Transfer single dimension from in- to outfile."""

    # Determine dimension size
    if dim.isunlimited():
        size = None
    elif dim.name == setup.lat_name:
        size = len_slice(setup.lat_slice)
    elif dim.name == setup.lon_name:
        size = len_slice(setup.lon_slice)
    else:
        size = dim.size

    # Create dimension
    fo.createDimension(dim.name, size)

    # Transfer corresponding variable
    try:
        var = fi.variables[dim.name]
    except KeyError:
        # There's none; that's fine
        pass
    else:
        transfer_variable(fo, var, setup)


def transfer_variables(fi, fo, setup):
    """Transfer all regular variables from in- to outfile."""
    for var in fi.variables.values():
        if var.name not in fi.dimensions:
            transfer_variable(fo, var, setup)


def transfer_variable(fo, var, setup):
    """Transfer single variable from in- to outfile."""

    # Create variable
    new_var = fo.createVariable(
        varname=var.name, datatype=var.datatype, dimensions=var.dimensions,
    )

    # Transfer data, slicing along lat/lon
    if var.dimensions:
        inds = []
        for dim_name in var.dimensions:
            if dim_name == setup.lat_name:
                inds.append(slice(setup.lat_slice))
            elif dim_name == setup.lon_name:
                inds.append(slice(setup.lon_slice))
            else:
                inds.append(slice(None))
        new_var[:] = var[inds]

    # Transfer variable attributes
    transfer_ncattrs(var, new_var)


if __name__ == "__main__":
    sys.exit(1)  # pragma: no cover
