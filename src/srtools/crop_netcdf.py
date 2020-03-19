# -*- coding: utf-8 -*-
"""
Crop fields in a NetCDF file.
"""
# Standard library
import functools
import sys

# Third-party
import click
import netCDF4 as nc4

__version__ = "0.1.0"

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
    default=(0, -1, 1),
)
@click.option(
    "--lon",
    "lon_slice",
    help="Longitude index slice arguments. STOP is exclusive.",
    type=(int, int, int),
    metavar="START STOP STEP",
    default=(0, -1, 1),
)
@click.option(
    "--lat-name", help="Name of latitude dimension.", default="lat",
)
@click.option(
    "--lon-name", help="Name of longitude dimension.", default="lon",
)
def main(in_file_path, out_file_path, **conf):

    click.echo(f"infile    : {in_file_path}")
    click.echo(f"outfile   : {out_file_path}")
    click.echo(f"lat_name  : {conf['lat_name']}")
    click.echo(f"lon_name  : {conf['lon_name']}")
    click.echo(f"lat_slice : ({', '.join([str(f) for f in conf['lat_slice']])})")
    click.echo(f"lon_slice : ({', '.join([str(f) for f in conf['lon_slice']])})")

    with nc4.Dataset(in_file_path, "r") as fi, nc4.Dataset(out_file_path, "w") as fo:
        prepare_slices(fi, conf)
        transfer_ncattrs(fi, fo)
        transfer_dimensions(fi, fo, **conf)
        transfer_variables(fi, fo, **conf)


def prepare_slices(fi, conf):
    """Prepare lon/lat slices from input arguments."""

    def prepare_slice(name):
        start, stop, step = conf.pop(f"{name}_slice")
        n = len(fi.dimensions[conf[f"{name}_name"]])
        if stop == -1:
            stop = n
        elif 0 <= stop < n:
            pass
        else:
            raise Exception(f"{name}: stop out of bounds: {stop} >= {n}")
        try:
            conf[f"{name}_slice"] = slice(start, stop, step)
        except ValueError:
            raise Exception(f"{name}: invalid slice args: ({start}, {stop}, {step})")

    prepare_slice("lon")
    prepare_slice("lat")


def len_slice(s):
    """Count number of elements selected by slice."""
    return len(range(s.start, s.stop, s.step))


def transfer_ncattrs(fi, fo):
    """Transfer global attributes from in- to outfile."""
    ncattrs = {ncattr: fi.getncattr(ncattr) for ncattr in fi.ncattrs()}
    fo.setncatts(ncattrs)


def transfer_dimensions(fi, fo, **conf):
    """Transfer all dimensions from in- to outfile."""
    for dim in fi.dimensions.values():
        transfer_dimension(fi, fo, dim, **conf)


def transfer_dimension(fi, fo, dim, lat_name, lon_name, lat_slice, lon_slice, **conf):
    """Transfer single dimension from in- to outfile."""

    # Determine dimension size
    if dim.isunlimited():
        size = None
    elif dim.name == lat_name:
        size = len_slice(lat_slice)
    elif dim.name == lon_name:
        size = len_slice(lon_slice)
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
        transfer_variable(fi, fo, var, lat_name, lon_name, lat_slice, lon_slice, **conf)


def transfer_variables(fi, fo, **conf):
    """Transfer all regular variables from in- to outfile."""
    for var in fi.variables.values():
        if var.name not in fi.dimensions:
            transfer_variable(fi, fo, var, **conf)


def transfer_variable(fi, fo, var, lat_name, lon_name, lat_slice, lon_slice, **conf):
    """Transfer single variable from in- to outfile."""

    # Create variable
    new_var = fo.createVariable(
        varname=var.name, datatype=var.datatype, dimensions=var.dimensions,
    )

    # Transfer data, slicing along lat/lon
    if var.dimensions:
        inds = []
        for dim_name in var.dimensions:
            inds.append(
                {lat_name: lat_slice, lon_name: lon_slice}.get(dim_name, slice(None),)
            )
        new_var[:] = var[inds]

    # Transfer variable attributes
    transfer_ncattrs(var, new_var)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
