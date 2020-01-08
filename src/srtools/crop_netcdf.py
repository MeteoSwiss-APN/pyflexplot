# -*- coding: utf-8 -*-
"""
Crop fields in a NetCDF file.
"""
import click
import functools
import netCDF4 as nc4

__version__ = "0.1.0"

# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"],)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "in_file_path", metavar="infile", nargs=1, type=click.Path(exists=True),
)
@click.argument(
    "out_file_path", metavar="outfile", nargs=1, type=click.Path(),
)
@click.option(
    "--rlat",
    "rlat_slice",
    help="Rotated latitude index slice arguments. STOP is exclusive.",
    type=(int, int, int),
    metavar="START STOP STEP",
    default=(0, -1, 1),
)
@click.option(
    "--rlon",
    "rlon_slice",
    help="Rotated longitude index slice arguments. STOP is exclusive.",
    type=(int, int, int),
    metavar="START STOP STEP",
    default=(0, -1, 1),
)
@click.option(
    "--rlat-name", help="Name of rotated latitude dimension.", default="rlat",
)
@click.option(
    "--rlon-name", help="Name of rotated longitude dimension.", default="rlon",
)
def main(in_file_path, out_file_path, **conf):

    click.echo()
    click.echo(f"infile     : {in_file_path}")
    click.echo(f"outfile    : {out_file_path}")
    click.echo(f"rlat_name  : {conf['rlat_name']}")
    click.echo(f"rlon_name  : {conf['rlon_name']}")
    click.echo(f"rlat_slice : ({', '.join([str(f) for f in conf['rlat_slice']])})")
    click.echo(f"rlon_slice : ({', '.join([str(f) for f in conf['rlon_slice']])})")
    click.echo()

    with nc4.Dataset(in_file_path, "r") as fi, nc4.Dataset(out_file_path, "w") as fo:
        prepare_slices(fi, conf)
        transfer_ncattrs(fi, fo)
        transfer_dimensions(fi, fo, **conf)
        transfer_variables(fi, fo, **conf)


def prepare_slices(fi, conf):
    """Prepare rlon/rlat slices from input arguments."""

    def prepare_slice(name):
        start, stop, step = conf.pop(f"{name}_slice")
        n = len(fi.dimensions[conf[f"{name}_name"]])
        if stop == -1:
            stop = n
        elif 0 <= stop < n:
            pass
        else:
            raise Exception(f"{name}: stop out of bounds: {stop} > {n}")
        try:
            conf[f"{name}_slice"] = slice(start, stop, step)
        except ValueError:
            raise Exception(f"{name}: invalid slice args: ({start}, {stop}, {step})")

    prepare_slice("rlon")
    prepare_slice("rlat")


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


def transfer_dimension(
    fi, fo, dim, rlat_name, rlon_name, rlat_slice, rlon_slice, **conf
):
    """Transfer single dimension from in- to outfile."""

    # Determine dimension size
    if dim.isunlimited():
        size = None
    elif dim.name == rlat_name:
        size = len_slice(rlat_slice)
    elif dim.name == rlon_name:
        size = len_slice(rlon_slice)
    else:
        size = dim.size

    # Create dimension
    fo.createDimension(dim.name, size)

    # Transfer corresponding variable
    try:
        var = fi.variables[dim.name]
    except KeyError as e:
        # There's none; that's fine
        pass
    else:
        transfer_variable(
            fi, fo, var, rlat_name, rlon_name, rlat_slice, rlon_slice, **conf
        )


def transfer_variables(fi, fo, **conf):
    """Transfer all regular variables from in- to outfile."""
    for var in fi.variables.values():
        if var.name not in fi.dimensions:
            transfer_variable(fi, fo, var, **conf)


def transfer_variable(
    fi, fo, var, rlat_name, rlon_name, rlat_slice, rlon_slice, **conf
):
    """Transfer single variable from in- to outfile."""

    # Create variable
    new_var = fo.createVariable(
        varname=var.name, datatype=var.datatype, dimensions=var.dimensions,
    )

    # Transfer data, slicing along rlat/rlon
    if var.dimensions:
        inds = []
        for dim_name in var.dimensions:
            inds.append(
                {rlat_name: rlat_slice, rlon_name: rlon_slice,}.get(
                    dim_name, slice(None),
                )
            )
        new_var[:] = var[inds]

    # Transfer variable attributes
    transfer_ncattrs(var, new_var)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
