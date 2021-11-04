"""Crop fields in a NetCDF file."""
from __future__ import annotations

# Standard library
import dataclasses as dc
import sys
from typing import Any
from typing import Optional
from typing import Union

# Third-party
import click
import netCDF4 as nc4

__version__ = "0.1.0"


@dc.dataclass
class Setup:
    lat_name: str
    lon_name: str
    lat_slice: slice
    lon_slice: slice
    set_const: Optional[float]

    def dict(self) -> dict[str, Any]:
        return dc.asdict(self)


# Show default values of options by default
_click_option = click.option
click.option = lambda *args, **kwargs: _click_option(
    *args, **{**kwargs, "show_default": True}
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "in_file_path",
    metavar="infile",
    nargs=1,
    type=click.Path(exists=True),
)
@click.argument(
    "out_file_path",
    metavar="outfile",
    nargs=1,
    type=click.Path(),
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
    "--lat-name",
    help="Name of latitude dimension.",
    default="lat",
)
@click.option(
    "--lon-name",
    help="Name of longitude dimension.",
    default="lon",
)
@click.option(
    "--set-const",
    help="Set field to constant value.",
    type=float,
    default=None,
)
def main(in_file_path: str, out_file_path: str, **kwargs_setup: Any) -> None:
    kwargs_setup["lat_slice"] = slice(*kwargs_setup["lat_slice"])
    kwargs_setup["lon_slice"] = slice(*kwargs_setup["lon_slice"])
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


def transfer_ncattrs(fi: nc4.Dataset, fo: nc4.Dataset) -> None:
    """Transfer global attributes from in- to outfile."""
    ncattrs = {ncattr: fi.getncattr(ncattr) for ncattr in fi.ncattrs()}
    fo.setncatts(ncattrs)


def transfer_dimensions(fi: nc4.Dataset, fo: nc4.Dataset, setup: Setup) -> None:
    """Transfer all dimensions from in- to outfile."""
    if setup.lat_name not in fi.dimensions:
        raise Exception(f"dimension '{setup.lat_name}' not among {list(fi.dimensions)}")
    if setup.lon_name not in fi.dimensions:
        raise Exception(f"dimension '{setup.lon_name}' not among {list(fi.dimensions)}")
    for dim in fi.dimensions.values():
        transfer_dimension(fi, fo, dim, setup)


def transfer_dimension(
    fi: nc4.Dataset, fo: nc4.Dataset, dim: nc4.Dimension, setup: Setup
) -> None:
    """Transfer single dimension from in- to outfile."""
    # Determine dimension size
    if dim.isunlimited():
        size = None
    elif dim.name == setup.lat_name and setup.lat_slice != slice(None):
        size = len(
            range(setup.lat_slice.start, setup.lat_slice.stop, setup.lat_slice.step)
        )
    elif dim.name == setup.lon_name and setup.lon_slice != slice(None):
        size = len(
            range(setup.lon_slice.start, setup.lon_slice.stop, setup.lon_slice.step)
        )
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


def len_slice(
    arg: Union[slice, tuple[Optional[int], Optional[int], Optional[int]]], n: int
) -> int:
    """Count number of elements selected by slice."""
    if not isinstance(arg, slice):
        if arg == (None, None, None):
            return n
        arg = slice(*arg)
    return len(range(arg.start, arg.stop, arg.step))


def transfer_variables(fi: nc4.Dataset, fo: nc4.Dataset, setup: Setup) -> None:
    """Transfer all regular variables from in- to outfile."""
    for var in fi.variables.values():
        if var.name not in fi.dimensions:
            transfer_variable(fo, var, setup)


def transfer_variable(fo: nc4.Dataset, var: nc4.Dataset, setup: Setup) -> None:
    """Transfer single variable from in- to outfile."""
    # Create variable
    new_var = fo.createVariable(
        varname=var.name, datatype=var.datatype, dimensions=var.dimensions
    )

    # Transfer data, slicing along lat/lon
    if var.dimensions:
        inds = []
        for dim_name in var.dimensions:
            if dim_name == setup.lat_name:
                inds.append(setup.lat_slice)
            elif dim_name == setup.lon_name:
                inds.append(setup.lon_slice)
            else:
                inds.append(slice(None))
        if (
            setup.set_const is not None
            and var.name not in fo.dimensions
            and setup.lat_name in var.dimensions
            and setup.lon_name in var.dimensions
        ):
            new_var[:] = setup.set_const
        else:
            new_var[:] = var[inds]

    # Transfer variable attributes
    transfer_ncattrs(var, new_var)


if __name__ == "__main__":
    sys.exit(1)  # pragma: no cover
