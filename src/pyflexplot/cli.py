# -*- coding: utf-8 -*-
"""
Command line interface.
"""
import click
import dataclasses
import functools
import logging as log
import os
import re
import sys
import tomlkit
import warnings

from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

from srutils.click import CharSepList
from srutils.click import DerivChoice
from srutils.various import isiterable

from .examples import show_example
from .field_specs import FieldSpecs
from . import __version__
from .io import FileReader
from .plotter import Plotter
from .utils import count_to_log_level
from .var_specs import MultiVarSpecs

# # To debug segmentation fault, uncomment and run with PYTHONFAULTHANDLER=1
# import faulthandler
# faulthandler.enable()


# Show default values of options by default
click.option = functools.partial(click.option, show_default=True)


# comma_sep_list_of_unique_ints = CharSepList(int, ",", unique=True)
plus_sep_list_of_unique_ints = CharSepList(int, "+", unique=True)


def not_implemented(msg):
    def f(ctx, param, value):
        if value:
            click.echo(f"not implemented: {msg}")
            ctx.exit(1)

    return f


@click.command(context_settings={"help_option_names": ["-h", "--help"]},)
@click.version_option(__version__, "--version", "-V", message="%(version)s")
@click.option(
    "--infile",
    "infiles",
    help="Input file path(s). May contain format keys.",
    type=str,
    multiple=True,
)
@click.option(
    "--outfile", help="Output file path. May contain format keys", type=str,
)
# --- Execution
@click.option(
    "--dry-run",
    "dry_run",
    help="Perform a trial run with no changes made.",
    is_flag=True,
    default=False,
    # SR_TMP <
    is_eager=True,
    callback=not_implemented("--dry-run"),
    # SR_TMP >
)
@click.option(
    "--verbose",
    "-v",
    "verbose",
    help="Increase verbosity; specify multiple times for more.",
    count=True,
)
@click.option(
    "--open-first",
    "open_first_cmd",
    help=(
        "Shell command to open the first plot as soon as it is available. The file "
        "path is appended to the command, unless explicitly embedded with the format "
        "key '{file}', which allows one to use more complex commands than simple "
        "application names (example: 'eog {file} >/dev/null 2>&1' instead of 'eog' to "
        "silence the application 'eog')."
    ),
)
@click.option(
    "--open-all", "open_all_cmd", help="Like --open-first, but for all plots.",
)
@click.option(
    "--example",
    help="Example commands.",
    type=click.Choice(["naz-det-sh"]),
    callback=show_example,
    expose_value=False,
)
# --- Input
@click.option(
    "--config", "config_file", help="Configuration file (TOML).", type=click.File("r"),
)
@click.option(
    "--time-idx",
    "time_idxs",
    help="Index of time (zero-based). Format key: '{time_idx}'.",
    type=int,
    multiple=True,
)
@click.option(
    "--age-class-idx",
    "age_class_idxs",
    help="Index of age class (zero-based). Format key: '{age_class_idx}'.",
    type=int,
    multiple=True,
)
@click.option(
    "--nout-rel-idx",
    "nout_rel_idxs",
    help="Index of noutrel (zero-based). Format key: '{noutrel_idx}'.",
    type=int,
    multiple=True,
)
@click.option(
    "--release-point-idx",
    "release_point_idxs",
    help="Index of release point (zero-based). Format key: '{rls_pt_idx}'.",
    type=int,
    multiple=True,
)
@click.option(
    "--species-id",
    "species_ids",
    help=(
        "Species id(s) (default: 0). To sum up multiple species, combine their ids "
        "with '+'. Format key: '{species_id}'."
    ),
    type=plus_sep_list_of_unique_ints,
    multiple=True,
)
@click.option(
    "--level-idx",
    "level_idxs",
    help=(
        "Index/indices of vertical level (zero-based, bottom-up). To sum up multiple "
        "levels, combine their indices with '+'. Format key: '{level_idx}'."
    ),
    type=plus_sep_list_of_unique_ints,
    multiple=True,
)
@click.option(
    "--deposition-type",
    "deposition_types",
    help=(
        "Type of deposition. Part of the plot variable name that may be embedded in "
        "the plot file path with the format key '{variable}'."
    ),
    type=DerivChoice(["wet", "dry"], {"tot": ("wet", "dry")}),
    multiple=True,
)
@click.option(
    "--member-id",
    "-m",
    "ens_member_id_lst",
    help=(
        "Ensemble member id. Repeat for multiple members. Omit for deterministic "
        "simulation data. Use the format key '{member_id}' to embed the member id(s) "
        "in the plot file path."
    ),
    type=int,
    multiple=True,
)
# --- Preproc
@click.option(
    "--integrate/--no-integrate",
    "integrate_lst",
    help=(
        "Integrate field over time. Use the format key '{integrate}' to embed "
        "'[no-]int' in the plot file path."
    ),
    is_flag=True,
    default=[False],
    multiple=True,
)
@click.option(
    "--scale",
    "scale_fact",
    help="Scale field before plotting. Useful for debugging.",
    type=float,
    default=None,
)
# --- Plot
# SR_TMP <
@click.option(
    "--field",
    "field",
    help="Input field to be plotted.",
    type=click.Choice(["concentration", "deposition"]),
)
@click.option(
    "--simulation-type",
    "simulation_type",
    help="Type of simulation.",
    type=click.Choice(["deterministic", "ensemble"]),
)
@click.option(
    "--plot-type",
    "plot_type",
    help="Type of plot.",
    type=click.Choice(
        [
            "auto",
            "affected_area",
            "affected_area_mono",
            "ens_mean",
            "ens_max",
            "ens_thr_agrmt",
        ]
    ),
    default="auto",
)
# SR_TMP >
@click.option(
    "--lang",
    "lang",
    help="Language. Format key: '{lang}'.",
    type=click.Choice(["en", "de"]),
)
@click.option(
    "--domain",
    "domain",
    help=(
        "Plot domain. Defaults to 'data', which derives the domain size from the input "
        "data. Use the format key '{domain}' to embed the domain name in the plot file "
        "path."
    ),
    type=click.Choice(["auto", "ch"]),
)
@click.option(
    "--reverse-legend/--no-reverse-legend",
    "reverse_legend",
    help=(
        "Reverse the order of the level ranges and corresponding colors in the plot "
        "legend such that the levels increase from top to bottom instead of decreasing."
    ),
    is_flag=True,
    default=False,
)
# ---
@click.pass_context
def cli(ctx, config_file, **conf_raw):
    """
    Read NetCDF output of a deterministic or ensemble ``FLEXPART`` dispersion
    simulation from INFILE(S) to create the plot OUTFILE.

    Both INFILE(S) and OUTFILE may contain format keys that are replaced by
    values derived from options and/or the input data, which allows one to
    specify multipe input and/or output files with a single string. The syntax
    is that of Python format strings, with the variable name wrapped in curly
    braces.

    Example:

    \b
    $ pyflexplot "ens_mem{member_id:03}.nc" "ens_spc-{species_id:02}.png" \\
    >   --member-id={1..10} --species-id={0,1} ...
    # input  : ens_mem001.nc, ens_mem002.nc, ...,  ens_mem010.nc
    # output : ens_spc00.png, ens_spc01.png

    """

    click.echo("Welcome fellow PyFlexPlotter!")

    log.basicConfig(level=count_to_log_level(conf_raw["verbose"]))

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Remove empty arguments
    for key, val in conf_raw.copy().items():
        if val is None or (isiterable(val, str_ok=False) and not val):
            del conf_raw[key]

    # Read config file
    # SR_TODO Write classmethod to instatiate Conf from config file AND CLI
    if config_file is None:
        config = None
    else:
        config = Config.from_file(config_file)
        config.update(conf_raw)

        # SR_TMP < TODO eliminate
        conf_raw["infiles"] = config.infiles
        conf_raw["member_ids"] = config.member_ids
        conf_raw["outfile"] = config.outfile
        #
        conf_raw["field"] = config.field
        conf_raw["simulation_type"] = config.simulation_type
        conf_raw["plot_type"] = config.plot_type
        #
        conf_raw["domain"] = config.domain or "auto"
        conf_raw["lang"] = config.lang
        #
        conf_raw["age_class_idxs"] = config.age_class_idxs
        conf_raw["deposition_types"] = config.deposition_types
        conf_raw["level_idxs"] = config.level_idxs
        conf_raw["nout_rel_idxs"] = config.nout_rel_idxs
        conf_raw["release_point_idxs"] = config.release_point_idxs
        conf_raw["species_ids"] = config.species_ids
        conf_raw["time_idxs"] = config.time_idxs
        #
        conf_raw["integrates"] = config.integrates
        # SR_TMP >

    # SR_TMP <
    mandatory_args = ["field", "simulation_type"]
    for arg in mandatory_args:
        if not conf_raw.get(arg):
            raise Exception(f"argument missing: {arg}")
    # SR_TMP >

    # Create plots
    create_plots(conf_raw)

    return 0


@dataclass
class Config:
    #
    infiles: Optional[List[str]] = None
    member_ids: Optional[List[int]] = None
    outfile: Optional[str] = None
    #
    field: Optional[str] = None
    simulation_type: Optional[str] = None
    plot_type: Optional[str] = None
    #
    domain: Optional[str] = None
    lang: Optional[str] = None
    #
    age_class_idxs: Optional[List[int]] = None
    deposition_types: Optional[List[str]] = None
    level_idxs: Optional[Union[List[int], List[List[int]]]] = None
    nout_rel_idxs: Optional[List[int]] = None
    release_point_idxs: Optional[List[int]] = None
    species_ids: Optional[Union[List[int], List[List[int]]]] = None
    time_idxs: Optional[List[int]] = None
    #
    integrates: Optional[List[bool]] = None

    def update(self, dct):
        for key, val in dct.items():
            if not hasattr(self, key):
                warnings.warn(f"{type(self).__name__}.update: unknown key: {key}")
            else:
                setattr(self, key, val)

    @classmethod
    def from_file(cls, file):
        s = file.read()
        try:
            data = tomlkit.parse(s)
        except Exception as e:
            raise Exception(
                f"error parsing TOML file {file.name} ({type(e).__name__}: {e})"
            )
        # SR_TMP <
        required_keys = ["plot"]
        optional_keys = []
        unknown_keys = [
            k for k in data.keys() if k not in required_keys + optional_keys
        ]
        if unknown_keys:
            raise NotImplementedError(
                f"{len(unknown_keys)} unknown section(s) in {file.name}", unknown_keys,
            )
        # SR_TMP >
        obj = cls(**data["plot"])
        return obj


def create_plots(conf_raw):
    """
    Read and plot FLEXPART data.
    """

    # SR_TMP <
    if conf_raw["plot_type"] == "auto":
        conf_raw["plot_type"] = conf_raw["field"]
    # SR_TMP >

    field = conf_raw["field"]
    simulation_type = conf_raw["simulation_type"]
    if simulation_type == "deterministic":
        cls_name = f"{field}"
    elif simulation_type == "ensemble":
        cls_name = f"{conf_raw['plt']['plot_type']}_{field}"

    # Read input fields
    field_lst = read_fields(cls_name, conf_raw)

    # Prepare plotter
    plotter = Plotter()

    def fct_plot():
        return plotter.run(
            cls_name,
            field_lst,
            conf_raw["outfile"],
            lang=conf_raw["lang"],
            scale_fact=conf_raw.get("scale_fact"),
        )

    # Note: Plotter.run yields the output file paths on-the-go
    out_file_paths = []
    for i, out_file_path in enumerate(fct_plot()):
        out_file_paths.append(out_file_path)

        if conf_raw.get("open_first_cmd") and i == 0:
            # Open the first file as soon as it's available
            open_plots(conf_raw["open_first_cmd"], [out_file_path])

    if conf_raw.get("open_all_cmd"):
        # Open all plots
        open_plots(conf_raw["open_all_cmd"], out_file_paths)


def prep_var_specs_dct(conf_raw):
    """
    Prepare the variable specifications dict from the raw CLI input.
    """

    var_specs_raw = {
        "time_lst": conf_raw.get("time_idxs") or (0,),
        "nageclass_lst": conf_raw.get("age_class_idxs") or (0,),
        "noutrel_lst": conf_raw.get("nout_rel_idxs") or (0,),
        "numpoint_lst": conf_raw.get("release_point_idxs") or (0,),
        "species_id_lst": conf_raw.get("species_ids") or (1,),
        "integrate_lst": conf_raw.get("integrates") or (False,),
    }

    field = conf_raw["field"]
    if field == "concentration":
        var_specs_raw["level_lst"] = conf_raw.get("level_idxs") or (0,)
    elif field == "deposition":
        var_specs_raw["deposition_lst"] = conf_raw.get("deposition_types") or ("tot",)
    else:
        raise NotImplementedError(f"field='{field}'")

    var_specs_dct = {}
    for key, val in var_specs_raw.items():
        if key.endswith("_lst"):
            key = key[: -len("_lst")]
        if isinstance(val, (tuple, list)):
            val = [tuple(e) if isinstance(e, list) else e for e in val]
        else:
            raise Exception(f"invalid value type {type(val).__name__}", key)
        if len(val) == 1:
            val = next(iter(val))
        var_specs_dct[key] = val

    return var_specs_dct


def read_fields(cls_name, conf_raw):
    """
    TODO
    """

    # SR_TMP <
    simulation_type = conf_raw["simulation_type"]
    lang = conf_raw["lang"]
    plot_type = conf_raw["plot_type"]
    # SR_TMP >

    # SR_TMP < TODO find cleaner solution
    if simulation_type == "ensemble":
        if plot_type == "ens_thr_agrmt":
            ens_var_setup = {"thr": 1e-9}
        else:
            ens_var_setup = {}
    else:
        ens_var_setup = None
    # SR_TMP >

    # SR_TMP <<< TODO clean this up
    attrs = {"lang": lang}
    if simulation_type == "ensemble":
        if conf_raw["ens_member_id_lst"] is not None:
            attrs["member_ids"] = conf_raw["ens_member_id_lst"]
        assert plot_type.startswith("ens_"), plot_type
        attrs["ens_var"] = plot_type[4:]
        attrs["ens_var_setup"] = ens_var_setup

    # Create variable specification objects
    var_specs_dct = prep_var_specs_dct(conf_raw)
    multi_var_specs_lst = MultiVarSpecs.create(
        cls_name, var_specs_dct, lang=lang, words=None,
    )

    # Determine fields specifications (one for each eventual plot)
    fld_specs_lst = [
        FieldSpecs(cls_name, multi_var_specs, attrs, lang=lang)
        for multi_var_specs in multi_var_specs_lst
    ]

    # Read fields
    field_lst = []
    for raw_path in conf_raw["infiles"]:
        field_lst.extend(FileReader(raw_path).run(fld_specs_lst, lang=lang))

    return field_lst


def open_plots(cmd, file_paths):
    """
    Open a plot file using a shell command.
    """

    # If not yet included, append the output file path
    if "{file}" not in cmd:
        cmd += " {file}"

    # Ensure that the command is run in the background
    if not cmd.rstrip().endswith("&"):
        cmd += " &"

    # Run the command
    cmd = cmd.format(file=" ".join(file_paths))
    os.system(cmd)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
