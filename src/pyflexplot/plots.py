# -*- coding: utf-8 -*-
"""
Plot types.

Note that this file is currently a mess because it is the result of collecting
all sorts of plot-type specific logic from throughout the code in order to
centralize it.

There's tons of nested if-statements etc. because the goal during the cleanup
phase that included the centralization was to avoid falling into the trap of
premature design/overdesign again (as has happened repeatedly during the early
stages of development).

Instead, all the logic is collected here in a straightforward but dirty way
until sane design choices emerge from the code mess.
"""
# Standard library
import warnings
from datetime import datetime
from textwrap import dedent
from typing import Any
from typing import cast
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import matplotlib as mpl
import numpy as np

# First-party
from srutils.geo import Degrees

# Local
from .boxed_plot import BoxedPlot
from .boxed_plot import BoxedPlotConfig
from .boxed_plot import DummyBoxedPlot
from .data import Field
from .data import FieldAllNaNError
from .formatting import escape_format_keys
from .formatting import format_level_ranges
from .formatting import format_range
from .logging import log
from .meta_data import format_unit
from .meta_data import MetaData
from .plot_elements import MapAxes
from .plot_elements import MapAxesConf
from .plot_elements import TextBoxAxes
from .plot_layouts import BoxedPlotLayoutDeterministic
from .plot_layouts import BoxedPlotLayoutEnsemble
from .plot_layouts import BoxedPlotLayoutType
from .setup import FilePathFormatter
from .setup import Setup
from .setup import SetupCollection
from .typing import ColorType
from .words import SYMBOLS
from .words import TranslatedWords
from .words import WORDS
from .words import Words


def create_map_conf(field: Field) -> MapAxesConf:
    domain = field.var_setups.collect_equal("domain")
    model = field.var_setups.collect_equal("model")

    conf_base: Dict[str, Any] = {"lang": field.var_setups.collect_equal("lang")}

    conf_model_ifs: Dict[str, Any] = {"geo_res": "50m"}
    conf_model_cosmo: Dict[str, Any] = {"geo_res": "10m"}

    # SR_TMP < TODO generalize based on meta data
    conf_domain_japan: Dict[str, Any] = {
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 4_000_000,
        "lllat": 20,
        "urlat": 50,
        "lllon": 110,
        "urlon": 160,
        "ref_dist_conf": {"dist": 500},
    }
    # SR_TMP >
    conf_scale_continent: Dict[str, Any] = {
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 300_000,
    }
    conf_scale_country: Dict[str, Any] = {
        "geo_res_cities": "10m",
        "geo_res_rivers": "10m",
        "min_city_pop": 0,
        "ref_dist_conf": {"dist": 25},
    }

    conf: Dict[str, Any]
    if model.startswith("cosmo") and domain == "auto":
        conf = {
            **conf_base,
            **conf_model_cosmo,
            **conf_scale_continent,
            "zoom_fact": 1.05,
        }
    elif model.startswith("cosmo1") and domain == "ch":
        conf = {
            **conf_base,
            **conf_model_cosmo,
            **conf_scale_country,
            "zoom_fact": 3.6,
            "rel_offset": (-0.02, 0.045),
        }
    elif model.startswith("cosmo2") and domain == "ch":
        conf = {
            **conf_base,
            **conf_model_cosmo,
            **conf_scale_country,
            "zoom_fact": 3.23,
            "rel_offset": (0.037, 0.1065),
        }
    elif model == "ifs" and domain == "auto":
        # SR_TMP < TODO Generalize IFS domains
        conf = {**conf_base, **conf_model_ifs, **conf_domain_japan}
        # SR_TMP >
    else:
        raise Exception(f"unknown domain '{domain}' for model '{model}'")

    return MapAxesConf(**conf)


def capitalize(s: str) -> str:
    """Capitalize the first letter while leaving all others as they are."""
    if not s:
        return s
    try:
        return s[0].upper() + s[1:]
    except Exception:
        raise ValueError("s not capitalizable", s)


# SR_TODO Create dataclass with default values for text box setup
def create_plot_config(
    setup: Setup, words: TranslatedWords, symbols: Words, mdata: MetaData,
) -> BoxedPlotConfig:
    words.set_active_lang(setup.core.lang)
    new_config_dct: Dict[str, Any] = {
        "setup": setup,
        "mdata": mdata,
    }

    if setup.core.input_variable == "concentration":
        new_config_dct["n_levels"] = 8
    elif setup.core.input_variable == "deposition":
        new_config_dct["n_levels"] = 9
    if setup.get_simulation_type() == "deterministic":
        if setup.core.plot_variable == "affected_area_mono":
            new_config_dct["extend"] = "none"
            new_config_dct["n_levels"] = 1
    elif setup.get_simulation_type() == "ensemble":
        if setup.core.ens_variable not in ["minimum", "maximum", "median", "mean"]:
            new_config_dct.update(
                {
                    "extend": "both",
                    "legend_rstrip_zeros": False,
                    "level_range_style": "int",
                    "level_ranges_align": "left",
                    "mark_field_max": False,
                    "levels_scale": "lin",
                }
            )
            if setup.core.ens_variable == "probability":
                new_config_dct.update({"n_levels": 9, "d_level": 10})
            elif setup.core.ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                new_config_dct.update({"n_levels": 9, "d_level": 3})
            elif setup.core.ens_variable == "cloud_occurrence_probability":
                new_config_dct.update({"n_levels": 9, "d_level": 10})

    # Colors
    n_levels = new_config_dct["n_levels"]
    extend = new_config_dct.get("extend", "max")
    cmap = new_config_dct.get("cmap", "flexplot")
    if setup.core.plot_variable == "affected_area_mono":
        colors = (np.array([(200, 200, 200)]) / 255).tolist()
    elif cmap == "flexplot":
        colors = colors_flexplot(n_levels, extend)
    else:
        cmap = mpl.cm.get_cmap(cmap)
        colors = [cmap(i / (n_levels - 1)) for i in range(n_levels)]
    new_config_dct["colors"] = colors

    # Markers
    new_config_dct["markers"] = {
        "max": {
            "marker": "+",
            "color": "black",
            "markersize": 10,
            "markeredgewidth": 1.5,
        },
        "site": {
            "marker": "^",
            "markeredgecolor": "red",
            "markerfacecolor": "white",
            "markersize": 7.5,
            "markeredgewidth": 1.5,
        },
    }

    new_config_dct["labels"] = create_box_labels(setup, words, symbols, mdata)

    return BoxedPlotConfig(**new_config_dct)


# pylint: disable=R0914  # too-many-locals
def create_box_labels(
    setup: Setup, words: TranslatedWords, symbols: Words, mdata: MetaData,
) -> Dict[str, Dict[str, Any]]:

    # SR_TMP <
    names = format_names_etc(setup, words, mdata)
    short_name = names["short"]
    long_name = names["long"]
    var_name = names["var"]
    unit = names["unit"]
    # SR_TMP >

    labels: Dict[str, Dict[str, Any]] = {}
    assert isinstance(mdata.simulation_integr_start.value, datetime)  # mypy
    assert isinstance(mdata.simulation_now.value, datetime)  # mypy
    integr_period = format_integr_period(
        mdata.simulation_integr_start.value,
        mdata.simulation_now.value,
        setup,
        words,
        cap=True,
    )
    labels["top"] = {
        "tl": "",  # SR_TMP force into 1st position of dict (for tests)
        "bl": (
            f"{integr_period}"
            f" {words['since']}"
            f" +{mdata.simulation_integr_start_rel}"
        ),
        "tr": (f"{mdata.simulation_now} (+{mdata.simulation_now_rel})"),
        "br": (
            f"{mdata.simulation_now_rel} {words['since']}"
            f" {words['release_start']}"
            # f" {words['at', 'place']} {mdata.release_site_name}"
        ),
    }
    labels["title_2nd"] = {
        "tc": f"{mdata.species_name}",
        "bc": f"{mdata.release_site_name}",
    }
    labels["right_top"] = {
        "title": f"{words['data'].c}",
        "lines": [],
    }
    labels["right_middle"] = {
        "title": "",  # SR_TMP Force into 1st position in dict (for tests)
        "title_unit": "",  # SR_TMP Force into 2nd position in dict (for tests)
        "release_site": words["release_site"].s,
        "site": words["site"].s,
        "max": words["maximum", "abbr"].s,
        "maximum": words["maximum"].s,
    }
    labels["right_bottom"] = {
        "title": words["release"].t,
        "start": words["start"].s,
        "end": words["end"].s,
        "latitude": words["latitude"].s,
        "longitude": words["longitude"].s,
        "lat_deg_fmt": format_coord_label("north", words, symbols),
        "lon_deg_fmt": format_coord_label("east", words, symbols),
        "height": words["height"].s,
        "rate": words["rate"].s,
        "mass": words["total_mass"].s,
        "site": words["site"].s,
        "release_site": words["release_site"].s,
        "max": words["maximum", "abbr"].s,
        "name": words["substance"].s,
        "half_life": words["half_life"].s,
        "deposit_vel": words["deposition_velocity", "abbr"].s,
        "sediment_vel": words["sedimentation_velocity", "abbr"].s,
        "washout_coeff": words["washout_coeff"].s,
        "washout_exponent": words["washout_exponent"].s,
    }

    labels["bottom"] = {
        "model_info": format_model_info(setup, words, mdata),
        "copyright": f"{symbols['copyright']}{words['meteoswiss']}",
    }

    if setup.core.input_variable == "concentration":
        labels["right_middle"]["tc"] = (
            f"{words['level']}:" f" {escape_format_keys(format_level_label(mdata))}"
        )
        labels["right_top"]["lines"].insert(
            0, f"{words['level'].c}:\t{escape_format_keys(format_level_label(mdata))}"
        )

    if setup.get_simulation_type() == "ensemble":
        if setup.core.ens_variable == "probability":
            labels["right_top"]["lines"].append(
                f"{words['cloud']}:\t{symbols['geq']} {setup.core.ens_param_thr}"
                f" {mdata.format('variable_unit')}"
            )
        elif setup.core.ens_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            labels["right_top"]["lines"].append(
                f"{words['cloud_density'].c}:\t{words['minimum', 'abbr']}"
                f" {setup.core.ens_param_thr} {mdata.format('variable_unit')}"
            )
            n_min = setup.core.ens_param_mem_min or 1
            n_tot = len((setup.ens_member_id or []))
            labels["right_top"]["lines"].append(
                f"{words['number_of', 'abbr'].c} {words['member', 'pl']}:"
                f"\t{words['minimum', 'abbr']} {setup.core.ens_param_mem_min}"
                r"$\,/\,$"
                f"{n_tot} ({n_min/(n_tot or 1):.0%})"
            )
        elif setup.core.ens_variable == "cloud_occurrence_probability":
            labels["right_top"]["lines"].append(
                f"{words['cloud_density'].c}:\t{words['minimum', 'abbr']}"
                f" {setup.core.ens_param_thr} {mdata.format('variable_unit')}"
            )
            n_min = setup.core.ens_param_mem_min or 1
            n_tot = len((setup.ens_member_id or []))
            labels["right_top"]["lines"].append(
                f"{words['number_of', 'abbr'].c} {words['member', 'pl']}:"
                f"\t{words['minimum', 'abbr']} {setup.core.ens_param_mem_min}"
                r"$\,/\,$"
                f"{n_tot} ({n_min/(n_tot or 1):.0%})"
            )

    # box_labels["top"]["tl"] = f"{long_name} {words['of']} {mdata.species_name}"
    labels["top"]["tl"] = long_name
    labels["right_top"]["lines"].insert(
        0, f"{words['input_variable'].c}:\t{capitalize(var_name)}",
    )
    labels["right_middle"]["title"] = short_name
    labels["right_middle"]["title_unit"] = f"{short_name} ({unit})"
    labels["right_middle"]["unit"] = unit

    # Capitalize all labels
    for label_group in labels.values():
        for name, label in label_group.items():
            if name == "lines":
                label = [capitalize(line) for line in label]
            else:
                label = capitalize(label)
            label_group[name] = label

    return labels


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0915  # too-many-statements
def format_names_etc(
    setup: Setup, words: TranslatedWords, mdata: MetaData
) -> Dict[str, str]:
    long_name = ""
    short_name = ""

    def format_var_names(setup: Setup, words: TranslatedWords) -> Tuple[str, str]:
        if setup.core.input_variable == "concentration":
            var_name = str(words["activity_concentration"])
            if setup.core.integrate:
                var_name_rel = (
                    f"{words['of', 'fg']} {words['integrated', 'g']}"
                    f" {words['activity_concentration']}"
                )
            else:
                var_name_rel = f"{words['of', 'fg']} {words['activity_concentration']}"
        elif setup.core.input_variable == "deposition":
            dep_type_word = (
                "total"
                if setup.deposition_type_str == "tot"
                else setup.deposition_type_str
            )
            var_name = f"{words[dep_type_word, 'f']} {words['surface_deposition']}"
            var_name_rel = (
                f"{words['of', 'fg']} {words[dep_type_word, 'g']}"
                f" {words['surface_deposition']}"
            )
        return var_name, var_name_rel

    # pylint: disable=W0621  # redefined-outer-name
    def format_unit(setup: Setup, words: TranslatedWords, mdata: MetaData) -> str:
        if setup.get_simulation_type() == "ensemble":
            if setup.core.ens_variable == "probability":
                return "%"
            elif setup.core.ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                return f"{words['hour', 'pl']}"
            elif setup.core.ens_variable == "cloud_occurrence_probability":
                return "%"
        return mdata.format("variable_unit")

    unit = format_unit(setup, words, mdata)
    var_name, var_name_rel = format_var_names(setup, words)

    # Short/long names #1: By variable
    if setup.core.input_variable == "concentration":
        if setup.core.integrate:
            long_name = f"{words['integrated']} {var_name}"
            short_name = (
                f"{words['integrated', 'abbr']} {words['concentration', 'abbr']}"
            )
        else:
            long_name = var_name
            short_name = str(words["concentration"])
    elif setup.core.input_variable == "deposition":
        long_name = var_name
        short_name = str(words["deposition"])

    # Short/long names #2: By plot variable/type
    if setup.get_simulation_type() == "deterministic":
        if setup.core.plot_variable.startswith("affected_area"):
            long_name = f"{words['affected_area']} {var_name_rel}"
    elif setup.get_simulation_type() == "ensemble":
        if setup.core.ens_variable == "minimum":
            long_name = f"{words['ensemble_minimum']} {var_name_rel}"
        elif setup.core.ens_variable == "maximum":
            long_name = f"{words['ensemble_maximum']} {var_name_rel}"
        elif setup.core.ens_variable == "median":
            long_name = f"{words['ensemble_median']} {var_name_rel}"
        elif setup.core.ens_variable == "mean":
            long_name = f"{words['ensemble_mean']} {var_name_rel}"
        elif setup.core.ens_variable == "probability":
            short_name = f"{words['probability']}"
            long_name = f"{words['probability']} {var_name_rel}"
        elif setup.core.ens_variable == "cloud_arrival_time":
            long_name = f"{words['cloud_arrival_time']}"
            short_name = f"{words['arrival']}"
        elif setup.core.ens_variable == "cloud_departure_time":
            long_name = f"{words['cloud_departure_time']}"
            short_name = f"{words['departure']}"
        elif setup.core.ens_variable == "cloud_occurrence_probability":
            short_name = f"{words['probability']}"
            long_name = f"{words['cloud_occurrence_probability']}"

    # SR_TMP <
    if not short_name:
        raise Exception("no short name")
    if not long_name:
        raise Exception("no long name")
    # SR_TMP >

    return {"short": short_name, "long": long_name, "var": var_name, "unit": unit}


def prepare_plot(
    field_lst: Sequence[Field],
    prev_out_file_paths: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Union[BoxedPlot, DummyBoxedPlot]:
    """Create plots while yielding them with the plot file path one by one."""
    log(dbg=f"preparing setups for plot based on {len(field_lst)} fields")
    var_setups_lst = [field.var_setups for field in field_lst]
    setup = SetupCollection.merge(var_setups_lst).compress()
    out_file_path = FilePathFormatter(prev_out_file_paths).format(setup)
    log(dbg=f"preparing plot {out_file_path}")
    if dry_run:
        return DummyBoxedPlot(out_file_path)
    else:
        # SR_TMP <  SR_MULTIPANEL
        if len(field_lst) > 1:
            raise NotImplementedError("multipanel plot")
        field = next(iter(field_lst))
        map_conf = create_map_conf(field)
        # SR_TMP >  SR_MULTIPANEL
        config = create_plot_config(setup, WORDS, SYMBOLS, cast(MetaData, field.mdata))
        return BoxedPlot(field, out_file_path, config, map_conf)


def create_plot(plot: BoxedPlot, write: bool = True) -> None:
    log(dbg=f"creating plot {plot.file_path}")
    layout: BoxedPlotLayoutType
    if plot.config.setup.get_simulation_type() == "deterministic":
        layout = BoxedPlotLayoutDeterministic(aspect=plot.config.fig_aspect)
    elif plot.config.setup.get_simulation_type() == "ensemble":
        layout = BoxedPlotLayoutEnsemble(aspect=plot.config.fig_aspect)
    axs_map = plot.add_map_plot(layout.rect_center())
    plot_add_text_boxes(plot, layout)
    plot_add_markers(plot, axs_map)
    if write:
        plot.write()
    else:
        plot.clean()
    log(dbg=f"created plot {plot.file_path}")


# SR_TMP <<< TODO Clean up nested functions! Eventually introduce class(es) of some kind
# pylint: disable=R0915  # too-many-statements
def plot_add_text_boxes(plot: BoxedPlot, layout: BoxedPlotLayoutType) -> None:
    # pylint: disable=R0915  # too-many-statements
    def fill_box_title(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the title box."""
        for position, label in plot.config.labels.get("top", {}).items():
            if position == "tl":
                font_size = plot.config.font_sizes.title_large
            else:
                font_size = plot.config.font_sizes.content_large
            box.text(
                label, loc=position, fontname=plot.config.font_name, size=font_size
            )

    def fill_box_2nd_title(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the secondary title box of the deterministic plot layout."""
        font_size = plot.config.font_sizes.content_large
        labels = plot.config.labels["title_2nd"]
        box.text(labels["tc"], loc="tc", fontname=plot.config.font_name, size=font_size)
        box.text(labels["bc"], loc="bc", fontname=plot.config.font_name, size=font_size)

    # pylint: disable=R0915  # too-many-statements
    def fill_box_data_info(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the data information box of the ensemble plot layout."""
        labels = plot.config.labels["right_top"]
        box.text(
            labels["title"],
            loc="tc",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.title_small,
        )
        box.text_block_hfill(
            labels["lines"],
            dy_unit=-4.0,
            dy_line=2.5,
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_small,
        )

    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=R0913  # too-many-arguments
    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R0915  # too-many-statements
    def fill_box_legend(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the box containing the plot legend."""

        labels = plot.config.labels["right_middle"]
        mdata = plot.config.mdata

        # Box title
        box.text(
            labels["title_unit"],
            loc="tc",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.title_small,
        )

        # dy_line: float = 3.0
        dy_line: float = 2.5

        w_legend_box: float = 4.0
        h_legend_box: float = 2.0

        dx_legend_box: float = -10
        dx_legend_label: float = -3

        dx_marker: float = dx_legend_box + 0.5 * w_legend_box
        dx_marker_label: float = dx_legend_label - 0.5

        # Vertical position of legend (depending on number of levels)
        dy0_labels = -5.0
        dy0_boxes = dy0_labels - 0.8 * h_legend_box

        # Format level ranges (contour plot legend)
        legend_labels = format_level_ranges(
            levels=plot.levels,
            style=plot.config.level_range_style,
            extend=plot.config.extend,
            rstrip_zeros=plot.config.legend_rstrip_zeros,
            align=plot.config.level_ranges_align,
        )

        # Legend labels (level ranges)
        box.text_block(
            legend_labels[::-1],
            loc="tc",
            dy_unit=dy0_labels,
            dy_line=dy_line,
            dx=dx_legend_label,
            ha="left",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_medium,
            family="monospace",
        )

        # Legend color boxes
        colors = plot.config.colors
        dy = dy0_boxes
        for color in colors[::-1]:
            box.color_rect(
                loc="tc",
                x_anker="left",
                dx=dx_legend_box,
                dy=dy,
                w=w_legend_box,
                h=h_legend_box,
                fc=color,
                ec="black",
                lw=1.0,
            )
            dy -= dy_line

        dy0_markers = dy0_boxes - dy_line * (len(legend_labels) - 0.3)
        dy0_marker = dy0_markers

        # Field maximum marker
        if plot.config.mark_field_max:
            dy_marker_label_max = dy0_marker
            dy0_marker -= dy_line
            dy_max_marker = dy_marker_label_max - 0.7
            assert plot.config.markers is not None  # mypy
            box.marker(
                loc="tc", dx=dx_marker, dy=dy_max_marker, **plot.config.markers["max"],
            )
            if np.isnan(plot.field.fld).all():
                s_val = "NaN"
            else:
                fld_max = np.nanmax(plot.field.fld)
                if 0.001 <= fld_max < 0.01:
                    s_val = f"{fld_max:.5f}"
                elif 0.01 <= fld_max < 0.1:
                    s_val = f"{fld_max:.4f}"
                elif 0.1 <= fld_max < 1:
                    s_val = f"{fld_max:.3f}"
                elif 1 <= fld_max < 10:
                    s_val = f"{fld_max:.2f}"
                elif 10 <= fld_max < 100:
                    s_val = f"{fld_max:.1f}"
                elif 100 <= fld_max < 1000:
                    s_val = f"{fld_max:.0f}"
                else:
                    s_val = f"{fld_max:.2E}"
                # s_val += r"$\,$" + labels["unit"]
            s = f"{labels['max']}: {s_val}"
            # s = f"{labels['max']} ({s_val})"
            # s = f"{labels['maximum']}:\n({s_val})"
            box.text(
                s=s,
                loc="tc",
                dx=dx_marker_label,
                dy=dy_marker_label_max,
                ha="left",
                fontname=plot.config.font_name,
                size=plot.config.font_sizes.content_medium,
            )

        # Release site marker
        if plot.config.mark_release_site:
            dy_site_label = dy0_marker
            dy0_marker -= dy_line
            dy_site_marker = dy_site_label - 0.7
            assert plot.config.markers is not None  # mypy
            box.marker(
                loc="tc",
                dx=dx_marker,
                dy=dy_site_marker,
                **plot.config.markers["site"],
            )
            # s = f"{labels['release_site']}"
            # s = f"{labels['site']} ({mdata.release_site_name})"
            s = f"{labels['site']}: {mdata.release_site_name}"
            box.text(
                s=s,
                loc="tc",
                dx=dx_marker_label,
                dy=dy_site_label,
                ha="left",
                fontname=plot.config.font_name,
                size=plot.config.font_sizes.content_medium,
            )

    # pylint: disable=R0915  # too-many-statements
    def fill_box_release_info(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the box containing the release info."""

        labels = plot.config.labels["right_bottom"]
        mdata = plot.config.mdata

        # Box title
        box.text(
            s=labels["title"],
            loc="tc",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.title_small,
        )

        # Release site coordinates
        lat = Degrees(mdata.release_site_lat.value)
        lon = Degrees(mdata.release_site_lon.value)
        lat_deg = labels["lat_deg_fmt"].format(d=lat.degs(), m=lat.mins(), f=lat.frac())
        lon_deg = labels["lon_deg_fmt"].format(d=lon.degs(), m=lon.mins(), f=lon.frac())

        # SR_TMP < TODO clean this up, especially for ComboMetaData (units messed up)!
        height = mdata.format("release_height", add_unit=True)
        rate = mdata.format("release_rate", add_unit=True)
        mass = mdata.format("release_mass", add_unit=True)
        substance = mdata.format("species_name", join_combo=" / ")
        half_life = mdata.format("species_half_life", add_unit=True)
        deposit_vel = mdata.format("species_deposit_vel", add_unit=True)
        sediment_vel = mdata.format("species_sediment_vel", add_unit=True)
        washout_coeff = mdata.format("species_washout_coeff", add_unit=True)
        washout_exponent = mdata.format("species_washout_exponent")
        # SR_TMP >

        info_blocks = dedent(
            f"""\
            {labels['site']}:\t{mdata.release_site_name}
            {labels['latitude']}:\t{lat_deg}
            {labels['longitude']}:\t{lon_deg}
            {labels['height']}:\t{height}

            {labels['start']}:\t{mdata.release_start}
            {labels['end']}:\t{mdata.release_end}
            {labels['rate']}:\t{rate}
            {labels['mass']}:\t{mass}

            {labels['name']}:\t{substance}
            {labels['half_life']}:\t{half_life}
            {labels['deposit_vel']}:\t{deposit_vel}
            {labels['sediment_vel']}:\t{sediment_vel}
            {labels['washout_coeff']}:\t{washout_coeff}
            {labels['washout_exponent']}:\t{washout_exponent}
            """
        )

        # Add lines bottom-up (to take advantage of baseline alignment)
        box.text_blocks_hfill(
            info_blocks,
            dy_unit=-4.0,
            dy_line=2.5,
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_small,
        )

    # pylint: disable=R0915  # too-many-statements
    def fill_box_footer(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the footer box containing the copyright etc."""

        labels = plot.config.labels["bottom"]

        # FLEXPART/model info
        box.text(
            s=labels["model_info"],
            loc="tl",
            dx=-0.7,
            dy=0.5,
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_small,
        )

        # MeteoSwiss Copyright
        box.text(
            s=labels["copyright"],
            loc="tr",
            dx=0.7,
            dy=0.5,
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_small,
        )

    plot.add_text_box("top", layout.rect_top(), fill_box_title)
    if isinstance(layout, BoxedPlotLayoutDeterministic):
        plot.add_text_box("right_top", layout.rect_right_top(), fill_box_2nd_title)
    elif isinstance(layout, BoxedPlotLayoutEnsemble):
        plot.add_text_box("right_top", layout.rect_right_top(), fill_box_data_info)
    plot.add_text_box("right_middle", layout.rect_right_middle(), fill_box_legend)
    plot.add_text_box("right_bottom", layout.rect_right_bottom(), fill_box_release_info)
    plot.add_text_box("bottom", layout.rect_bottom(), fill_box_footer, frame_on=False)


def plot_add_markers(plot: BoxedPlot, axs_map: MapAxes) -> None:
    config = plot.config

    if config.mark_release_site:
        assert isinstance(config.mdata.release_site_lon.value, float)  # mypy
        assert isinstance(config.mdata.release_site_lat.value, float)  # mypy
        assert config.markers is not None  # mypy
        axs_map.marker(
            lat=config.mdata.release_site_lat.value,
            lon=config.mdata.release_site_lon.value,
            **config.markers["site"],
        )

    if config.mark_field_max:
        assert config.markers is not None  # mypy
        try:
            max_lat, max_lon = plot.field.locate_max()
        except FieldAllNaNError:
            warnings.warn("skip maximum marker (all-nan field)")
        else:
            axs_map.marker(lat=max_lat, lon=max_lon, **config.markers["max"])


def colors_flexplot(n_levels: int, extend: str) -> Sequence[ColorType]:

    color_under = "darkgray"
    color_over = "lightgray"

    # def rgb(*vals):
    #     return np.array(vals, float) / 255

    # colors_core_8_old = [
    #     rgb(224, 196, 172),
    #     rgb(221, 127, 215),
    #     rgb(99, 0, 255),
    #     rgb(100, 153, 199),
    #     rgb(34, 139, 34),
    #     rgb(93, 255, 2),
    #     rgb(199, 255, 0),
    #     rgb(255, 239, 57),
    # ]
    colors_core_8 = [
        "bisque",
        "violet",
        "rebeccapurple",
        "cornflowerblue",
        "forestgreen",
        "yellowgreen",
        "greenyellow",
        "yellow",
    ]

    colors_core_7 = [colors_core_8[i] for i in (0, 1, 2, 3, 5, 6, 7)]
    colors_core_6 = [colors_core_8[i] for i in (1, 2, 3, 4, 5, 7)]
    colors_core_5 = [colors_core_8[i] for i in (1, 2, 4, 5, 7)]
    colors_core_4 = [colors_core_8[i] for i in (1, 2, 4, 7)]

    try:
        colors_core = {
            5: colors_core_4,
            6: colors_core_5,
            7: colors_core_6,
            8: colors_core_7,
            9: colors_core_8,
        }[n_levels]
    except KeyError:
        raise ValueError(f"n_levels={n_levels}")

    if extend == "none":
        return colors_core
    elif extend == "min":
        return [color_under] + colors_core
    elif extend == "max":
        return colors_core + [color_over]
    elif extend == "both":
        return [color_under] + colors_core + [color_over]
    raise ValueError(f"extend='{extend}'")


def format_model_info(setup: Setup, words: TranslatedWords, mdata: MetaData) -> str:
    if setup.model == "cosmo1":
        model_name = "COSMO-1"
    elif setup.model == "cosmo1e":
        model_name = "COSMO-1E"
    elif setup.model == "cosmo2":
        model_name = "COSMO-2"
    elif setup.model == "cosmo2e":
        model_name = "COSMO-2E"
    elif setup.model == "ifs":
        model_name = "IFS"

    model_info = None
    if setup.get_simulation_type() == "deterministic":
        if setup.model in ["cosmo1", "cosmo2", "ifs"]:
            model_info = model_name
        elif setup.model in ["cosmo1e", "cosmo2e"]:
            model_info = f"{model_name} {words['control_run']}"
    elif setup.get_simulation_type() == "ensemble":
        model_info = (
            f"{model_name} {words['ensemble']}"
            f" ({len(setup.ens_member_id or [])} {words['member', 'pl']}:"
            f" {format_range(setup.ens_member_id or [], fmt='02d')})"
        )
    if model_info is None:
        raise NotImplementedError(
            f"model setup '{setup.model}-{setup.get_simulation_type()}'"
        )
    return (
        f"{words['flexpart']} {words['based_on']} {model_info}"
        f", {mdata.simulation_start}"
    )


def format_level_label(mdata: MetaData) -> str:
    bot_unit = mdata.variable_level_bot_unit.value
    top_unit = mdata.variable_level_top_unit.value
    bot_level = mdata.variable_level_bot.value
    top_level = mdata.variable_level_top.value
    if top_unit != bot_unit:
        raise Exception("inconsistent level units", bot_unit, top_unit)
    unit = cast(str, bot_unit)
    level = format_vertical_level_range(bot_level, top_level, unit)
    if not level:
        return ""
    return f"{format_unit(level)}"


def format_vertical_level_range(
    value_bottom: Union[float, Sequence[float]],
    value_top: Union[float, Sequence[float]],
    unit: str,
) -> Optional[str]:

    if (value_bottom, value_top) == (-1, -1):
        return None

    def fmt(bot, top):
        return f"{bot:g}" + r"$-$" + f"{top:g} {unit}"

    try:
        # One level range (early exit)
        return fmt(value_bottom, value_top)
    except TypeError:
        pass

    # Multiple level ranges
    assert isinstance(value_bottom, Collection)  # mypy
    assert isinstance(value_top, Collection)  # mypy
    bots = sorted(value_bottom)
    tops = sorted(value_top)
    if len(bots) != len(tops):
        raise Exception(f"inconsistent no. levels: {len(bots)} != {len(tops)}")
    n = len(bots)
    if n == 2:
        # Two level ranges
        if tops[0] == bots[1]:
            return fmt(bots[0], tops[1])
        else:
            return f"{fmt(bots[0], tops[0])} + {fmt(bots[1], tops[1])}"
    elif n == 3:
        # Three level ranges
        if tops[0] == bots[1] and tops[1] == bots[2]:
            return fmt(bots[0], tops[2])
        else:
            raise NotImplementedError("3 non-continuous level ranges")
    else:
        raise NotImplementedError(f"{n} sets of levels")


def format_integr_period(
    start: datetime,
    now: datetime,
    setup: Setup,
    words: TranslatedWords,
    cap: bool = False,
) -> str:
    if not setup.core.integrate:
        operation = words["averaged_over"].s
    elif setup.core.input_variable == "concentration":
        operation = words["summed_over"].s
    elif setup.core.input_variable == "deposition":
        operation = words["accumulated_over"].s
    period = now - start
    hours = int(period.total_seconds() / 3600)
    minutes = int((period.total_seconds() / 60) % 60)
    s = f"{operation} {hours:d}:{minutes:02d}$\\,$h"
    if cap:
        s = s[0].upper() + s[1:]
    return s


def format_coord_label(direction: str, words: TranslatedWords, symbols: Words) -> str:
    deg_unit = f"{symbols['deg']}{symbols['short_space']}"
    min_unit = f"'{symbols['short_space']}"
    dir_unit = words[direction, "abbr"]
    if direction == "north":
        deg_dir_unit = words["degN"]
    elif direction == "east":
        deg_dir_unit = words["degE"]
    else:
        raise NotImplementedError("unit for direction", direction)
    return f"{{d}}{deg_unit}{{m}}{min_unit}{dir_unit} ({{f:.4f}}{deg_dir_unit})"
