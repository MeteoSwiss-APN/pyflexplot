# pylint: disable=C0302  # too-many-lines (>1000)
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
from datetime import timedelta
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
from pyflexplot.utils.datetime import init_datetime
from srutils.dict import recursive_update
from srutils.geo import Degrees
from srutils.plotting import truncate_cmap

# Local
from . import __version__
from .data import Field
from .data import FieldAllNaNError
from .meta_data import format_meta_datum
from .meta_data import MetaData
from .plot_layouts import BoxedPlotLayoutDeterministic
from .plot_layouts import BoxedPlotLayoutEnsemble
from .plot_layouts import BoxedPlotLayoutType
from .plotting.boxed_plot import BoxedPlot
from .plotting.boxed_plot import BoxedPlotConfig
from .plotting.boxed_plot import DummyBoxedPlot
from .plotting.boxed_plot import FontSizes
from .plotting.map_axes import CloudDomain
from .plotting.map_axes import Domain
from .plotting.map_axes import MapAxes
from .plotting.map_axes import MapAxesConfig
from .plotting.map_axes import ReleaseSiteDomain
from .plotting.text_box_axes import TextBoxAxes
from .setup import FilePathFormatter
from .setup import Setup
from .setup import SetupCollection
from .utils.formatting import escape_format_keys
from .utils.formatting import format_level_ranges
from .utils.formatting import format_range
from .utils.logging import log
from .utils.typing import ColorType
from .words import SYMBOLS
from .words import TranslatedWords
from .words import WORDS
from .words import Words


def prepare_plot(
    field_lst: Sequence[Field],
    prev_out_file_paths: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Tuple[List[str], Union[BoxedPlot, DummyBoxedPlot]]:
    """Create plots while yielding them with the plot file path one by one."""
    log(dbg=f"preparing setups for plot based on {len(field_lst)} fields")

    # Merge setups across Field objects
    var_setups_lst = [field.var_setups for field in field_lst]
    setup = SetupCollection.merge(var_setups_lst).compress()

    # Merge `nc_meta_data` across Field objects
    nc_meta_data: Dict[str, Any] = {}
    for field in field_lst:
        recursive_update(nc_meta_data, field.nc_meta_data, inplace=True)

    out_file_paths: List[str] = []
    for out_file_template in (
        [setup.outfile] if isinstance(setup.outfile, str) else setup.outfile
    ):
        out_file_path = FilePathFormatter(prev_out_file_paths).format(
            out_file_template, setup, nc_meta_data
        )
        log(dbg=f"preparing plot {out_file_path}")
        out_file_paths.append(out_file_path)
    if dry_run:
        return out_file_paths, DummyBoxedPlot()
    else:
        # SR_TMP <  SR_MULTIPANEL
        if len(field_lst) > 1:
            raise NotImplementedError("multipanel plot")
        field = next(iter(field_lst))
        map_config = create_map_config(field)
        # SR_TMP >  SR_MULTIPANEL
        config = create_plot_config(setup, WORDS, SYMBOLS, cast(MetaData, field.mdata))
        return out_file_paths, BoxedPlot(field, config, map_config)


def create_plot(
    plot: BoxedPlot,
    out_file_paths: Sequence[str],
    write: bool = True,
    show_version: bool = True,
) -> None:
    layout: BoxedPlotLayoutType
    if plot.config.setup.get_simulation_type() == "deterministic":
        layout = BoxedPlotLayoutDeterministic(aspect=plot.config.fig_aspect)
    elif plot.config.setup.get_simulation_type() == "ensemble":
        layout = BoxedPlotLayoutEnsemble(aspect=plot.config.fig_aspect)
    else:
        raise NotImplementedError(
            f"simulation type '{plot.config.setup.get_simulation_type()}'"
        )
    axs_map = plot.add_map_plot(layout.rect_center())
    plot_add_text_boxes(plot, layout, show_version)
    plot_add_markers(plot, axs_map)
    for out_file_path in out_file_paths:
        log(dbg=f"creating plot {out_file_path}")
        if write:
            plot.write(out_file_path)
        log(dbg=f"created plot {out_file_path}")
    plot.clean()


# SR_TMP <<< TODO Clean up nested functions! Eventually introduce class(es) of some kind
# pylint: disable=R0915  # too-many-statements
def plot_add_text_boxes(
    plot: BoxedPlot, layout: BoxedPlotLayoutType, show_version: bool = True
) -> None:
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
        labels = plot.config.labels["data_info"]
        box.text_block_hfill(
            labels["lines"],
            dy_unit=-1.0,
            dy_line=3.0,
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.content_medium,
        )

    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=R0913  # too-many-arguments
    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R0915  # too-many-statements
    def fill_box_legend(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the box containing the plot legend."""

        labels = plot.config.labels["legend"]
        mdata = plot.config.mdata

        # Box title
        box.text(
            labels["title"],
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
            box.text(
                s=format_max_marker_label(labels, plot.field.fld),
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
            box.text(
                s=f"{labels['site']}: {format_meta_datum(mdata.release.site)}",
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

        labels = plot.config.labels["release_info"]
        mdata = plot.config.mdata

        # Box title
        box.text(
            s=labels["title"],
            loc="tc",
            fontname=plot.config.font_name,
            size=plot.config.font_sizes.title_small,
        )

        # Release site coordinates
        lat = Degrees(mdata.release.lat)
        lon = Degrees(mdata.release.lon)
        lat_deg = labels["lat_deg_fmt"].format(d=lat.degs(), m=lat.mins(), f=lat.frac())
        lon_deg = labels["lon_deg_fmt"].format(d=lon.degs(), m=lon.mins(), f=lon.frac())

        height = format_meta_datum(mdata.release.height, mdata.release.height_unit)
        height = height.replace("meters", labels["height_unit"])  # SR_TMP
        rate = format_meta_datum(mdata.release.rate, mdata.release.rate_unit)
        mass = format_meta_datum(mdata.release.mass, mdata.release.mass_unit)
        substance = format_meta_datum(mdata.species.name, join_values=" / ")
        half_life = format_meta_datum(
            mdata.species.half_life, mdata.species.half_life_unit
        )
        deposit_vel = format_meta_datum(
            mdata.species.deposition_velocity, mdata.species.deposition_velocity_unit
        )
        sediment_vel = format_meta_datum(
            mdata.species.sedimentation_velocity,
            mdata.species.sedimentation_velocity_unit,
        )
        washout_coeff = format_meta_datum(
            mdata.species.washout_coefficient, mdata.species.washout_coefficient_unit
        )
        washout_exponent = format_meta_datum(mdata.species.washout_exponent)

        # SR_TMP <
        release_start = format_meta_datum(
            cast(datetime, mdata.simulation.start)
            + cast(timedelta, mdata.release.start_rel)
        )
        release_end = format_meta_datum(
            cast(datetime, mdata.simulation.start)
            + cast(timedelta, mdata.release.end_rel)
        )
        # SR_TMP >
        info_blocks = dedent(
            f"""\
            {labels['site']}:\t{format_meta_datum(mdata.release.site)}
            {labels['latitude']}:\t{lat_deg}
            {labels['longitude']}:\t{lon_deg}
            {labels['height']}:\t{height}

            {labels['start']}:\t{release_start}
            {labels['end']}:\t{release_end}
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

    def fill_box_bottom_left(box: TextBoxAxes, plot: BoxedPlot) -> None:
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

    def fill_box_bottom_right(box: TextBoxAxes, plot: BoxedPlot) -> None:
        labels = plot.config.labels["bottom"]
        if show_version:
            box.text(
                s=f"v{__version__}",
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
    plot.add_text_box(
        "bottom_left", layout.rect_bottom_left(), fill_box_bottom_left, frame_on=False
    )
    plot.add_text_box(
        "bottom_right",
        layout.rect_bottom_right(),
        fill_box_bottom_right,
        frame_on=False,
    )


def plot_add_markers(plot: BoxedPlot, axs_map: MapAxes) -> None:
    config = plot.config
    mdata = config.mdata
    if config.mark_release_site:
        assert config.markers is not None  # mypy
        axs_map.add_marker(
            p_lat=mdata.release.lat, p_lon=mdata.release.lon, **config.markers["site"],
        )
    if config.mark_field_max:
        assert config.markers is not None  # mypy
        try:
            max_lat, max_lon = plot.field.locate_max()
        except FieldAllNaNError:
            warnings.warn("skip maximum marker (all-nan field)")
        else:
            axs_map.add_marker(p_lat=max_lat, p_lon=max_lon, **config.markers["max"])


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0914  # too-many-locals
# pylint: disable=R0915  # too-many-statements
def create_map_config(field: Field) -> MapAxesConfig:
    domain_type = field.var_setups.collect_equal("domain")
    model_name = field.var_setups.collect_equal("model")
    scale_fact = field.var_setups.collect_equal("scale_fact")

    conf_global_scale: Dict[str, Any] = {
        "geo_res": "110m",
        "geo_res_cities": "110m",
        "geo_res_rivers": "110m",
        "min_city_pop": 1_000_000,
        # Note: Ref dist indicator seems not to work properly at global scale!
        "ref_dist_on": False,
    }
    conf_continental_scale: Dict[str, Any] = {
        "geo_res": "50m",
        "geo_res_cities": "110m",
        "geo_res_rivers": "110m",
        "min_city_pop": 1_000_000,
        "ref_dist_config": {"dist": 250},
    }
    conf_regional_scale: Dict[str, Any] = {
        "geo_res": "10m",
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 300_000,
        "ref_dist_config": {"dist": 100},
    }
    conf_country_scale: Dict[str, Any] = {
        "geo_res": "10m",
        "geo_res_cities": "10m",
        "geo_res_rivers": "10m",
        "min_city_pop": 0,
        "ref_dist_config": {"dist": 25},
    }

    # Derive configuration of map axes from domain and model
    map_axes_config: Optional[Dict[str, Any]] = None
    if domain_type == "full":
        if model_name.startswith("COSMO"):
            map_axes_config = conf_regional_scale
        elif model_name == "IFS-HRES-EU":
            map_axes_config = conf_continental_scale
        elif model_name == "IFS-HRES":
            map_axes_config = conf_global_scale
    elif domain_type in ["release_site", "cloud", "alps"]:
        map_axes_config = conf_regional_scale
    elif domain_type == "ch":
        map_axes_config = conf_country_scale
    if map_axes_config is None:
        raise NotImplementedError(
            f"map axes config for model '{model_name}' and domain type '{domain_type}'"
        )

    # Set scaling factor of plot elements
    map_axes_config["scale_fact"] = scale_fact

    # Initialize domain (projection and extent)
    domain: Optional[Domain] = None
    if domain_type == "full":
        if model_name.startswith("COSMO"):
            domain = Domain(field, zoom_fact=1.01)
        else:
            domain = Domain(field)
    elif domain_type == "release_site":
        domain = ReleaseSiteDomain(field)
    elif domain_type == "alps":
        if model_name == "IFS-HRES-EU":
            domain = Domain(field, zoom_fact=3.4, rel_offset=(-0.165, -0.11))
    elif domain_type == "cloud":
        domain = CloudDomain(field, zoom_fact=0.9)
    elif domain_type == "ch":
        if model_name.startswith("COSMO-1"):
            domain = Domain(field, zoom_fact=3.6, rel_offset=(-0.02, 0.045))
        elif model_name.startswith("COSMO-2"):
            domain = Domain(field, zoom_fact=3.23, rel_offset=(0.037, 0.1065))
        elif model_name == "IFS-HRES-EU":
            domain = Domain(field, zoom_fact=11.0, rel_offset=(-0.18, -0.11))
    if domain is None:
        raise NotImplementedError(
            f"domain config for model '{model_name}' and domain type '{domain_type}'"
        )

    return MapAxesConfig(
        lang=field.var_setups.collect_equal("lang"), domain=domain, **map_axes_config,
    )


# SR_TODO Create dataclass with default values for text box setup
# pylint: disable=R0912  # too-many-branches
def create_plot_config(
    setup: Setup, words: TranslatedWords, symbols: Words, mdata: MetaData,
) -> BoxedPlotConfig:
    words.set_active_lang(setup.core.lang)
    new_config_dct: Dict[str, Any] = {
        "setup": setup,
        "mdata": mdata,
        "fig_size": (12.5 * setup.scale_fact, 8.0 * setup.scale_fact),
        "font_sizes": FontSizes().scale(setup.scale_fact),
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
        if setup.core.ens_variable in ["minimum", "maximum", "median", "mean"]:
            pass
        else:
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
            if setup.core.ens_variable.endswith("probability"):
                new_config_dct.update(
                    {
                        "n_levels": 9,
                        "d_level": 10,
                        "cmap": truncate_cmap("terrain_r", 0.05),
                        "extend": "max",
                    }
                )
            elif setup.core.ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                new_config_dct.update({"n_levels": 8, "d_level": 3})
                if setup.core.ens_variable == "cloud_arrival_time":
                    new_config_dct.update(
                        {
                            "cmap": "viridis",
                            "color_under": "slategray",
                            "color_over": "lightgray",
                        }
                    )
                if setup.core.ens_variable == "cloud_departure_time":
                    new_config_dct.update(
                        {
                            "cmap": "viridis_r",
                            "color_under": "lightgray",
                            "color_over": "slategray",
                        }
                    )

    # Colors
    n_levels = new_config_dct["n_levels"]
    extend = new_config_dct.get("extend", "max")
    cmap = new_config_dct.get("cmap", "flexplot")
    if setup.core.plot_variable == "affected_area_mono":
        new_config_dct["colors"] = (np.array([(200, 200, 200)]) / 255).tolist()
    elif cmap == "flexplot":
        new_config_dct["colors"] = colors_flexplot(n_levels, extend)
        new_config_dct["color_above_closed"] = "black"
    else:
        cmap = mpl.cm.get_cmap(cmap)
        color_under = new_config_dct.get("color_under")
        color_over = new_config_dct.get("color_over")
        n_colors = n_levels - 1
        if extend in ["min", "both"] and not color_under:
            n_colors += 1
        if extend in ["max", "both"] and not color_over:
            n_colors += 1
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        if extend in ["min", "both"] and color_under:
            colors.insert(0, color_under)
        if extend in ["max", "both"] and color_over:
            colors.append(color_over)
        new_config_dct["colors"] = colors

    # Markers
    new_config_dct["markers"] = {
        "max": {
            "marker": "+",
            "color": "black",
            "markersize": 10 * setup.scale_fact,
            "markeredgewidth": 1.5 * setup.scale_fact,
        },
        "site": {
            "marker": "^",
            "markeredgecolor": "red",
            "markerfacecolor": "white",
            "markersize": 7.5 * setup.scale_fact,
            "markeredgewidth": 1.5 * setup.scale_fact,
        },
    }

    new_config_dct["labels"] = create_box_labels(setup, words, symbols, mdata)

    return BoxedPlotConfig(**new_config_dct)


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0914  # too-many-locals
# pylint: disable=R0915  # too-many-statements
def create_box_labels(
    setup: Setup, words: TranslatedWords, symbols: Words, mdata: MetaData,
) -> Dict[str, Dict[str, Any]]:

    # SR_TMP <
    names = format_names_etc(setup, words, mdata)
    short_name = names["short"]
    long_name = names["long"]
    var_name_abbr = names["var_abbr"]
    ens_var_name = names["ens_var"]
    unit = names["unit"]
    # SR_TMP >

    labels: Dict[str, Dict[str, Any]] = {}
    integr_period = format_integr_period(
        mdata.simulation.reduction_start, mdata.simulation.now, setup, words, cap=True
    )
    # SR_TMP < TODO move to mdata.simulation
    time_since_release_start = (
        mdata.simulation.now - mdata.simulation.start - mdata.release.start_rel
    )
    lead_time = mdata.simulation.now - init_datetime(cast(int, setup.base_time))
    # SR_TMP >
    labels["top"] = {
        "tl": "",  # SR_TMP force into 1st position of dict (for tests)
        "bl": (
            f"{integr_period} ({words['since']}"
            f" {format_meta_datum(mdata.simulation.reduction_start)})"
        ),
        "tr": (
            f"{format_meta_datum(mdata.simulation.now)}"
            f" ({words['lead_time']} +{format_meta_datum(lead_time)})"
        ),
        "br": (
            f"{format_meta_datum(time_since_release_start)} {words['since']}"
            f" {words['release_start']}"
            # f" {words['at', 'place']} {format_meta_datum(mdata.release.site)}"
        ),
    }
    labels["title_2nd"] = {
        "tc": f"{format_meta_datum(mdata.species.name)}",
        "bc": f"{format_meta_datum(mdata.release.site)}",
    }
    labels["data_info"] = {
        "lines": [],
    }
    labels["legend"] = {
        "title": "",  # SR_TMP Force into 2nd position in dict (for tests)
        "release_site": words["release_site"].s,
        "site": words["site"].s,
        "max": words["maximum", "abbr"].s,
        "maximum": words["maximum"].s,
    }
    labels["release_info"] = {
        "title": words["release"].t,
        "start": words["start"].s,
        "end": words["end"].s,
        "latitude": words["latitude"].s,
        "longitude": words["longitude"].s,
        "lat_deg_fmt": format_coord_label("north", words, symbols),
        "lon_deg_fmt": format_coord_label("east", words, symbols),
        "height": words["height"].s,
        # SR_TMP <
        "height_unit": r"$\,$" + words["m_agl"].s,
        # SR_TMP >
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
        "model_info": format_model_info(setup, words),
        "copyright": f"{symbols['copyright']}{words['meteoswiss']}",
    }

    if setup.core.input_variable == "concentration":
        labels["legend"]["tc"] = (
            f"{words['height']}:"
            f" {escape_format_keys(format_level_label(mdata, words))}"
        )

    labels["data_info"]["lines"].append(
        f"{words['substance'].c}:"
        f"\t{format_meta_datum(mdata.species.name, join_values=' / ')}",
    )
    labels["data_info"]["lines"].append(
        f"{words['input_variable'].c}:\t{capitalize(var_name_abbr)}"
    )
    if setup.get_simulation_type() == "ensemble":
        # SR_TMP <
        if setup.core.ens_variable == "cloud_occurrence_probability":
            labels["data_info"]["lines"].append(
                f"{words['ensemble_variable', 'abbr']}:"
                f"\t{words['cloud_probability', 'abbr']}"
            )
        else:
            labels["data_info"]["lines"].append(
                f"{words['ensemble_variable', 'abbr']}:\t{ens_var_name}"
            )
        # SR_TMP >
        if setup.core.ens_variable == "probability":
            labels["data_info"]["lines"].append(
                f"{words['threshold']}:\t{symbols['geq']} {setup.core.ens_param_thr}"
                f" {format_meta_datum(unit=format_meta_datum(mdata.variable.unit))}"
            )
        elif setup.core.ens_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
            "cloud_occurrence_probability",
        ]:
            labels["data_info"]["lines"].append(
                # f"{words['cloud_density']}:\t{words['minimum', 'abbr']}"
                # f"{words['threshold']}:\t"
                f"{words['cloud_threshold', 'abbr']}:\t"
                f" {setup.core.ens_param_thr}"
                f" {format_meta_datum(unit=format_meta_datum(mdata.variable.unit))}"
            )
            n_min = setup.core.ens_param_mem_min or 0
            n_tot = len((setup.ens_member_id or []))
            labels["data_info"]["lines"].append(
                # f"{words['number_of', 'abbr'].c} {words['member', 'pl']}:"
                # f"\t {n_min}"
                f"{words['minimum', 'abbr'].c} {words['member', 'pl']}:"
                f"\t{n_min}"
                r"$\,/\,$"
                f"{n_tot} ({n_min/(n_tot or 0.001):.0%})"
            )
            if setup.core.ens_variable == "cloud_occurrence_probability":
                labels["data_info"]["lines"].append(
                    f"{words['time_window']}:\t{setup.core.ens_param_time_win}"
                    # SR_TMP < TODO Properly implement in hours (now steps, I think)!
                    # f" {words['hour', 'abbr']}"
                    "???"
                    # SR_TMP >
                )
    if setup.core.input_variable == "concentration":
        labels["data_info"]["lines"].append(
            f"{words['height'].c}:"
            f"\t{escape_format_keys(format_level_label(mdata, words))}"
        )

    # box_labels["top"]["tl"] = (
    #     f"{long_name} {words['of']}"
    #     f" {format_meta_datum(mdata.species.name)}"
    # )
    labels["top"]["tl"] = long_name
    if setup.get_simulation_type() == "deterministic":
        labels["legend"]["title"] = f"{short_name} ({unit})"
    elif setup.get_simulation_type() == "ensemble":
        if setup.core.ens_variable == "cloud_arrival_time":
            labels["legend"][
                "title"
            ] = f"{words['hour', 'pl']} {words['until']} {words['arrival']}"
        elif setup.core.ens_variable == "cloud_departure_time":
            labels["legend"][
                "title"
            ] = f"{words['hour', 'pl']} {words['until']} {words['departure']}"
        else:
            # SR_TMP <
            if unit == "%":
                labels["legend"]["title"] = f"{words['probability']} ({unit})"
            else:
                labels["legend"]["title"] = f"{unit}"

            # SR_TMP >
    labels["legend"]["unit"] = unit

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

    def format_var_names(setup: Setup, words: TranslatedWords) -> Tuple[str, str, str]:
        if setup.core.input_variable == "concentration":
            var_name = str(words["activity_concentration"])
            var_name_abbr = str(words["activity_concentration", "abbr"])
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
            var_name_abbr = (
                f"{words[dep_type_word, 'f']} {words['surface_deposition', 'abbr']}"
            )
            var_name_rel = (
                f"{words['of', 'fg']} {words[dep_type_word, 'g']}"
                f" {words['surface_deposition']}"
            )
        else:
            raise NotImplementedError(f"input variable '{setup.core.input_variable}'")
        return var_name, var_name_abbr, var_name_rel

    # pylint: disable=W0621  # redefined-outer-name
    def _format_unit(setup: Setup, words: TranslatedWords, mdata: MetaData) -> str:
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
        return format_meta_datum(unit=format_meta_datum(mdata.variable.unit))

    unit = _format_unit(setup, words, mdata)
    var_name, var_name_abbr, var_name_rel = format_var_names(setup, words)

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

    # Short/long names #2: By plot variable/type; ensemble variable name
    ens_var_name = "none"
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
        if ens_var_name == "none":
            ens_var_name = f"{words[setup.core.ens_variable].c}"

    # SR_TMP <
    if not short_name:
        raise Exception("no short name")
    if not long_name:
        raise Exception("no long name")
    # SR_TMP >

    return {
        "short": short_name,
        "long": long_name,
        "var": var_name,
        "var_abbr": var_name_abbr,
        "var_rel": var_name_rel,
        "ens_var": ens_var_name,
        "unit": unit,
    }


def capitalize(s: str) -> str:
    """Capitalize the first letter while leaving all others as they are."""
    if not s:
        return s
    try:
        return s[0].upper() + s[1:]
    except Exception:
        raise ValueError(f"string not capitalizable: '{s}'")


def format_max_marker_label(labels: Dict[str, Any], fld: np.ndarray) -> str:
    if np.isnan(fld).all():
        s_val = "NaN"
    else:
        fld_max = np.nanmax(fld)
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
    return f"{labels['max']}: {s_val}"
    # return f"{labels['max']} ({s_val})"
    # return f"{labels['maximum']}:\n({s_val})"


def format_model_info(setup: Setup, words: TranslatedWords) -> str:
    model_name = setup.model
    model_info = None
    if setup.get_simulation_type() == "deterministic":
        if setup.model in ["COSMO-1", "COSMO-2", "IFS-HRES", "IFS-HRES-EU"]:
            model_info = model_name
        elif setup.model in ["COSMO-1E", "COSMO-2E"]:
            model_info = f"{model_name} {words['control_run']}"
        else:
            raise NotImplementedError(f"model '{setup.model}'")
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
    assert setup.base_time is not None  # mypy
    base_time = init_datetime(setup.base_time)
    return (
        f"{words['flexpart']} {words['based_on']} {model_info}"
        f", {format_meta_datum(base_time)}"
    )


def format_level_label(mdata: MetaData, words: TranslatedWords) -> str:
    unit = mdata.variable.level_unit
    if unit == "meters":
        unit = words["m_agl"].s
    level = format_vertical_level_range(
        mdata.variable.bottom_level, mdata.variable.top_level, unit
    )
    if not level:
        return ""
    return f"{format_meta_datum(unit=level)}"


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
    else:
        raise NotImplementedError(
            f"operation for {'' if setup.core.integrate else 'non-'}integrated"
            f" input variable '{setup.core.input_variable}'"
        )
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


def colors_from_cmap(cmap, n_levels, extend):
    """Get colors from cmap for given no. levels and extend param."""
    colors = cmap(np.linspace(0, 1, n_levels + 1))
    if extend not in ["min", "both"]:
        colors.pop(0)
    if extend not in ["max", "both"]:
        colors.pop(-1)
    return colors
