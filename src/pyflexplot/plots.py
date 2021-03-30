# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot types.

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
import os
from datetime import datetime
from datetime import timedelta
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
from matplotlib.colors import Colormap

# First-party
from srutils.datetime import init_datetime
from srutils.format import format_numbers_range
from srutils.geo import Degrees
from srutils.plotting import truncate_cmap
from words import Word

# Local
from . import __version__
from .input.field import Field
from .input.field import FieldGroup
from .input.field import FieldStats
from .input.meta_data import format_meta_datum
from .input.meta_data import MetaData
from .output import FilePathFormatter
from .plot_layouts import BoxedPlotLayout
from .plotting.boxed_plot import BoxedPlot
from .plotting.boxed_plot import BoxedPlotConfig
from .plotting.boxed_plot import ContourLevelsConfig
from .plotting.boxed_plot import ContourLevelsLegendConfig
from .plotting.boxed_plot import FontConfig
from .plotting.boxed_plot import FontSizes
from .plotting.boxed_plot import MarkersConfig
from .plotting.map_axes import MapAxesConfig
from .plotting.text_box_axes import TextBoxAxes
from .setups.model_setup import ModelSetup
from .setups.plot_panel_setup import PlotPanelSetup
from .setups.plot_setup import PlotSetup
from .utils.formatting import escape_format_keys
from .utils.formatting import format_level_ranges
from .utils.logging import log
from .utils.typing import ColorType
from .utils.typing import RectType
from .words import SYMBOLS
from .words import TranslatedWords
from .words import WORDS
from .words import Words


def format_out_file_paths(
    field_group: FieldGroup, prev_paths: List[str], dest_dir: Optional[str] = None
) -> List[str]:
    plot_setup = field_group.plot_setup
    # SR_TMP <
    mdata = next(iter(field_group)).mdata
    for field in field_group:
        if field.mdata != mdata:
            raise NotImplementedError(
                f"meta data differ between fields:\n{field.mdata}\n!=\n{mdata}"
            )
    release_mdata = mdata.release
    simulation_mdata = mdata.simulation
    # SR_TMP >
    out_file_templates: Sequence[str] = (
        [plot_setup.outfile]
        if isinstance(plot_setup.outfile, str)
        else plot_setup.outfile
    )
    out_file_paths: List[str] = []
    for out_file_template in out_file_templates:
        if dest_dir:
            if out_file_template.startswith("/"):
                raise Exception(
                    "passing a dest_dir ('{dest_dir}') is incompatible with absolute"
                    " plot paths ('{out_file_template}')"
                )
            out_file_template = os.path.relpath(
                os.path.abspath(f"{dest_dir}/{out_file_template}")
            )
        out_file_path = FilePathFormatter(prev_paths).format(
            out_file_template,
            plot_setup,
            release_site=release_mdata.raw_site_name,
            release_start=simulation_mdata.start + release_mdata.start_rel,
            time_steps=tuple(simulation_mdata.time_steps),
        )
        log(dbg=f"preparing plot '{out_file_path}'")
        out_file_paths.append(out_file_path)
    return out_file_paths


# pylint: disable=R0914  # too-many-locals (>15)
def create_plot(
    field_group: FieldGroup,
    file_paths: Sequence[str],
    *,
    write: bool = True,
    show_version: bool = True,
) -> BoxedPlot:
    plot_setup = field_group.plot_setup
    # SR_TMP <  SR_MULTIPANEL
    mdata = next(iter(field_group)).mdata
    for field in field_group:
        if field.mdata != mdata:
            raise NotImplementedError(
                f"meta data differ between fields:\n{field.mdata}\n!=\n{mdata}"
            )
    if len(field_group) > 1:
        print("warning: using time_props of first panel of multipanel plot")
    time_stats = next(iter(field_group)).time_props.stats
    # SR_TMP >  SR_MULTIPANEL
    labels = create_box_labels(plot_setup, mdata)
    plot_config = create_plot_config(
        setup=plot_setup, time_stats=time_stats, labels=labels
    )
    map_configs: List[MapAxesConfig] = [
        create_map_config(
            field_group.plot_setup,
            field.panel_setup,
            plot_config.layout.get_aspect("center"),
        )
        for field in field_group
    ]
    plot = BoxedPlot(plot_config)
    # SR_TMP <
    if len(field_group) == 1:
        field = next(iter(field_group))
        map_config = next(iter(map_configs))
        rect = plot.config.layout.get_rect("center")
        plot.add_map_plot("map", field, map_config, rect)
    elif len(field_group) != 4:
        raise NotImplementedError(f"{len(field_group)} number of panels")
    else:
        # SR_TODO Define sub-rects in layout (new layout type with four panels)
        def derive_sub_rects(
            rect: RectType,
        ) -> Tuple[RectType, RectType, RectType, RectType]:
            x0, y0, w, h = rect
            rel_pad = 0.025
            w_pad = rel_pad * w
            h_pad = w_pad  # SR_TMP
            w = 0.5 * w - 0.5 * w_pad
            h = 0.5 * h - 0.5 * h_pad
            x1 = x0 + w + w_pad
            y1 = y0 + h + h_pad
            return ((x0, y1, w, h), (x1, y1, w, h), (x0, y0, w, h), (x1, y0, w, h))

        sub_rects = derive_sub_rects(plot.config.layout.get_rect("center"))
        for idx, (field, map_config, sub_rect) in enumerate(
            zip(field_group, map_configs, sub_rects)
        ):
            name = f"map{idx}"
            plot.add_map_plot(name, field, map_config, sub_rect)
    # SR_TMP >

    plot_add_text_boxes(plot, field_group, plot.config.layout, show_version)
    for file_path in file_paths:
        log(dbg=f"creating plot {file_path}")
        if write:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            log(dbg=f"writing plot {file_path}")
            plot.write(file_path)
        log(dbg=f"created plot {file_path}")
    plot.clean()
    return plot


# SR_TMP <<< TODO Clean up nested functions! Eventually introduce class(es) of some kind
# pylint: disable=R0915  # too-many-statements
def plot_add_text_boxes(
    plot: BoxedPlot,
    fields: FieldGroup,
    layout: BoxedPlotLayout,
    show_version: bool = True,
) -> None:
    # pylint: disable=R0915  # too-many-statements
    def fill_box_title(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the title box."""
        labels = plot.config.labels["title"]
        for position, label in labels.items():
            if "position" == "tl":
                size = plot.config.font.sizes.title_medium
            else:
                size = plot.config.font.sizes.content_large
            box.text(
                label,
                loc=position,
                fontname=plot.config.font.name,
                size=size,
            )

    def fill_box_2nd_title(box: TextBoxAxes, plot: BoxedPlot, field: Field) -> None:
        """Fill the secondary title box of the deterministic plot layout."""
        font_size = plot.config.font.sizes.content_large
        mdata = field.mdata
        box.text(
            capitalize(format_meta_datum(mdata.species.name)),
            loc="tc",
            fontname=plot.config.font.name,
            size=font_size,
        )
        box.text(
            capitalize(format_meta_datum(mdata.release.site_name)),
            loc="bc",
            fontname=plot.config.font.name,
            size=font_size,
        )

    # pylint: disable=R0915  # too-many-statements
    def fill_box_data_info(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the data information box of the ensemble plot layout."""
        labels = plot.config.labels["data_info"]
        box.text_block_hfill(
            labels["lines"],
            dy_unit=-1.0,
            dy_line=3.0,
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.content_medium,
        )

    # pylint: disable=R0912  # too-many-branches
    # pylint: disable=R0913  # too-many-arguments
    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R0915  # too-many-statements
    def fill_box_legend(box: TextBoxAxes, plot: BoxedPlot, field: Field) -> None:
        """Fill the box containing the plot legend."""
        labels = plot.config.labels["legend"]
        mdata = field.mdata

        # Box title
        box.text(
            labels["title"],
            loc="tc",
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.title_small,
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
        legend_labels = plot.config.levels.legend.labels

        # Legend labels (level ranges)
        box.text_block(
            legend_labels[::-1],
            loc="tc",
            dy_unit=dy0_labels,
            dy_line=dy_line,
            dx=dx_legend_label,
            ha="left",
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.content_medium,
            family="monospace",
        )

        # Legend color boxes
        colors = plot.config.colors
        assert colors is not None  # SR_TMP
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
        if plot.config.markers.mark_field_max:
            dy_marker_label_max = dy0_marker
            dy0_marker -= dy_line
            dy_max_marker = dy_marker_label_max - 0.7
            assert plot.config.markers.markers is not None  # mypy
            box.marker(
                loc="tc",
                dx=dx_marker,
                dy=dy_max_marker,
                **plot.config.markers.markers["max"],
            )
            box.text(
                s=format_max_marker_label(labels, field.fld),
                loc="tc",
                dx=dx_marker_label,
                dy=dy_marker_label_max,
                ha="left",
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.content_medium,
            )

        # Release site marker
        if plot.config.markers.mark_release_site:
            dy_site_label = dy0_marker
            dy0_marker -= dy_line
            dy_site_marker = dy_site_label - 0.7
            assert plot.config.markers.markers is not None  # mypy
            box.marker(
                loc="tc",
                dx=dx_marker,
                dy=dy_site_marker,
                **plot.config.markers.markers["site"],
            )
            box.text(
                s=f"{labels['site']}: {format_meta_datum(mdata.release.site_name)}",
                loc="tc",
                dx=dx_marker_label,
                dy=dy_site_label,
                ha="left",
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.content_medium,
            )

    # pylint: disable=R0915  # too-many-statements
    def fill_box_release_info(box: TextBoxAxes, plot: BoxedPlot) -> None:
        """Fill the box containing the release info."""
        labels = plot.config.labels["release_info"]

        # Box title
        box.text(
            s=labels["title"],
            loc="tc",
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.title_small,
        )

        # Add lines bottom-up (to take advantage of baseline alignment)
        box.text_blocks_hfill(
            labels["lines_str"],
            dy_unit=-4.0,
            dy_line=2.5,
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.content_small,
        )

    def fill_box_bottom_left(box: TextBoxAxes, plot: BoxedPlot) -> None:
        labels = plot.config.labels["bottom"]
        # FLEXPART/model info
        box.text(
            s=labels["model_info"],
            loc="tl",
            dx=-0.7,
            dy=0.5,
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.content_small,
        )

    def fill_box_bottom_right(box: TextBoxAxes, plot: BoxedPlot) -> None:
        labels = plot.config.labels["bottom"]
        if show_version:
            box.text(
                s=f"v{__version__}",
                loc="tl",
                dx=-0.7,
                dy=0.5,
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.content_small,
            )
        # MeteoSwiss Copyright
        box.text(
            s=labels["copyright"],
            loc="tr",
            dx=0.7,
            dy=0.5,
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.content_small,
        )

    # SR_TMP <
    if len(fields) > 1:
        print(
            "warning: plot_add_text_boxes: selecting field of first of multiple panels"
        )
    field = next(iter(fields))
    # SR_TMP >

    plot.add_text_box("top", layout.get_rect("top"), fill_box_title)
    if layout.name == "post_vintage":
        plot.add_text_box(
            "right_top",
            layout.get_rect("right_top"),
            lambda box, plot: fill_box_2nd_title(box, plot, field),
        )
    elif layout.name == "post_vintage_ens":
        plot.add_text_box(
            "right_top",
            layout.get_rect("right_top"),
            fill_box_data_info,
        )
    plot.add_text_box(
        "right_middle",
        layout.get_rect("right_middle"),
        lambda box, plot: fill_box_legend(box, plot, field),
    )
    plot.add_text_box(
        "right_bottom", layout.get_rect("right_bottom"), fill_box_release_info
    )
    plot.add_text_box(
        "bottom_left",
        layout.get_rect("bottom_left"),
        fill_box_bottom_left,
        frame_on=False,
    )
    plot.add_text_box(
        "bottom_right",
        layout.get_rect("bottom_right"),
        fill_box_bottom_right,
        frame_on=False,
    )


def create_map_config(
    plot_setup: PlotSetup, panel_setup: PlotPanelSetup, aspect: float
) -> MapAxesConfig:
    model_name = plot_setup.model.name
    scale_fact = plot_setup.scale_fact
    domain_type = panel_setup.domain
    lang = panel_setup.lang

    config_dct: Dict[str, Any] = {
        "aspect": aspect,
        "lang": lang,
        "scale_fact": scale_fact,
    }
    conf_continental_scale: Dict[str, Any] = {
        "geo_res": "50m",
        "geo_res_cities": "110m",
        "geo_res_rivers": "110m",
        "min_city_pop": 1_000_000,
        "ref_dist_config": {"dist": 250},
    }
    conf_regional_scale: Dict[str, Any] = {
        "geo_res": "50m",
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
    if domain_type == "full":
        if model_name.startswith("COSMO"):
            config_dct.update(conf_regional_scale)
        elif model_name == "IFS-HRES-EU":
            config_dct.update(conf_continental_scale)
        elif model_name == "IFS-HRES":
            raise NotImplementedError("global IFS-HRES domain")
    elif domain_type in ["release_site", "cloud", "alps"]:
        config_dct.update(conf_regional_scale)
    elif domain_type == "ch":
        config_dct.update(conf_country_scale)
    else:
        raise NotImplementedError(
            f"map axes config for model '{model_name}' and domain '{domain_type}'"
        )
    return MapAxesConfig(**config_dct)


# SR_TODO Create dataclass with default values for text box setup
# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0914  # too-many-locals (>15)
def create_plot_config(
    setup: PlotSetup,
    time_stats: FieldStats,
    labels: Dict[str, Dict[str, Any]],
) -> BoxedPlotConfig:
    plot_variable = setup.panels.collect_equal("plot_variable")
    # SR_TMP <
    if setup.plot_type == "multipanel" and setup.multipanel_param == "ens_variable":
        ens_variable = "+".join(setup.panels.collect("ens_variable"))
    else:
        ens_variable = setup.panels.collect_equal("ens_variable")
    # SR_TMP >

    plot_config_dct: Dict[str, Any] = {
        "fig_size": (12.5 * setup.scale_fact, 8.0 * setup.scale_fact),
    }

    # Layout
    fig_aspect = np.divide(*plot_config_dct["fig_size"])
    plot_config_dct["layout"] = BoxedPlotLayout.create(
        setup.layout.type, aspect=fig_aspect
    )

    # Fonts
    font_config = FontConfig(sizes=FontSizes().scale(setup.scale_fact))

    # Levels and legend
    levels_config_dct: Dict[str, Any] = {
        "include_lower": False,
        "scale": "log",
    }
    legend_config_dct: Dict[str, Any] = {}
    if plot_variable == "concentration":
        levels_config_dct["n"] = 8
    elif plot_variable.endswith("deposition"):
        levels_config_dct["n"] = 9
    if plot_variable == "affected_area" and ens_variable != "probability":
        levels_config_dct["extend"] = "none"
        levels_config_dct["levels"] = np.array([0.0, np.inf])
        levels_config_dct["scale"] = "lin"
    elif setup.model.simulation_type == "ensemble" and ens_variable == "probability":
        levels_config_dct["extend"] = "max"
        levels_config_dct["scale"] = "lin"
        levels_config_dct["levels"] = np.arange(5, 95.1, 15)
        legend_config_dct["range_style"] = "up"
        legend_config_dct["range_align"] = "right"
        legend_config_dct["rstrip_zeros"] = True
    elif plot_variable in [
        "cloud_arrival_time",
        "cloud_departure_time",
    ] or ens_variable in [
        "cloud_arrival_time",
        "cloud_departure_time",
    ]:
        levels_config_dct["scale"] = "lin"
        legend_config_dct["range_style"] = "base"
        legend_config_dct["range_align"] = "right"
        legend_config_dct["range_widths"] = (4, 3, 4)
        legend_config_dct["rstrip_zeros"] = True
        # SR_TMP < Adapt to simulation duration (not always 33 h)!
        levels_config_dct["levels"] = [0, 3, 6, 9, 12, 18, 24, 33]
        # SR_TMP >
        if plot_variable == "cloud_arrival_time" or (
            ens_variable == "cloud_arrival_time"
        ):
            levels_config_dct["extend"] = "min"
        elif plot_variable == "cloud_departure_time" or (
            ens_variable == "cloud_departure_time"
        ):
            levels_config_dct["extend"] = "max"
    # SR_TMP < TODO proper multipanel support
    if setup.plot_type == "multipanel":
        if setup.multipanel_param == "ens_variable":
            print(
                "warning: create_plot_config: selecting ens_variable of first of"
                " multiple panels"
            )
            ens_variable = next(iter(setup.panels)).ens_variable
            levels_config_dct["levels"] = levels_from_time_stats(
                simulation_type=setup.model.simulation_type,
                ens_variable=ens_variable,
                time_stats=time_stats,
                levels_config_dct=levels_config_dct,
            )
        else:
            raise NotImplementedError(f"multipanel_param='{setup.multipanel_param}'")
    else:
        levels_config_dct["levels"] = levels_from_time_stats(
            simulation_type=setup.model.simulation_type,
            ens_variable=setup.panels.collect_equal("ens_variable"),
            time_stats=time_stats,
            levels_config_dct=levels_config_dct,
        )
    # SR_TMP >
    legend_config_dct["labels"] = format_level_ranges(
        levels=levels_config_dct["levels"],
        style=legend_config_dct.get("range_style", "base"),
        extend=levels_config_dct.get("extend", "max"),
        rstrip_zeros=legend_config_dct.get("rstrip_zeros", True),
        align=legend_config_dct.get("range_align", "center"),
        widths=legend_config_dct.get("range_widths"),
        include="lower" if levels_config_dct.get("include_lower", True) else "upper",
    )
    levels_config_dct["legend"] = ContourLevelsLegendConfig(**legend_config_dct)
    levels_config = ContourLevelsConfig(**levels_config_dct)

    # Colors
    extend = levels_config.extend
    cmap: Union[str, Colormap] = "flexplot"
    color_under: Optional[str] = None
    color_over: Optional[str] = None
    if plot_variable == "affected_area" and ens_variable != "probability":
        cmap = "mono"
    elif setup.model.simulation_type == "ensemble" and ens_variable == "probability":
        # cmap = truncate_cmap("nipy_spectral_r", 0.275, 0.95)
        cmap = truncate_cmap("terrain_r", 0.075)
    elif plot_variable == "cloud_arrival_time" or (
        ens_variable == "cloud_arrival_time"
    ):
        cmap = "viridis"
        color_under = "slategray"
        color_over = "lightgray"
    elif plot_variable == "cloud_departure_time" or (
        ens_variable == "cloud_departure_time"
    ):
        cmap = "viridis_r"
        color_under = "lightgray"
        color_over = "slategray"
    if cmap == "flexplot":
        n_levels = levels_config.n
        assert n_levels  # SR_TMP
        colors = colors_flexplot(n_levels, extend)
    elif cmap == "mono":
        colors = (np.array([(200, 200, 200)]) / 255).tolist()
    else:
        levels = levels_config.levels
        assert levels is not None  # mypy
        n_levels = len(levels)
        cmap = mpl.cm.get_cmap(cmap)
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
    plot_config_dct["colors"] = colors

    # Markers
    markers_config_dct: Dict[str, Any] = {}
    if setup.model.simulation_type == "deterministic":
        if plot_variable in [
            "affected_area",
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            markers_config_dct["mark_field_max"] = False
        else:
            markers_config_dct["mark_field_max"] = True
    elif setup.model.simulation_type == "ensemble":
        markers_config_dct["mark_field_max"] = False
        if ens_variable in [
            "minimum",
            "maximum",
            "median",
            "mean",
            "std_dev",
            "med_abs_dev",
            "percentile",
        ]:
            pass
        else:
            markers_config_dct["mark_field_max"] = False
    markers = {}
    markers["max"] = {
        "marker": "+",
        "color": "black",
        "markersize": 10 * setup.scale_fact,
        "markeredgewidth": 1.5 * setup.scale_fact,
    }
    markers["site"] = {
        "marker": "^",
        "markeredgecolor": "red",
        "markerfacecolor": "white",
        "markersize": 7.5 * setup.scale_fact,
        "markeredgewidth": 1.5 * setup.scale_fact,
    }
    markers_config_dct["markers"] = markers
    markers_config = MarkersConfig(**markers_config_dct)

    return BoxedPlotConfig(
        setup=setup,
        labels=labels,
        font=font_config,
        levels=levels_config,
        markers=markers_config,
        **plot_config_dct,
    )


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0914  # too-many-locals
# pylint: disable=R0915  # too-many-statements
def create_box_labels(setup: PlotSetup, mdata: MetaData) -> Dict[str, Dict[str, Any]]:
    words = WORDS
    symbols = SYMBOLS
    words.set_active_lang(setup.panels.collect_equal("lang"))

    ens_params = setup.panels.collect_equal("ens_params")
    plot_variable = setup.panels.collect_equal("plot_variable")
    # SR_TMP <
    if setup.plot_type == "multipanel" and setup.multipanel_param == "ens_variable":
        ens_variable = "+".join(setup.panels.collect("ens_variable"))
    else:
        ens_variable = setup.panels.collect_equal("ens_variable")
    # SR_TMP >

    # Format variable name in various ways
    names = format_names_etc(setup, words, mdata)
    short_name = names["short"]
    var_name_abbr = names["var_abbr"]
    ens_var_name = names["ens_var"]
    unit = names["unit"]

    labels: Dict[str, Dict[str, Any]] = {}

    # Title box
    integr_period = format_integr_period(
        mdata.simulation.reduction_start,
        mdata.simulation.now,
        setup,
        words,
        cap=True,
    )
    labels["title"] = {
        "tl": capitalize(format_names_etc(setup, words, mdata)["long"]),
        "bl": capitalize(
            f"{integr_period} ({words['since']}"
            f" {format_meta_datum(mdata.simulation.reduction_start)})"
        ),
        "tr": capitalize(
            f"{format_meta_datum(mdata.simulation.now)}"
            f" ({words['lead_time']} +{format_meta_datum(mdata.simulation.lead_time)})"
        ),
        "br": capitalize(
            f"{format_meta_datum(mdata.simulation.now_rel - mdata.release.start_rel)}"
            f" {words['after']} {words['release_start']}"
        ),
    }

    # Data info box
    labels["data_info"] = {
        "lines": [],
    }
    labels["data_info"]["lines"].append(
        f"{words['substance'].c}:"
        f"\t{format_meta_datum(mdata.species.name, join_values=' / ')}",
    )
    labels["data_info"]["lines"].append(
        f"{words['input_variable'].c}:\t{capitalize(var_name_abbr)}"
    )
    if plot_variable == "concentration":
        labels["data_info"]["lines"].append(
            f"{words['height'].c}:"
            f"\t{escape_format_keys(format_level_label(mdata, words))}"
        )
    if setup.model.simulation_type == "ensemble":
        if ens_variable == "probability":
            op = {"lower": "gt", "upper": "lt"}[ens_params.thr_type]
            labels["data_info"]["lines"].append(
                f"{words['selection']}:\t{symbols[op]} {ens_params.thr}"
                f" {format_meta_datum(unit=format_meta_datum(mdata.variable.unit))}"
            )
        elif ens_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            labels["data_info"]["lines"].append(
                # f"{words['cloud_density']}:\t{words['minimum', 'abbr']}"
                # f"{words['threshold']}:\t"
                f"{words['cloud_threshold', 'abbr']}:\t {ens_params.thr}"
                f" {format_meta_datum(unit=format_meta_datum(mdata.variable.unit))}"
            )
            n_min = ens_params.mem_min or 0
            n_tot = len((setup.model.ens_member_id or []))
            labels["data_info"]["lines"].append(
                # f"{words['number_of', 'abbr'].c} {words['member', 'pl']}:"
                # f"\t {n_min}"
                f"{words['minimum', 'abbr'].c} {words['member', 'pl']}:"
                f"\t{n_min}"
                r"$\,/\,$"
                f"{n_tot} ({n_min/(n_tot or 0.001):.0%})"
            )
        labels["data_info"]["lines"].append(
            f"{words['ensemble_variable', 'abbr']}:\t{capitalize(ens_var_name)}"
        )

    # Legend box
    labels["legend"] = {
        "title": "",  # SR_TMP Force into 2nd position in dict (for tests)
        "release_site": words["release_site"].s,
        "site": words["site"].s,
        "max": words["maximum", "abbr"].s,
        "maximum": words["maximum"].s,
    }
    if plot_variable == "concentration":
        labels["legend"]["tc"] = (
            f"{words['height']}:"
            f" {escape_format_keys(format_level_label(mdata, words))}"
        )
    # Legend box title
    if not unit:
        title = f"{short_name}"
    elif unit == "%":
        title = f"{words['probability']} ({unit})"
    elif plot_variable == "cloud_arrival_time" or ens_variable == "cloud_arrival_time":
        title = f"{words['hour', 'pl']} {words['until']} {words['arrival']}"
    elif (
        plot_variable == "cloud_departure_time"
        or ens_variable == "cloud_departure_time"
    ):
        title = f"{words['hour', 'pl']} {words['until']} {words['departure']}"
    else:
        title = f"{short_name} ({unit})"
    labels["legend"]["title"] = title
    labels["legend"]["unit"] = unit

    # Bottom box
    labels["bottom"] = {
        "model_info": format_model_info(setup.model, words),
        "copyright": f"{symbols['copyright']}{words['meteoswiss']}",
    }

    # Release info
    site_name = format_meta_datum(mdata.release.site_name)
    site_lat_lon = format_release_site_coords_labels(words, symbols, mdata)
    release_height = format_meta_datum(
        mdata.release.height, mdata.release.height_unit
    ).replace("meters", r"$\,$" + words["m_agl"].s)
    release_start = format_meta_datum(
        cast(datetime, mdata.simulation.start)
        + cast(timedelta, mdata.release.start_rel)
    )
    release_end = format_meta_datum(
        cast(datetime, mdata.simulation.start) + cast(timedelta, mdata.release.end_rel)
    )
    release_rate = format_meta_datum(mdata.release.rate, mdata.release.rate_unit)
    release_mass = format_meta_datum(mdata.release.mass, mdata.release.mass_unit)
    substance = format_meta_datum(mdata.species.name, join_values=" / ")
    half_life = format_meta_datum(mdata.species.half_life, mdata.species.half_life_unit)
    deposit_vel = format_meta_datum(
        mdata.species.deposition_velocity,
        mdata.species.deposition_velocity_unit,
    )
    sediment_vel = format_meta_datum(
        mdata.species.sedimentation_velocity,
        mdata.species.sedimentation_velocity_unit,
    )
    washout_coeff = format_meta_datum(
        mdata.species.washout_coefficient,
        mdata.species.washout_coefficient_unit,
    )
    washout_exponent = format_meta_datum(mdata.species.washout_exponent)
    lines_parts: List[Optional[Tuple[Word, str]]] = [
        (words["site"], site_name),
        (words["latitude"], site_lat_lon[0]),
        (words["longitude"], site_lat_lon[1]),
        (words["height"], release_height),
        None,
        (words["start"], release_start),
        (words["end"], release_end),
        (words["rate"], release_rate),
        (words["total_mass"], release_mass),
        None,
        (words["substance"], substance),
        (words["half_life"], half_life),
        (words["deposition_velocity", "abbr"], deposit_vel),
        (words["sedimentation_velocity", "abbr"], sediment_vel),
        (words["washout_coeff"], washout_coeff),
        (words["washout_exponent"], washout_exponent),
    ]
    lines_str = ""
    for line_parts in lines_parts:
        if line_parts is None:
            lines_str += "\n\n"
        else:
            left, right = line_parts
            lines_str += f"{capitalize(left)}:\t{right}\n"
    labels["release_info"] = {
        "title": capitalize(words["release"].t),
        "lines_str": lines_str,
    }

    # Capitalize all labels
    for label_group in labels.values():
        for name, label in label_group.items():
            if name == "lines":
                label = [capitalize(line) for line in label]
            elif name == "blocks":
                label = [[capitalize(line) for line in block] for block in label]
            elif isinstance(label, str):
                label = capitalize(label)
            label_group[name] = label

    return labels


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0915  # too-many-statements
def format_names_etc(
    setup: PlotSetup, words: TranslatedWords, mdata: MetaData
) -> Dict[str, str]:
    ens_params = setup.panels.collect_equal("ens_params")
    plot_variable = setup.panels.collect_equal("plot_variable")
    integrate = setup.panels.collect_equal("integrate")
    # SR_TMP <
    if setup.plot_type == "multipanel" and setup.multipanel_param == "ens_variable":
        ens_variable = "+".join(setup.panels.collect("ens_variable"))
    else:
        ens_variable = setup.panels.collect_equal("ens_variable")
    # SR_TMP >

    long_name = ""
    short_name = ""

    def format_var_names(
        plot_variable: str, words: TranslatedWords
    ) -> Tuple[str, str, str]:
        if plot_variable.endswith("_deposition"):
            if plot_variable == "tot_deposition":
                dep_type_word = "total"
            elif plot_variable == "dry_deposition":
                dep_type_word = "dry"
            elif plot_variable == "wet_deposition":
                dep_type_word = "wet"
            word = f"{dep_type_word}_surface_deposition"
            if not integrate:
                word = f"incremental_{word}"
        elif plot_variable == "concentration":
            word = "air_activity_concentration"
            if integrate:
                word = f"integrated_{word}"
        else:
            word = plot_variable
        var_name_abbr = words[word, "abbr"].s
        var_name = words[word].s
        var_name_rel = words[word, "of"].s
        return var_name, var_name_abbr, var_name_rel

    # pylint: disable=W0621  # redefined-outer-name
    def _format_unit(setup: PlotSetup, words: TranslatedWords, mdata: MetaData) -> str:
        if setup.model.simulation_type == "ensemble":
            if ens_variable == "probability":
                return "%"
            elif ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                return f"{words['hour', 'pl']}"
        return format_meta_datum(unit=format_meta_datum(mdata.variable.unit))

    unit = _format_unit(setup, words, mdata)
    var_name, var_name_abbr, var_name_rel = format_var_names(plot_variable, words)

    # Short/long names #1: By variable
    if plot_variable == "concentration":
        if integrate:
            long_name = var_name
            short_name = words["integrated_concentration", "abbr"].s
        else:
            long_name = var_name
            short_name = words["concentration"].s
    elif plot_variable.endswith("deposition"):
        long_name = var_name
        short_name = words["deposition"].s
    elif plot_variable in [
        "affected_area",
        "cloud_arrival_time",
        "cloud_departure_time",
    ]:
        word = plot_variable
        long_name = words[word].s
        short_name = words[word, "abbr"].s

    # Short/long names #2: By ensemble variable type
    ens_var_name = "none"
    if setup.model.simulation_type == "ensemble":
        if ens_variable in [
            "minimum",
            "maximum",
            "median",
            "mean",
        ]:
            long_name = f"{words[f'ensemble_{ens_variable}']} {var_name_rel}"
        elif ens_variable == "std_dev":
            ens_var_name = words["standard_deviation"].s
            long_name = f"{words['ensemble_standard_deviation']} {var_name_rel}"
        elif ens_variable == "med_abs_dev":
            ens_var_name = words["median_absolute_deviation"].s
            long_name = (
                f"{words['ensemble_median_absolute_deviation', 'abbr']} {var_name_rel}"
            )
        elif ens_variable == "percentile":
            assert ens_params.pctl is not None  # mypy
            pctl = ens_params.pctl
            th = {1: "st", 2: "nd", 3: "rd"}.get(pctl, "th")  # type: ignore
            long_name = (
                f"{pctl:g}{words['th', th]}" f" {words['percentile']} {var_name_rel}"
            )
            ens_var_name = f"{pctl:g}{words['th', th]} {words['percentile']}"
        elif ens_variable == "probability":
            short_name = words["probability"].s
            long_name = f"{words['probability']} {var_name_rel}"
        elif ens_variable == "cloud_arrival_time":
            long_name = words["ensemble_cloud_arrival_time"].s
            short_name = words["arrival"].s
            ens_var_name = words["cloud_arrival_time", "abbr"].s
        elif ens_variable == "cloud_departure_time":
            long_name = words["ensemble_cloud_departure_time"].s
            short_name = words["departure"].s
            ens_var_name = words["cloud_departure_time", "abbr"].s
        if ens_var_name == "none":
            # SR_TMP <
            # ens_var_name = words[ens_variable].c
            if (
                setup.plot_type == "multipanel"
                and setup.multipanel_param == "ens_variable"
            ):
                ens_var_name = "\n+ ".join(
                    [
                        words[ens_variable_i].c
                        for ens_variable_i in setup.collect("ens_variable")
                    ]
                )
            else:
                ens_var_name = words[ens_variable].c
            # SR_TMP >

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


def capitalize(s: Union[str, Word]) -> str:
    """Capitalize the first letter while leaving all others as they are."""
    s = str(s)
    if not s:
        return s
    try:
        return s[0].upper() + s[1:]
    except Exception as e:
        raise ValueError(f"string not capitalizable: '{s}'") from e


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


def format_release_site_coords_labels(
    words: TranslatedWords, symbols: Words, mdata: MetaData
) -> Tuple[str, str]:
    lat_deg_fmt = capitalize(format_coord_label("north", words, symbols))
    lon_deg_fmt = capitalize(format_coord_label("east", words, symbols))
    lat = Degrees(mdata.release.lat)
    lon = Degrees(mdata.release.lon)
    lat_deg = lat_deg_fmt.format(d=lat.degs(), m=lat.mins(), f=lat.frac())
    lon_deg = lon_deg_fmt.format(d=lon.degs(), m=lon.mins(), f=lon.frac())
    return (lat_deg, lon_deg)


def format_model_info(model_setup: ModelSetup, words: TranslatedWords) -> str:
    # SR_TMP <
    simulation_type = model_setup.simulation_type
    model_name = model_setup.name
    base_time = model_setup.base_time
    ens_member_id = model_setup.ens_member_id
    # SR_TMP >
    model_info = None
    if simulation_type == "deterministic":
        if model_name in ["COSMO-1", "COSMO-2", "IFS-HRES", "IFS-HRES-EU"]:
            model_info = model_name
        elif model_name in ["COSMO-1E", "COSMO-2E"]:
            model_info = f"{model_name} {words['control_run']}"
        else:
            raise NotImplementedError(f"model '{model_name}'")
    elif simulation_type == "ensemble":
        model_info = (
            f"{model_name} {words['ensemble']}"
            f" ({len(ens_member_id or [])} {words['member', 'pl']}:"
            f" {format_numbers_range(ens_member_id or [], fmt='02d')})"
        )
    if model_info is None:
        raise NotImplementedError(
            f"model_name='{model_name}' and simulation_type='{simulation_type}'"
        )
    assert base_time is not None  # mypy
    base_time_dt = init_datetime(base_time)
    return (
        f"{words['flexpart']} {words['based_on']} {model_info}"
        f", {format_meta_datum(base_time_dt)}"
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
    setup: PlotSetup,
    words: TranslatedWords,
    cap: bool = False,
) -> str:
    plot_variable = setup.panels.collect_equal("plot_variable")
    integrate = setup.panels.collect_equal("integrate")
    if not integrate:
        operation = words["averaged_over"].s
    elif plot_variable in [
        "concentration",
        "affected_area",
        "cloud_arrival_time",
        "cloud_departure_time",
    ]:
        operation = words["summed_over"].s
    elif plot_variable.endswith("deposition"):
        operation = words["accumulated_over"].s
    else:
        raise NotImplementedError(
            f"operation for {'' if integrate else 'non-'}integrated"
            f" input variable '{plot_variable}'"
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
        deg_dir_unit = words["degN"].s
    elif direction == "east":
        deg_dir_unit = words["degE"].s
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
    except KeyError as e:
        raise ValueError(f"n_levels={n_levels}") from e

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


def levels_from_time_stats(
    simulation_type: str,
    ens_variable: Optional[str],
    time_stats: FieldStats,
    levels_config_dct: Dict[str, Any],
) -> np.ndarray:
    def _auto_levels_log10(n_levels: int, val_max: float) -> List[float]:
        if not np.isfinite(val_max):
            raise ValueError("val_max not finite", val_max)
        # SR_TMP <
        if val_max == 0.0:
            val_max = 1e-6
        # SR_TMP >
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        return 10 ** np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d
        )

    if simulation_type == "ensemble":
        assert ens_variable is not None  # mypy
        if ens_variable.endswith("probability") or ens_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            return levels_config_dct["levels"]
    if levels_config_dct.get("n"):
        return _auto_levels_log10(levels_config_dct["n"], val_max=time_stats.max)
    return levels_config_dct["levels"]
