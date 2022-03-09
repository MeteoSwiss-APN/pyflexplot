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
from __future__ import annotations

# Standard library
import os
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from pprint import pformat
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
from srutils.format import ordinal
from srutils.geo import Degrees
from srutils.plotting import linear_cmap
from srutils.plotting import truncate_cmap
from words import WordT

# Local
from . import __version__
from .input.field import Field
from .input.field import FieldGroup
from .input.meta_data import format_meta_datum
from .input.meta_data import ReleaseMetaData
from .input.meta_data import SimulationMetaData
from .input.meta_data import SpeciesMetaData
from .input.meta_data import VariableMetaData
from .output import FilePathFormatter
from .plot_layouts import BoxedPlotLayout
from .plotting.boxed_plot import BoxedPlot
from .plotting.boxed_plot import BoxedPlotConfig
from .plotting.boxed_plot import BoxedPlotPanelConfig
from .plotting.boxed_plot import ContourLevelsConfig
from .plotting.boxed_plot import ContourLevelsLegendConfig
from .plotting.boxed_plot import FontConfig
from .plotting.boxed_plot import FontSizes
from .plotting.boxed_plot import MarkersConfig
from .plotting.domain import CloudDomain
from .plotting.domain import Domain
from .plotting.domain import ReleaseSiteDomain
from .plotting.map_axes import MapAxesConfig
from .plotting.text_box_axes import TextBoxAxes
from .setups.layout_setup import LayoutSetup
from .setups.model_setup import ModelSetup
from .setups.plot_panel_setup import PlotPanelSetup
from .setups.plot_setup import PlotSetup
from .utils.formatting import escape_format_keys
from .utils.formatting import format_level_ranges
from .utils.logging import log
from .utils.typing import ColorType
from .words import SYMBOLS
from .words import TranslatedWords
from .words import WORDS
from .words import Words


def format_out_file_paths(
    field_group: FieldGroup, prev_paths: List[str], dest_dir: Optional[str] = None
) -> List[str]:
    plot_setup = field_group.plot_setup
    _mdata = next(iter(field_group)).mdata
    release_site_name = _mdata.release.raw_site_name
    release_start_rel = _mdata.release.start_rel
    simulation_start = _mdata.simulation.start
    simulation_time_steps = _mdata.simulation.time_steps
    for idx, field in enumerate(field_group):
        if idx == 0:
            continue
        if release_site_name != field.mdata.release.raw_site_name:
            raise NotImplementedError(
                "release site name differs between fields:"
                f"\n{release_site_name}\n!=\n{field.mdata.release.raw_site_name}"
            )
        if release_start_rel != field.mdata.release.start_rel:
            raise NotImplementedError(
                "rel. release start differs between fields:"
                f"\n{release_start_rel}\n!=\n{field.mdata.release.start_rel}"
            )
        if simulation_start != field.mdata.simulation.start:
            raise NotImplementedError(
                "simulation start differs between fields:"
                f"\n{simulation_start}\n!=\n{field.mdata.simulation.start}"
            )
        if simulation_time_steps != field.mdata.simulation.time_steps:
            raise NotImplementedError(
                "simulation time steps differ between fields:"
                f"\n{simulation_time_steps}\n!=\n{field.mdata.simulation.time_steps}"
            )
    out_file_templates: Sequence[str] = (
        [plot_setup.files.output]
        if isinstance(plot_setup.files.output, str)
        else plot_setup.files.output
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
            release_site=release_site_name,
            release_start=simulation_start + release_start_rel,
            time_steps=tuple(simulation_time_steps),
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
    _mdata = next(iter(field_group)).mdata
    release_mdata = _mdata.release
    species_mdata = _mdata.species
    variable_mdata = _mdata.variable
    simulation_mdata_lst = [field.mdata.simulation for field in field_group]
    simulation_duration_hours_lst = [
        int(field.mdata.simulation.get_duration("hours")) for field in field_group
    ]
    assert len(set(simulation_duration_hours_lst)) == 1
    simulation_duration_hours = next(iter(simulation_duration_hours_lst))
    simulation_time_steps_lst: List[Tuple[int, ...]] = [
        tuple(field.mdata.simulation.time_steps) for field in field_group
    ]
    assert len(set(simulation_time_steps_lst)) == 1
    simulation_time_steps: List[int] = list(next(iter(simulation_time_steps_lst)))
    # SR_TMP >  SR_MULTIPANEL
    val_max = max(field.time_props.stats.max for field in field_group)
    labels = create_box_labels(
        plot_setup, release_mdata, species_mdata, variable_mdata, simulation_mdata_lst
    )
    plot_config = create_plot_config(
        plot_setup, labels, simulation_duration_hours, simulation_time_steps, val_max
    )
    map_configs: List[MapAxesConfig] = [
        create_map_config(
            field_group.plot_setup,
            field.panel_setup,
            plot_config.layout.get_aspect("center"),
        )
        for field in field_group
    ]
    domains: List[Domain] = []
    for field, map_config in zip(field_group, map_configs):
        domains.append(get_domain(field, map_config.aspect))
    plot = BoxedPlot(plot_config)
    plot.add_map_plot_panels(field_group, domains, map_configs)

    plot_add_text_boxes(plot, field_group, plot.config.layout, show_version)
    for file_path in file_paths:
        log(dbg=f"creating plot {file_path}")
        if write:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            log(dbg=f"writing plot {file_path}")
            plot.write(file_path)
            # SR_TMP < TODO clean this up; add optional setup param for file name
            if "standalone_release_info" in plot.config.labels:
                write_standalone_release_info(
                    file_path,
                    plot.config,
                )
            # SR_TMP >
        log(dbg=f"created plot {file_path}")
    plot.clean()
    return plot


def write_standalone_release_info(plot_path: str, plot_config: BoxedPlotConfig) -> str:
    path = Path(plot_path).with_suffix(f".release_info{Path(plot_path).suffix}")
    log(inf=f"write standalone release info to {path}")
    layout = BoxedPlotLayout(
        plot_config.setup.layout.derive({"type": "standalone_release_info"}),
        aspects={"tot": 1.5},
        rects={"tot": (0, 0, 1, 1)},
    )
    species_id = plot_config.setup.panels.collect_equal("dimensions.species_id")
    n_species = 1 if isinstance(species_id, int) else len(species_id)
    width = 1.67 + 0.67 * n_species
    config = BoxedPlotConfig(
        fig_size=(width, 2.5),
        layout=layout,
        labels=plot_config.labels,
        panels=[],
        setup=plot_config.setup,
    )

    def fill_box(box: TextBoxAxes, plot: BoxedPlot) -> None:
        labels = plot.config.labels["standalone_release_info"]

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
            dy_unit=-10.0,
            dy_line=8.0,
            fontname=plot.config.font.name,
            size=plot.config.font.sizes.content_small,
        )

    plt = BoxedPlot(config)
    plt.add_text_box("standalone_release_info", (0, 0, 1, 1), fill=fill_box)
    plt.write(path)
    return str(path)


# pylint: disable=R0912  # too-many-branches (>12)
def get_domain(field: Field, aspect: float) -> Domain:
    """Initialize Domain object (projection and extent)."""
    lat = field.lat
    lon = field.lon
    model_name = field.model_setup.name
    domain_type = field.panel_setup.domain
    domain_size_lat = field.panel_setup.domain_size_lat
    domain_size_lon = field.panel_setup.domain_size_lon
    assert field.mdata is not None  # mypy
    release_lat = field.mdata.release.lat
    release_lon = field.mdata.release.lon
    field_proj = field.projs.data
    mask_nz = field.time_props.mask_nz
    domain: Optional[Domain] = None
    if domain_type == "full":
        if model_name in ["COSMO-1", "COSMO-1E", "COSMO-2", "COSMO-E"]:
            domain = Domain(lat, lon, config={"zoom_fact": 1.01})
        elif model_name in ["COSMO-2E"]:
            domain = Domain(lat, lon, config={"zoom_fact": 1.025})
        else:
            domain = Domain(lat, lon)
    elif domain_type == "release_site":
        domain = ReleaseSiteDomain(
            lat,
            lon,
            config={
                "aspect": aspect,
                "field_proj": field_proj,
                "min_size_lat": domain_size_lat,
                "min_size_lon": domain_size_lon,
                "release_lat": release_lat,
                "release_lon": release_lon,
            },
        )
    elif domain_type == "alps":
        if model_name == "IFS-HRES-EU":
            domain = Domain(
                lat, lon, config={"zoom_fact": 3.4, "rel_offset": (-0.165, -0.11)}
            )
    elif domain_type == "cloud":
        domain = CloudDomain(
            lat,
            lon,
            mask=mask_nz,
            config={
                "zoom_fact": 0.9,
                "aspect": aspect,
                "min_size_lat": domain_size_lat,
                "min_size_lon": domain_size_lon,
                "periodic_lon": (model_name == "IFS-HRES"),
                "release_lat": release_lat,
                "release_lon": release_lon,
            },
        )
    elif domain_type == "ch":
        if model_name in ["COSMO-1", "COSMO-1E", "COSMO-2E"]:
            domain = Domain(
                lat, lon, config={"zoom_fact": 3.6, "rel_offset": (-0.02, 0.045)}
            )
        elif model_name in ["COSMO-2", "COSMO-E"]:
            domain = Domain(
                lat, lon, config={"zoom_fact": 3.23, "rel_offset": (0.037, 0.1065)}
            )
        elif model_name == "IFS-HRES-EU":
            domain = Domain(
                lat, lon, config={"zoom_fact": 11.0, "rel_offset": (-0.18, -0.11)}
            )
    if domain is None:
        raise NotImplementedError(
            f"domain for model '{model_name}' and domain type '{domain_type}'"
        )
    return domain


# SR_TMP <<< TODO Clean up nested functions! Eventually introduce class(es) of some kind
# pylint: disable=R0914  # too-many-locals (>15)
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

    def fill_box_2nd_title(
        box: TextBoxAxes,
        plot: BoxedPlot,
        release_mdata: ReleaseMetaData,
        species_mdata: SpeciesMetaData,
    ) -> None:
        """Fill the secondary title box of the deterministic plot layout."""
        font_size = plot.config.font.sizes.content_large
        box.text(
            capitalize(format_meta_datum(species_mdata.name)),
            loc="tc",
            fontname=plot.config.font.name,
            size=font_size,
        )
        box.text(
            capitalize(format_meta_datum(release_mdata.site_name)),
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
    def fill_box_legend(
        box: TextBoxAxes,
        plot: BoxedPlot,
        release_mdata: ReleaseMetaData,
        max_vals: Sequence[float],
    ) -> None:
        """Fill the box containing the plot legend."""
        labels = plot.config.labels["legend"]

        # SR_TMP TODO multipanel
        legend_labels: Sequence[str] = next(
            iter(plot.config.panels)
        ).levels.legend.labels
        colors_lst: Sequence[Sequence[ColorType]] = [
            panel.colors for panel in plot.config.panels
        ]
        markers: MarkersConfig = next(iter(plot.config.panels)).markers
        if len(plot.config.panels) > 1:
            for panel_config in plot.config.panels[1:]:
                if panel_config.levels.legend.labels != legend_labels:
                    raise NotImplementedError("legend labels differ")
                if panel_config.markers != markers:
                    raise NotImplementedError("markers differ")
        # SR_TMP >

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
        def add_color_rects(colors: Sequence[ColorType], dx: float, w: float) -> None:
            dy = dy0_boxes
            for color in colors[::-1]:
                box.color_rect(
                    loc="tc",
                    x_anker="left",
                    dx=dx,
                    dy=dy,
                    w=w,
                    h=h_legend_box,
                    fc=color,
                    ec="black",
                    lw=1.0,
                )
                dy -= dy_line

        if all(colors == colors_lst[0] for colors in colors_lst[1:]):
            add_color_rects(next(iter(colors_lst)), dx_legend_box, w_legend_box)
        else:
            w = w_legend_box / len(colors_lst) * 2
            dx = dx_legend_box - w
            for colors in colors_lst:
                add_color_rects(colors, dx, w)
                dx += w

        dy0_markers = dy0_boxes - dy_line * (len(legend_labels) - 0.3)
        dy0_marker = dy0_markers

        # Field maximum marker
        if markers.mark_field_max:
            dy_marker_label_max = dy0_marker
            dy0_marker -= dy_line
            dy_max_marker = dy_marker_label_max - 0.7
            assert markers.markers is not None  # mypy
            box.marker(
                loc="tc",
                dx=dx_marker,
                dy=dy_max_marker,
                **markers.markers["max"],
            )
            max_val = next(iter(max_vals)) if len(max_vals) == 1 else None
            max_marker_label = format_max_marker_label(labels, max_val)
            box.text(
                s=max_marker_label,
                loc="tc",
                dx=dx_marker_label,
                dy=dy_marker_label_max,
                ha="left",
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.content_medium,
            )

        # Release site marker
        if markers.mark_release_site:
            dy_site_label = dy0_marker
            dy0_marker -= dy_line
            dy_site_marker = dy_site_label - 0.7
            assert markers.markers is not None  # mypy
            box.marker(
                loc="tc",
                dx=dx_marker,
                dy=dy_site_marker,
                **markers.markers["site"],
            )
            box.text(
                s=f"{labels['site']}: {format_meta_datum(release_mdata.site_name)}",
                loc="tc",
                dx=dx_marker_label,
                dy=dy_site_label,
                ha="left",
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.content_medium,
            )

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
    release_mdata = next(iter(fields)).mdata.release
    species_mdata = next(iter(fields)).mdata.species
    for field in fields:
        if field.mdata.release != release_mdata:
            raise NotImplementedError(
                "release meta data differ between fields:"
                f"\n{field.mdata.release}\n!=\n{release_mdata}"
            )
        if field.mdata.species != species_mdata:
            raise NotImplementedError(
                "species meta data differ between fields:"
                f"\n{field.mdata.species}\n!=\n{species_mdata}"
            )
    # SR_TMP >
    max_vals = [np.nanmax(field.fld) for field in fields]

    plot.add_text_box("top", layout.get_rect("top"), fill_box_title)
    if layout.setup.type == "post_vintage":
        plot.add_text_box(
            "right_top",
            layout.get_rect("right_top"),
            lambda box, plot: fill_box_2nd_title(
                box, plot, release_mdata, species_mdata
            ),
        )
    elif layout.setup.type == "post_vintage_ens":
        plot.add_text_box(
            "right_top",
            layout.get_rect("right_top"),
            fill_box_data_info,
        )
    elif layout.setup.type == "standalone_details":
        plot.add_text_box(
            "right_top",
            layout.get_rect("right_top"),
            fill_box_data_info,
        )
    plot.add_text_box(
        "right_middle",
        layout.get_rect("right_middle"),
        lambda box, plot: fill_box_legend(box, plot, release_mdata, max_vals),
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
    scale_fact = plot_setup.layout.scale_fact
    plot_type = plot_setup.layout.plot_type
    domain_type = panel_setup.domain
    lang = panel_setup.lang
    n_panels = len(plot_setup.panels)

    if plot_type == "multipanel":
        # Shrink city labels etc.
        scale_fact *= 0.75

    if n_panels == 1:
        ref_dist_config = {
            "font_size": 11.0,
            "h_box": 0.06,
            "min_w_box": 0.075,
        }
    elif n_panels == 4:
        ref_dist_config = {
            "font_size": 9.0,
            "h_box": 0.09,
            "min_w_box": 0.12,
        }
    else:
        raise NotImplementedError(f"{n_panels} panels")
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
        "ref_dist_config": {
            **ref_dist_config,
            "dist": 250,
        },
    }
    conf_regional_scale: Dict[str, Any] = {
        "geo_res": "50m",
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 300_000,
        "ref_dist_config": {
            **ref_dist_config,
            "dist": 100,
        },
    }
    conf_country_scale: Dict[str, Any] = {
        "geo_res": "10m",
        "geo_res_cities": "10m",
        "geo_res_rivers": "10m",
        "min_city_pop": 0,
        "ref_dist_config": {
            **ref_dist_config,
            "dist": 25,
        },
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
def create_plot_config(
    setup: PlotSetup,
    labels: Dict[str, Dict[str, Any]],
    simulation_duration_hours: int,
    simulation_time_steps: Sequence[int],
    val_max: float,
) -> BoxedPlotConfig:
    fig_size = (12.5 * setup.layout.scale_fact, 8.0 * setup.layout.scale_fact)
    fig_aspect = np.divide(*fig_size)
    layout = BoxedPlotLayout.create(setup.layout, aspect=fig_aspect)
    font_config = FontConfig(sizes=FontSizes().scale(setup.layout.scale_fact))
    panels_config = [
        create_panel_config(
            panel_setup,
            setup.layout,
            setup.model,
            simulation_duration_hours,
            simulation_time_steps,
            val_max,
        )
        for panel_setup in setup.panels
    ]
    return BoxedPlotConfig(
        setup=setup,
        panels=panels_config,
        fig_size=fig_size,
        font=font_config,
        labels=labels,
        layout=layout,
    )


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0913  # too-many-args (>5)
# pylint: disable=R0914  # too-many-locals (>15)
def create_panel_config(
    panel_setup: PlotPanelSetup,
    layout_setup: LayoutSetup,
    model_setup: ModelSetup,
    simulation_duration_hours: int,
    simulation_time_steps: Sequence[int],
    val_max: float,
) -> BoxedPlotPanelConfig:
    plot_variable = panel_setup.plot_variable
    ens_variable = panel_setup.ens_variable

    # Label
    if layout_setup.plot_type != "multipanel":
        label = None
    elif layout_setup.multipanel_param == "ens_variable":
        label = capitalize(WORDS[panel_setup.ens_variable])
    elif layout_setup.multipanel_param == "ens_params.pctl":
        assert panel_setup.ens_params.pctl is not None  # mypy
        label = ordinal(panel_setup.ens_params.pctl, "g", panel_setup.lang)
    elif layout_setup.multipanel_param == "ens_params.thr":
        assert panel_setup.ens_params.thr is not None  # mypy
        label = f"{panel_setup.ens_params.thr:g}"
    elif layout_setup.multipanel_param == "time":
        label = format_meta_datum(
            init_datetime(simulation_time_steps[panel_setup.dimensions.time])
        )
    else:
        raise NotImplementedError(
            f"label for multipanel_param '{layout_setup.multipanel_param}'"
        )

    # Levels and legend
    levels_config_dct: Dict[str, Any] = {
        "include_lower": False,
        "scale": "log",
    }
    legend_config_dct: Dict[str, Any] = {}
    if plot_variable == "affected_area" and ens_variable != "probability":
        levels_config_dct["extend"] = "none"
        levels_config_dct["levels"] = np.array([0.0, np.inf])
        levels_config_dct["scale"] = "lin"
    elif model_setup.simulation_type == "ensemble" and ens_variable == "probability":
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
        levels_config_dct["levels"] = get_cloud_timing_levels(simulation_duration_hours)
        if (
            plot_variable == "cloud_arrival_time"
            or ens_variable == "cloud_arrival_time"
        ):
            levels_config_dct["extend"] = "min"
        elif (
            plot_variable == "cloud_departure_time"
            or ens_variable == "cloud_departure_time"
        ):
            levels_config_dct["extend"] = "max"
    if "levels" not in levels_config_dct:
        if plot_variable == "concentration":
            levels_config_dct["levels"] = levels_from_time_stats(
                n_levels=8, val_max=val_max
            )
        elif plot_variable.endswith("deposition"):
            levels_config_dct["levels"] = levels_from_time_stats(
                n_levels=9, val_max=val_max
            )
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

    def cmap2colors(
        cmap: Union[str, Colormap],
        levels_config: ContourLevelsConfig,
        color_under: Optional[str] = None,
        color_over: Optional[str] = None,
    ) -> Sequence[ColorType]:
        extend = levels_config.extend
        n_colors = levels_config.n - 1
        if extend in ["min", "both"] and not color_under:
            n_colors += 1
        if extend in ["max", "both"] and not color_over:
            n_colors += 1
        cmap = mpl.cm.get_cmap(cmap, lut=n_colors)
        colors: list[ColorType]
        try:
            colors = cmap.colors.tolist()
        except AttributeError:
            if n_colors == 1:
                colors = [cmap(0.0)]
            else:
                colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        if extend in ["min", "both"] and color_under:
            colors.insert(0, color_under)
        if extend in ["max", "both"] and color_over:
            colors.append(color_over)
        return colors

    # Colors
    if plot_variable == "affected_area" and ens_variable != "probability":
        colors = (np.array([(200, 200, 200)]) / 255).tolist()
    elif model_setup.simulation_type == "ensemble" and ens_variable == "probability":
        cmap = truncate_cmap("terrain_r", 0.075)
        colors = cmap2colors(cmap, levels_config)
    elif (
        model_setup.simulation_type == "ensemble"
        and ens_variable == "percentile"
        and layout_setup.color_style == "mono"
    ):
        ens_param_pctl = panel_setup.ens_params.pctl
        assert ens_param_pctl is not None  # mypy
        if ens_param_pctl <= 25:
            # cmap = linear_cmap("greens", "darkgreen")
            cmap = linear_cmap("browns", "olive")
        elif ens_param_pctl <= 45:
            cmap = linear_cmap("browns", "saddlebrown")
        elif ens_param_pctl <= 65:
            cmap = linear_cmap("purples", "indigo")
        elif ens_param_pctl <= 85:
            cmap = linear_cmap("blues", "darkblue")
        else:
            # cmap = "Greys"
            cmap = linear_cmap("grays", "black")
        cmap = truncate_cmap(cmap, 0.1)
        colors = cmap2colors(cmap, levels_config)
    elif plot_variable == "cloud_arrival_time" or (
        ens_variable == "cloud_arrival_time"
    ):
        # SR_TMP < TODO settle on one
        # cmap = "viridis"
        # cmap = "rainbow_r"
        # cmap = truncate_cmap("terrain", 0.0, 0.9)
        cmap = truncate_cmap("nipy_spectral_r", 0.20, 0.95)
        # SR_TMP >
        colors = cmap2colors(
            cmap,
            levels_config,
            color_under="slategray",
            color_over="lightgray",
        )
    elif plot_variable == "cloud_departure_time" or (
        ens_variable == "cloud_departure_time"
    ):
        # SR_TMP < TODO settle on one
        # cmap = "viridis_r"
        # cmap = "rainbow"
        # cmap = truncate_cmap("terrain_r", 0.1, 1.0)
        cmap = truncate_cmap("nipy_spectral", 0.05, 0.80)
        # SR_TMP >
        colors = cmap2colors(
            cmap,
            levels_config,
            color_under="lightgray",
            color_over="slategray",
        )
    else:
        colors = colors_flexplot(levels_config.n, levels_config.extend)

    # Markers
    markers_config_dct: Dict[str, Any] = {}
    if model_setup.simulation_type == "deterministic":
        if plot_variable in [
            "affected_area",
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            markers_config_dct["mark_field_max"] = False
        else:
            markers_config_dct["mark_field_max"] = True
    elif model_setup.simulation_type == "ensemble":
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
            markers_config_dct["mark_field_max"] = True
        else:
            markers_config_dct["mark_field_max"] = False
    markers = {}
    markers["max"] = {
        "marker": "+",
        "color": "black",
        "markersize": 10 * layout_setup.scale_fact,
        "markeredgewidth": 1.5 * layout_setup.scale_fact,
    }
    markers["site"] = {
        "marker": "^",
        "markeredgecolor": "red",
        "markerfacecolor": "white",
        "markersize": 7.5 * layout_setup.scale_fact,
        "markeredgewidth": 1.5 * layout_setup.scale_fact,
    }
    markers_config_dct["markers"] = markers
    markers_config = MarkersConfig(**markers_config_dct)
    return BoxedPlotPanelConfig(
        setup=panel_setup,
        colors=colors,
        label=label,
        levels=levels_config,
        markers=markers_config,
    )


# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0914  # too-many-locals
# pylint: disable=R0915  # too-many-statements
def create_box_labels(
    setup: PlotSetup,
    release_mdata: ReleaseMetaData,
    species_mdata: SpeciesMetaData,
    variable_mdata: VariableMetaData,
    simulation_mdata_lst: Sequence[SimulationMetaData],
) -> Dict[str, Dict[str, Any]]:
    words = WORDS
    symbols = SYMBOLS
    words.set_active_lang(setup.panels.collect_equal("lang"))

    # SR_TMP <
    plot_type = setup.layout.plot_type
    multipanel_param = setup.layout.multipanel_param
    if plot_type == "multipanel" and multipanel_param == "ens_variable":
        ens_variable = "+".join(setup.panels.collect("ens_variable"))
    else:
        ens_variable = setup.panels.collect_equal("ens_variable")
    if plot_type == "multipanel" and multipanel_param == "ens_params.thr":
        ens_param_thrs = setup.panels.collect("ens_params.thr")
    else:
        ens_param_thr = setup.panels.collect_equal("ens_params.thr")
    # SR_TMP >

    ens_param_thr_type = setup.panels.collect_equal("ens_params.thr_type")
    ens_param_mem_min = setup.panels.collect_equal("ens_params.mem_min")
    plot_variable = setup.panels.collect_equal("plot_variable")

    # Format variable name in various ways
    names = format_names_etc(setup, words, variable_mdata)
    short_name = names["short"]
    # long_name = names["long"]
    var_name_abbr = names["var_abbr"]
    ens_var_name = names["ens_var"]
    unit = names["unit"]

    labels: Dict[str, Dict[str, Any]] = {}

    # Title box
    simulation_reduction_start_fmtd: Optional[str]
    simulation_now_fmtd: Optional[str]
    if all(
        simulation_mdata == next(iter(simulation_mdata_lst))
        for simulation_mdata in simulation_mdata_lst
    ):
        simulation_mdata = next(iter(simulation_mdata_lst))
        simulation_integr_period = (
            simulation_mdata.now - simulation_mdata.reduction_start
        )
        integr_period_fmtd = format_integr_period(
            simulation_integr_period, setup, words, cap=True
        )
        simulation_reduction_start_fmtd = format_meta_datum(
            simulation_mdata.reduction_start
        )
        simulation_now_fmtd = format_meta_datum(simulation_mdata.now)
        simulation_lead_time_fmtd = f"+{format_meta_datum(simulation_mdata.lead_time)}"
        time_since_release_start_fmtd = format_meta_datum(
            simulation_mdata.now_rel - release_mdata.start_rel
        )
    else:
        # Simulation meta data differ for multipanel plots with multiple time steps
        simulation_integr_period_lst = []
        simulation_lead_time_lst = []
        time_since_release_start_lst = []
        for simulation_mdata in simulation_mdata_lst:
            simulation_integr_period_lst.append(
                simulation_mdata.now - simulation_mdata.reduction_start
            )
            simulation_lead_time_lst.append(simulation_mdata.lead_time)
            time_since_release_start_lst.append(
                simulation_mdata.now_rel - release_mdata.start_rel
            )
        if len(set(simulation_integr_period_lst)) == 1:
            simulation_integr_period_lst = [next(iter(simulation_integr_period_lst))]
        if len(set(simulation_lead_time_lst)) == 1:
            simulation_lead_time_lst = [next(iter(simulation_lead_time_lst))]
        if len(set(time_since_release_start_lst)) == 1:
            time_since_release_start_lst = [next(iter(time_since_release_start_lst))]
        # SR_TMP <
        integr_period_fmtd_lst = [
            format_integr_period(simulation_integr_period, setup, words, cap=True)
            for simulation_integr_period in simulation_integr_period_lst
        ]
        integr_period_fmtd = ""
        suffix = r"$\,$h"
        for i, s in enumerate(integr_period_fmtd_lst):
            assert s.endswith(suffix), s
            s = s[: -len(suffix)]
            if i == 0:
                integr_period_fmtd += (
                    s[::-1].split(" ", 1)[1][::-1]
                    + f" {words['previous']} "
                    + s[::-1].split(" ", 1)[0][::-1]
                )
            else:
                s_prev = integr_period_fmtd_lst[i - 1]
                assert s_prev.endswith(suffix), s
                s_prev = s_prev[: -len(suffix)]
                assert s[::-1].split(" ", 1)[1] == s_prev[::-1].split(" ", 1)[1], s
                integr_period_fmtd += r"$\,$/$\,$" + s[::-1].split(" ", 1)[0][::-1]
        integr_period_fmtd += suffix
        # SR_TMP >
        simulation_reduction_start_fmtd = None
        simulation_now_fmtd = None
        # SR_TMP <
        simulation_lead_time_fmtd = ""
        suffix = r"$\,$h"
        for i, simulation_lead_time in enumerate(simulation_lead_time_lst):
            s = f"+{format_meta_datum(simulation_lead_time)}"
            assert s.endswith(suffix), s
            s = s[: -len(suffix)]
            if i > 0:
                simulation_lead_time_fmtd += r"$\,$/$\,$"
            simulation_lead_time_fmtd += s
        simulation_lead_time_fmtd += suffix
        # SR_TMP >
        # SR_TMP <
        suffix = r"$\,$h"
        time_since_release_start_fmtd = ""
        for i, time_since_release_start in enumerate(time_since_release_start_lst):
            s = format_meta_datum(time_since_release_start)
            assert s.endswith(suffix), s
            s = s[: -len(suffix)]
            if i > 0:
                time_since_release_start_fmtd += r"$\,$/$\,$"
            time_since_release_start_fmtd += s
        time_since_release_start_fmtd += suffix
        # SR_TMP <
    labels["title"] = {}
    labels["title"]["tl"] = capitalize(
        format_names_etc(setup, words, variable_mdata)["long"]
    )
    labels["title"]["bl"] = capitalize(f"{integr_period_fmtd}")
    if simulation_reduction_start_fmtd is not None:
        labels["title"][
            "bl"
        ] += f" ({words['since']} {simulation_reduction_start_fmtd})"
    if simulation_now_fmtd is not None:
        labels["title"]["tr"] = capitalize(f"{simulation_now_fmtd}")
    labels["title"]["br"] = capitalize(
        f"{time_since_release_start_fmtd}" f" {words['after']} {words['release_start']}"
    )

    # Data info box
    labels["data_info"] = {
        "lines": [],
    }
    labels["data_info"]["lines"].append(
        f"{words['substance'].c}:"
        f"\t{format_meta_datum(species_mdata.name, join_values=' / ')}",
    )
    labels["data_info"]["lines"].append(
        f"{words['input_variable'].c}:\t{capitalize(var_name_abbr)}"
    )
    if plot_variable == "concentration":
        labels["data_info"]["lines"].append(
            f"{words['height'].c}:"
            f"\t{escape_format_keys(format_level_label(variable_mdata, words))}"
        )
    if setup.model.simulation_type == "ensemble":
        if ens_variable == "probability":
            op = {"lower": "gt", "upper": "lt"}[ens_param_thr_type]
            if plot_type == "multipanel" and multipanel_param == "ens_params.thr":
                labels["data_info"]["lines"].append(
                    f"{words['selection']}:\t{symbols[op]}"
                    f" {format_meta_datum(unit=format_meta_datum(variable_mdata.unit))}"
                )
                labels["data_info"]["lines"].append(
                    f"\t({', '.join(map(str, ens_param_thrs))})"
                )
            else:
                labels["data_info"]["lines"].append(
                    f"{words['selection']}:\t{symbols[op]} {ens_param_thr}"
                    f" {format_meta_datum(unit=format_meta_datum(variable_mdata.unit))}"
                )
        elif ens_variable in [
            "cloud_arrival_time",
            "cloud_departure_time",
        ]:
            labels["data_info"]["lines"].append(
                # f"{words['cloud_density']}:\t{words['minimum', 'abbr']}"
                # f"{words['threshold']}:\t"
                f"{words['cloud_threshold', 'abbr']}:\t {ens_param_thr}"
                f" {format_meta_datum(unit=format_meta_datum(variable_mdata.unit))}"
            )
            n_min = ens_param_mem_min or 0
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
            f" {escape_format_keys(format_level_label(variable_mdata, words))}"
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
    # SR_TMP <
    if not all(
        simulation_mdata.start == next(iter(simulation_mdata_lst)).start
        for simulation_mdata in simulation_mdata_lst
    ):
        raise NotImplementedError(
            "simulation starts differ:\n"
            + "\n".join(map(pformat, simulation_mdata_lst))
        )
    simulation_start = next(iter(simulation_mdata_lst)).start
    # SR_TMP >
    release_start = cast(datetime, simulation_start) + cast(
        timedelta, release_mdata.start_rel
    )
    release_end = cast(datetime, simulation_start) + cast(
        timedelta, release_mdata.end_rel
    )
    release_start_fmtd = format_meta_datum(release_start)
    release_end_fmtd = format_meta_datum(release_end)
    site_name = format_meta_datum(release_mdata.site_name)
    site_lat_lon = format_release_site_coords_labels(words, symbols, release_mdata)
    release_height = format_meta_datum(
        release_mdata.height, release_mdata.height_unit
    ).replace("meters", r"$\,$" + words["m_agl"].s)
    release_rate = format_meta_datum(release_mdata.rate, release_mdata.rate_unit)
    release_mass = format_meta_datum(release_mdata.mass, release_mdata.mass_unit)
    substance = format_meta_datum(species_mdata.name, join_values=" / ")
    half_life = format_meta_datum(species_mdata.half_life, species_mdata.half_life_unit)
    deposit_vel = format_meta_datum(
        species_mdata.deposition_velocity,
        species_mdata.deposition_velocity_unit,
    )
    sediment_vel = format_meta_datum(
        species_mdata.sedimentation_velocity,
        species_mdata.sedimentation_velocity_unit,
    )
    washout_coeff = format_meta_datum(
        species_mdata.washout_coefficient,
        species_mdata.washout_coefficient_unit,
    )
    washout_exponent = format_meta_datum(species_mdata.washout_exponent)
    lines_parts: List[Optional[Tuple[WordT, str]]]
    if setup.layout.type == "standalone_details":
        lines_parts = [
            (words["site"], site_name),
            (words["latitude"], site_lat_lon[0]),
            (words["longitude"], site_lat_lon[1]),
            (words["height"], release_height),
            (words["start"], release_start_fmtd),
            (words["end"], release_end_fmtd),
        ]
    else:
        lines_parts = [
            (words["site"], site_name),
            (words["latitude"], site_lat_lon[0]),
            (words["longitude"], site_lat_lon[1]),
            (words["height"], release_height),
            None,
            (words["start"], release_start_fmtd),
            (words["end"], release_end_fmtd),
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

    def format_lines_block(parts: Sequence[Optional[Tuple[WordT, str]]]) -> str:
        block = ""
        for line_parts in parts:
            if line_parts is None:
                block += "\n\n"
            else:
                left, right = line_parts
                block += f"{capitalize(left)}:\t{right}\n"
        return block

    labels["release_info"] = {
        "title": capitalize(words["release"].t),
        "lines_str": format_lines_block(lines_parts),
    }

    if setup.layout.type == "standalone_details":
        labels["standalone_release_info"] = {
            "title": capitalize(words["release"].t),
            "lines_str": format_lines_block(
                [
                    (words["start"], release_start_fmtd),
                    (words["end"], release_end_fmtd),
                    (words["rate"], release_rate),
                    (words["total_mass"], release_mass),
                    None,
                    (words["substance"], substance),
                    (words["half_life"], half_life),
                    (words["deposition_velocity", "abbr"], deposit_vel),
                    (words["sedimentation_velocity", "abbr"], sediment_vel),
                    (words["washout_coeff"], washout_coeff),
                    (words["washout_exponent"], washout_exponent),
                ],
            ),
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


def format_names_etc(
    setup: PlotSetup, words: TranslatedWords, variable_mdata: VariableMetaData
) -> Dict[str, str]:
    # SR_TMP <
    plot_type = setup.layout.plot_type
    multipanel_param = setup.layout.multipanel_param
    if plot_type == "multipanel" and multipanel_param == "ens_variable":
        ens_variable = "+".join(setup.panels.collect("ens_variable"))
    else:
        ens_variable = setup.panels.collect_equal("ens_variable")
    if plot_type == "multipanel" and multipanel_param == "ens_params.pctl":
        pass
    else:
        ens_param_pctl = setup.panels.collect_equal("ens_params.pctl")
    lang = setup.panels.collect_equal("lang")
    # SR_TMP >

    plot_variable = setup.panels.collect_equal("plot_variable")
    integrate = setup.panels.collect_equal("integrate")

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
    def _format_unit(
        setup: PlotSetup, words: TranslatedWords, variable_mdata: VariableMetaData
    ) -> str:
        if setup.model.simulation_type == "ensemble":
            if ens_variable == "probability":
                return "%"
            elif ens_variable in [
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                return f"{words['hour', 'pl']}"
        return format_meta_datum(unit=format_meta_datum(variable_mdata.unit))

    unit = _format_unit(setup, words, variable_mdata)
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
            if plot_type == "multipanel" and multipanel_param == "ens_params.pctl":
                ens_var_name = f"{words['percentile', 'pl']}"
            else:
                ens_var_name = (
                    f"{ordinal(ens_param_pctl, 'g', lang)} {words['percentile']}"
                )
            long_name = f"{ens_var_name} {var_name_rel}"
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
                setup.layout.plot_type == "multipanel"
                and setup.layout.multipanel_param == "ens_variable"
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


def capitalize(s: Union[str, WordT]) -> str:
    """Capitalize the first letter while leaving all others as they are."""
    s = str(s)
    if not s:
        return s
    try:
        return s[0].upper() + s[1:]
    except Exception as e:
        raise ValueError(f"string not capitalizable: '{s}'") from e


def format_max_marker_label(labels: Dict[str, Any], max_val: Optional[float]) -> str:
    if max_val is None:
        return labels["max"]
    if np.isnan(max_val):
        s_val = "NaN"
    else:
        if 0.001 <= max_val < 0.01:
            s_val = f"{max_val:.5f}"
        elif 0.01 <= max_val < 0.1:
            s_val = f"{max_val:.4f}"
        elif 0.1 <= max_val < 1:
            s_val = f"{max_val:.3f}"
        elif 1 <= max_val < 10:
            s_val = f"{max_val:.2f}"
        elif 10 <= max_val < 100:
            s_val = f"{max_val:.1f}"
        elif 100 <= max_val < 1000:
            s_val = f"{max_val:.0f}"
        else:
            s_val = f"{max_val:.2E}"
        # s_val += r"$\,$" + labels["unit"]
    return f"{labels['max']}: {s_val}"


def format_release_site_coords_labels(
    words: TranslatedWords, symbols: Words, release_mdata: ReleaseMetaData
) -> Tuple[str, str]:
    lat_deg_fmt = capitalize(format_coord_label("north", words, symbols))
    lon_deg_fmt = capitalize(format_coord_label("east", words, symbols))
    lat = Degrees(release_mdata.lat)
    lon = Degrees(release_mdata.lon)
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
        elif model_name in ["COSMO-E", "COSMO-1E", "COSMO-2E"]:
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


def format_level_label(variable_mdata: VariableMetaData, words: TranslatedWords) -> str:
    unit = variable_mdata.level_unit
    if unit == "meters":
        unit = words["m_agl"].s
    level = format_vertical_level_range(
        variable_mdata.bottom_level, variable_mdata.top_level, unit
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
    period: timedelta,
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


def get_cloud_timing_levels(simulation_duration_hours: int) -> np.ndarray:
    """Derive levels for cloud timing plots from simulation duration."""
    if simulation_duration_hours <= 21:
        return np.arange(0, simulation_duration_hours + 1, 3)
    levels = [0, 3, 6, 9, 12, 18] + list(range(24, simulation_duration_hours, 12))
    if levels[-1] < simulation_duration_hours:
        levels += [simulation_duration_hours]
    return np.array(levels)


def levels_from_time_stats(n_levels: int, val_max: float) -> np.ndarray:
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
