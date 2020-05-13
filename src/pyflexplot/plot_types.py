# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap
from pydantic import BaseModel

# Local
from .data import Field
from .formatting import format_range
from .meta_data import MetaData
from .meta_data import format_unit
from .meta_data import get_integr_type
from .plot_lib import MapAxesConf
from .setup import InputSetup
from .words import TranslatedWords
from .words import Words

# Custom types
ColorType = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
FontSizeType = Union[str, float]
RectType = Tuple[float, float, float, float]


def create_map_conf(field: Field) -> MapAxesConf:
    domain = field.var_setups.collect_equal("domain")
    model = field.nc_meta_data["analysis"]["model"]

    conf_base = {"lang": field.var_setups.collect_equal("lang")}

    conf_model_ifs = {"geo_res": "50m"}
    conf_model_cosmo = {"geo_res": "10m"}

    # SR_TMP < TODO generalize based on meta data
    conf_domain_japan = {
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
    conf_domain_eu = {
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 300_000,
        # SR_DBG <
        "zoom_fact": 1.02,
        # "zoom_fact": 0.975,  # SR_DBG
        # SR_DBG >
    }
    conf_domain_ch = {
        "geo_res_cities": "10m",
        "geo_res_rivers": "10m",
        "min_city_pop": 0,
        "ref_dist_conf": {"dist": 25},
        "rel_offset": (-0.02, 0.045),
    }

    if (model, domain) in [("cosmo1", "auto"), ("cosmo2", "auto")]:
        conf = {**conf_base, **conf_model_cosmo, **conf_domain_eu}
    elif (model, domain) == ("cosmo1", "ch"):
        conf = {**conf_base, **conf_model_cosmo, **conf_domain_ch, "zoom_fact": 3.6}
    elif (model, domain) == ("cosmo2", "ch"):
        conf = {**conf_base, **conf_model_cosmo, **conf_domain_ch, "zoom_fact": 3.2}
    elif (model, domain) == ("ifs", "auto"):
        # SR_TMP < TODO Generalize IFS domains
        conf = {**conf_base, **conf_model_ifs, **conf_domain_japan}
        # SR_TMP >
    else:
        raise Exception(f"unknown domain '{domain}' for model '{model}'")

    return MapAxesConf(**conf)


@dataclass
class FontSizes:
    title_large: FontSizeType = 14.0
    title_medium: FontSizeType = 12.0
    title_small: FontSizeType = 12.0
    content_large: FontSizeType = 12.0
    content_medium: FontSizeType = 10.0
    content_small: FontSizeType = 9.0


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class PlotLayout:
    aspect: float
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    w_right: float = 0.2
    h_rigtop: float = 0.15
    h_rigbot: float = 0.42
    h_bottom: float = 0.05

    def __post_init__(self):
        self.x_pad: float = self.y_pad / self.aspect
        self.h_tot: float = self.y1_tot - self.y0_tot
        self.y1_top: float = self.y1_tot
        self.y0_top: float = self.y1_top - self.h_top
        self.y1_center: float = self.y0_top - self.y_pad
        self.y0_center: float = self.y0_tot + self.h_bottom
        self.h_center: float = self.y1_center - self.y0_center
        self.w_tot: float = self.x1_tot - self.x0_tot
        self.x1_right: float = self.x1_tot
        self.x0_right: float = self.x1_right - self.w_right
        self.x1_left: float = self.x0_right - self.x_pad
        self.x0_left: float = self.x0_tot
        self.w_left: float = self.x1_left - self.x0_left
        self.y0_rigbot: float = self.y0_center
        self.y1_rigtop: float = self.y1_tot
        self.y0_rigtop: float = self.y1_rigtop - self.h_rigtop
        self.y1_rigbot: float = self.y0_rigbot + self.h_rigbot
        self.y1_rigmid: float = self.y0_rigtop - self.y_pad
        self.y0_rigmid: float = self.y1_rigbot + self.y_pad
        self.h_rigmid: float = self.y1_rigmid - self.y0_rigmid
        self.rect_top: RectType = (self.x0_left, self.y0_top, self.w_left, self.h_top)
        self.rect_center: RectType = (
            self.x0_left,
            self.y0_center,
            self.w_left,
            self.h_center,
        )
        self.rect_right_top: RectType = (
            self.x0_right,
            self.y0_rigtop,
            self.w_right,
            self.h_rigtop,
        )
        self.rect_right_middle: RectType = (
            self.x0_right,
            self.y0_rigmid,
            self.w_right,
            self.h_rigmid,
        )
        self.rect_right_bottom: RectType = (
            self.x0_right,
            self.y0_rigbot,
            self.w_right,
            self.h_rigbot,
        )
        self.rect_bottom: RectType = (
            self.x0_tot,
            self.y0_tot,
            self.w_tot,
            self.h_bottom,
        )


# SR_TODO Move class definition to pyflexplot.plot
# pylint: disable=R0902  # too-many-instance-attributes
class PlotConfig(BaseModel):
    setup: InputSetup  # SR_TODO consider removing this
    mdata: MetaData  # SR_TODO consider removing this
    cmap: Union[str, Colormap] = "flexplot"
    d_level: Optional[int] = None  # SR_TODO sensible default
    draw_colors: bool = True
    draw_contours: bool = False
    extend: str = "max"
    # SR_NOTE Figure size may change when boxes etc. are added
    # SR_TODO Specify plot size in a robust way (what you want is what you get)
    fig_size: Tuple[float, float] = (12.5, 8.0)
    font_name: str = "Liberation Sans"
    font_sizes: FontSizes = FontSizes()
    labels: Dict[str, Any] = {}
    legend_rstrip_zeros: bool = True
    level_ranges_align: str = "center"
    level_range_style: str = "base"
    levels_scale: str = "log"
    lw_frame: float = 1.0
    mark_field_max: bool = True
    mark_release_site: bool = True
    markers: Dict[str, Dict[str, Any]] = {
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
    model_info: str = ""  # SR_TODO sensible default
    n_levels: Optional[int] = None  # SR_TODO sensible default

    class Config:  # noqa
        allow_extra = False
        arbitrary_types_allowed = True

    @property
    def fig_aspect(self):
        return np.divide(*self.fig_size)


def capitalize(s: str) -> str:
    """Capitalize the first letter while leaving all others as they are."""
    if not s:
        return s
    try:
        return s[0].upper() + s[1:]
    except Exception:
        raise ValueError("s not capitalizable", s)


# SR_TODO Create dataclass with default values for text box setup
# pylint: disable=R0912  # too-many-branches
# pylint: disable=R0914  # too-many-locals
# pylint: disable=R0915  # too-many-statements
def create_plot_config(
    setup: InputSetup, words: TranslatedWords, symbols: Words, mdata: MetaData,
) -> "PlotConfig":
    words.set_active_lang(setup.lang)
    new_config_dct: Dict[str, Any] = {
        "setup": setup,
        "mdata": mdata,
        "labels": {},
    }
    new_config_dct["labels"]["top"] = {
        "tl": "",  # SR_TMP force into 1st position of dict (for tests)
        "bl": (
            f"{format_integr_period(mdata, setup, words, cap=True)}"
            f" {words['since']}"
            f" +{mdata.simulation_integr_start_rel}"
        ),
        "tr": (f"{mdata.simulation_now} (+{mdata.simulation_now_rel})"),
        "br": (
            f"{mdata.simulation_now_rel} {words['since']}"
            f" {words['release_start']}"
            f" {words['at', 'place']} {mdata.release_site_name}"
        ),
    }
    new_config_dct["labels"]["right_top"] = {
        "title": f"{words['data'].c}",
        "lines": [],
    }
    new_config_dct["labels"]["right_middle"] = {
        "title": "",  # SR_TMP Force into 1st position in dict (for tests)
        "title_unit": "",  # SR_TMP Force into 2nd position in dict (for tests)
        "release_site": words["release_site"].s,
        "max": words["max"].s,
    }
    new_config_dct["labels"]["right_bottom"] = {
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
        "max": words["max"].s,
        "name": words["substance"].s,
        "half_life": words["half_life"].s,
        "deposit_vel": words["deposition_velocity", "abbr"].s,
        "sediment_vel": words["sedimentation_velocity", "abbr"].s,
        "washout_coeff": words["washout_coeff"].s,
        "washout_exponent": words["washout_exponent"].s,
    }
    _info_fmt_base = (
        f"{words['flexpart']} {words['based_on']}"
        f" {mdata.simulation_model_name}{{ens}},"
        f" {mdata.simulation_start}"
    )
    new_config_dct["labels"]["bottom"] = {
        "model_info_det": _info_fmt_base.format(ens=""),
        "model_info_ens": _info_fmt_base.format(
            ens=(
                f" {words['ensemble']}"
                f" ({len(setup.ens_member_id or [])} {words['member', 'pl']}:"
                f" {format_range(setup.ens_member_id or [], fmt='02d')})"
            )
        ),
        "copyright": f"{symbols['copyright']}{words['meteoswiss']}",
    }

    long_name = ""
    short_name = ""
    var_name = ""
    unit = mdata.format("variable_unit")

    if setup.variable == "concentration":
        new_config_dct["n_levels"] = 8
        new_config_dct["labels"]["right_middle"]["tc"] = (
            f"{words['level']}:" f" {escape_format_keys(format_level_label(mdata))}"
        )
        var_name = str(words["activity_concentration"])
        if setup.integrate:
            long_name = f"{words['integrated']} {var_name}"
            short_name = (
                f"{words['integrated', 'abbr']} {words['concentration', 'abbr']}"
            )
            variable_rel = (
                f"{words['of', 'fg']} {words['integrated', 'g']}"
                f" {words['activity_concentration']}"
            )
        else:
            long_name = var_name
            short_name = str(words["concentration"])
            variable_rel = f"{words['of', 'fg']} {words['activity_concentration']}"
        new_config_dct["labels"]["right_top"]["lines"].insert(
            0, f"{words['level'].c}:\t{escape_format_keys(format_level_label(mdata))}"
        )

    elif setup.variable == "deposition":
        dep_type_word = (
            "total" if setup.deposition_type == "tot" else setup.deposition_type
        )
        var_name = f"{words[dep_type_word, 'f']} {words['surface_deposition']}"
        long_name = var_name
        short_name = str(words["deposition"])
        variable_rel = (
            f"{words['of', 'fg']} {words[dep_type_word, 'g']}"
            f" {words['surface_deposition']}"
        )
        new_config_dct["n_levels"] = 9

    if setup.simulation_type == "deterministic":
        new_config_dct["model_info"] = new_config_dct["labels"]["bottom"][
            "model_info_det"
        ]
        if setup.plot_type.startswith("affected_area"):
            long_name = f"{words['affected_area']} {variable_rel}"
            if setup.plot_type == "affected_area_mono":
                new_config_dct["extend"] = "none"
                new_config_dct["n_levels"] = 1

    elif setup.simulation_type == "ensemble":
        new_config_dct["model_info"] = new_config_dct["labels"]["bottom"][
            "model_info_ens"
        ]
        if setup.plot_type == "ens_min":
            long_name = f"{words['ensemble_minimum']} {variable_rel}"
        elif setup.plot_type == "ens_max":
            long_name = f"{words['ensemble_maximum']} {variable_rel}"
        elif setup.plot_type == "ens_median":
            long_name = f"{words['ensemble_median']} {variable_rel}"
        elif setup.plot_type == "ens_mean":
            long_name = f"{words['ensemble_mean']} {variable_rel}"
        elif setup.plot_type == "ens_thr_agrmt":
            new_config_dct.update(
                {
                    "extend": "min",
                    "n_levels": 7,
                    "d_level": 2,
                    "legend_rstrip_zeros": False,
                    "level_range_style": "int",
                    "level_ranges_align": "left",
                    "mark_field_max": False,
                    "levels_scale": "lin",
                }
            )
            new_config_dct["labels"]["right_top"]["lines"].append(
                f"{words['cloud']}:\t{symbols['geq']} {setup.ens_param_thr}"
                f" {mdata.format('variable_unit')}"
            )
            short_name = f"{words['number_of', 'abbr'].c} " f"{words['member', 'pl']}"
            long_name = (
                f"{words['threshold_agreement']}"
                f" ({words['number_of', 'abbr'].c} {words['member', 'pl']})"
            )
        elif setup.plot_type in ["ens_cloud_arrival_time", "ens_cloud_departure_time"]:
            new_config_dct.update(
                {
                    "extend": "both",
                    "n_levels": 9,
                    "d_level": 3,
                    "legend_rstrip_zeros": False,
                    "level_range_style": "int",
                    "level_ranges_align": "left",
                    "mark_field_max": False,
                    "levels_scale": "lin",
                }
            )
            new_config_dct["labels"]["right_top"]["lines"].append(
                f"{words['cloud_density'].c}:\t{words['minimum', 'abbr']}"
                f" {setup.ens_param_thr} {mdata.format('variable_unit')}"
            )
            n_min: int = setup.ens_param_mem_min or 1
            n_tot: int = len((setup.ens_member_id or []))
            new_config_dct["labels"]["right_top"]["lines"].append(
                f"{words['number_of', 'abbr'].c} {words['member', 'pl']}:"
                f"\t{words['minimum', 'abbr']} {setup.ens_param_mem_min}"
                r"$\,/\,$"
                f"{n_tot} ({n_min/(n_tot or 1):.0%})"
            )
            if setup.plot_type == "ens_cloud_arrival_time":
                long_name = f"{words['cloud_arrival_time']}"
                short_name = f"{words['arrival']}"
            elif setup.plot_type == "ens_cloud_departure_time":
                long_name = f"{words['cloud_departure_time']}"
                short_name = f"{words['departure']}"
            unit = f"{words['hour', 'pl']}"

    # SR_TMP <
    if not short_name:
        raise Exception("no short name")
    if not long_name:
        raise Exception("no short name")
    # SR_TMP >

    new_config_dct["labels"]["top"][
        "tl"
    ] = f"{long_name} {words['of']} {mdata.species_name}"
    new_config_dct["labels"]["right_top"]["lines"].insert(
        0, f"{words['input_variable'].c}:\t{capitalize(var_name)}",
    )
    new_config_dct["labels"]["right_middle"]["title"] = short_name
    new_config_dct["labels"]["right_middle"]["title_unit"] = f"{short_name} ({unit})"

    # Capitalize all labels
    for labels in new_config_dct["labels"].values():
        for name, label in labels.items():
            if name == "lines":
                label = [capitalize(line) for line in label]
            else:
                label = capitalize(label)
            labels[name] = label

    return PlotConfig(**new_config_dct)


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


def colors_from_plot_config(plot_config: PlotConfig) -> Sequence[ColorType]:
    assert plot_config.n_levels is not None  # mypy
    if plot_config.setup.plot_type == "affected_area_mono":
        return (np.array([(200, 200, 200)]) / 255).tolist()
    elif plot_config.cmap == "flexplot":
        return colors_flexplot(plot_config.n_levels, plot_config.extend)
    else:
        cmap = mpl.cm.get_cmap(plot_config.cmap)
        return [
            cmap(i / (plot_config.n_levels - 1)) for i in range(plot_config.n_levels)
        ]


def levels_from_time_stats(
    plot_config: PlotConfig, time_stats: Mapping[str, float]
) -> List[float]:
    def _auto_levels_log10(n_levels: int, val_max: float) -> List[float]:
        if not np.isfinite(val_max):
            raise ValueError("val_max not finite", val_max)
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        return 10 ** np.arange(
            log10_max - (n_levels - 1) * log10_d, log10_max + 0.5 * log10_d, log10_d
        )

    assert plot_config.n_levels is not None  # mypy
    if plot_config.setup.plot_type == "ens_thr_agrmt":
        n_max = 20  # SR_TMP SR_HC
        assert plot_config.d_level is not None  # mypy
        return (
            np.arange(
                n_max - plot_config.d_level * (plot_config.n_levels - 1),
                n_max + plot_config.d_level,
                plot_config.d_level,
            )
            + 1
        )
    elif plot_config.setup.plot_type in [
        "ens_cloud_arrival_time",
        "ens_cloud_departure_time",
    ]:
        assert plot_config.d_level is not None  # mypy
        return np.arange(0, plot_config.n_levels) * plot_config.d_level
    elif plot_config.setup.plot_type == "affected_area_mono":
        levels = _auto_levels_log10(n_levels=9, val_max=time_stats["max"])
        return np.array([levels[0], np.inf])
    else:
        return _auto_levels_log10(plot_config.n_levels, val_max=time_stats["max"])


def escape_format_keys(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def format_level_label(mdata: MetaData) -> str:
    unit = mdata.variable_level_bot_unit.value
    if mdata.variable_level_top_unit.value != unit:
        raise Exception(
            "inconsistent level units",
            mdata.variable_level_bot_unit,
            mdata.variable_level_top_unit,
        )
    assert isinstance(unit, str)  # mypy
    level = format_vertical_level_range(
        mdata.variable_level_bot.value, mdata.variable_level_top.value, unit
    )
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
            raise NotImplementedError(f"3 non-continuous level ranges")
    else:
        raise NotImplementedError(f"{n} sets of levels")


def format_integr_period(
    mdata: "MetaData", setup: InputSetup, words: TranslatedWords, cap: bool = False
) -> str:
    integr_type = get_integr_type(setup)
    if integr_type == "mean":
        operation = words["averaged_over"].s
    elif integr_type == "sum":
        operation = words["summed_over"].s
    elif integr_type == "accum":
        operation = words["accumulated_over"].s
    start = mdata.simulation_integr_start.value
    now = mdata.simulation_now.value
    assert isinstance(start, datetime)  # mypy
    assert isinstance(now, datetime)  # mypy
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
