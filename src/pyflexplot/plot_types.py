# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
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
from .meta_data import MetaData
from .meta_data import format_unit
from .meta_data import get_integr_type
from .plot_lib import MapAxesConf
from .setup import InputSetup
from .words import TranslatedWords
from .words import Words


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


def escape_format_keys(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def format_level_label(mdata: MetaData, words: TranslatedWords):
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
    # return f" {words['at', 'level']} {format_unit(level)}"
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
    mdata: "MetaData",
    setup: InputSetup,
    words: TranslatedWords,
    capitalize: bool = False,
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
    if capitalize:
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


# pylint: disable=R0911,R0912  # too-many-return-statements,too-many-branches
def get_long_name(
    setup: InputSetup, words: TranslatedWords, capitalize: bool = False
) -> str:
    s: str = ""
    if setup.plot_type and setup.plot_type.startswith("affected_area"):
        super_name = get_long_name(
            setup.derive({"variable": "deposition", "plot_type": "auto"}), words
        )
        s = f"{words['affected_area']} ({super_name})"
    elif setup.plot_type == "ens_thr_agrmt":
        super_name = get_short_name(setup.derive({"variable": "deposition"}), words)
        s = f"{words['threshold_agreement']} ({super_name})"
    elif setup.plot_type == "ens_cloud_arrival_time":
        s = f"{words['cloud_arrival_time']}"
    elif setup.variable == "deposition":
        if setup.deposition_type == "tot":
            word = "total"
        else:
            assert isinstance(setup.deposition_type, str)  # mypy
            word = setup.deposition_type
        dep = words[word, "f"].s
        if setup.plot_type == "ens_min":
            s = f"{words['ensemble_minimum']} {dep} {words['surface_deposition']}"
        elif setup.plot_type == "ens_max":
            s = f"{words['ensemble_maximum']} {dep} {words['surface_deposition']}"
        elif setup.plot_type == "ens_median":
            s = f"{words['ensemble_median']} {dep} {words['surface_deposition']}"
        elif setup.plot_type == "ens_mean":
            s = f"{words['ensemble_mean']} {dep} {words['surface_deposition']}"
        else:
            s = f"{dep} {words['surface_deposition']}"
    elif setup.variable == "concentration":
        if setup.plot_type == "ens_min":
            s = f"{words['ensemble_minimum']} {words['concentration']}"
        elif setup.plot_type == "ens_max":
            s = f"{words['ensemble_maximum']} {words['concentration']}"
        elif setup.plot_type == "ens_median":
            s = f"{words['ensemble_median']} {words['concentration']}"
        elif setup.plot_type == "ens_mean":
            s = f"{words['ensemble_mean']} {words['concentration']}"
        else:
            ctx = "abbr" if setup.integrate else "*"
            s = str(words["activity_concentration", ctx])
    if not s:
        raise NotImplementedError(
            f"long_name", setup.variable, setup.plot_type, setup.integrate
        )
    if capitalize:
        s = s[0].upper() + s[1:]
    return s


def get_short_name(
    setup: InputSetup, words: TranslatedWords, capitalize: bool = False
) -> str:
    s: str = ""
    if setup.variable == "concentration":
        if setup.plot_type == "ens_cloud_arrival_time":
            s = f"{words['arrival'].c} ({words['hour', 'pl']})"
        else:
            if setup.integrate:
                s = (
                    f"{words['integrated', 'abbr']}"
                    f" {words['concentration', 'abbr']}"
                )
            else:
                s = str(words["concentration"])
    elif setup.variable == "deposition":
        if setup.plot_type == "ens_thr_agrmt":
            s = f"{words['number_of', 'abbr'].c} " f"{words['member', 'pl']}"
        else:
            s = str(words["deposition"])
    if not s:
        raise NotImplementedError("short_name", setup.variable, setup.plot_type)
    if capitalize:
        s = s[0].upper() + s[1:]
    return s


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
    labels: Dict[str, Any] = {}
    legend_box_title: str = ""  # SR_TODO sensible default
    legend_rstrip_zeros: bool = True
    level_ranges_align: str = "center"
    level_range_style: str = "base"
    levels_scale: str = "log"
    lw_frame: float = 1.0
    mark_field_max: bool = True
    mark_release_site: bool = True
    model_info: str = ""  # SR_TODO sensible default
    n_levels: Optional[int] = None  # SR_TODO sensible default
    reverse_legend: bool = False

    class Config:  # noqa
        allow_extra = False
        arbitrary_types_allowed = True

    @property
    def fig_aspect(self):
        return np.divide(*self.fig_size)


# SR_TODO Create dataclass with default values for test box setup
def create_plot_config(
    setup: InputSetup, words: TranslatedWords, symbols: Words, mdata: MetaData,
) -> "PlotConfig":
    words.set_active_lang(setup.lang)
    new_config_dct: Dict[str, Any] = {
        "setup": setup,
        "mdata": mdata,
        "labels": {},
    }
    new_config_dct["labels"]["top_left"] = {
        "tl": (
            f"{get_long_name(setup, words, capitalize=True)}"
            f" {words['of']} {mdata.species_name}"
        ),
        "bl": (
            f"{format_integr_period(mdata, setup, words, capitalize=True)}"
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
    new_config_dct["labels"]["top_right"] = {}
    new_config_dct["labels"]["right_top"] = {
        "title": get_short_name(setup, words, capitalize=True),
        "title_unit": (
            f"{get_short_name(setup, words, capitalize=True)}"
            f" ({mdata.format('variable_unit')})"
        ),
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
                f" ({'X'} {words['member', 'pl']}: {'XX-YY'})"  # SR_TODO unhardcode!
            )
        ),
        "copyright": f"{symbols['copyright']}{words['meteoswiss']}",
    }
    if setup.variable == "concentration":
        new_config_dct["n_levels"] = 8
        new_config_dct["labels"]["top_right"]["tc"] = (
            f"{words['level']}:"
            f" {escape_format_keys(format_level_label(mdata, words))}"
        )
    elif setup.variable == "deposition":
        new_config_dct["n_levels"] = 9
    if setup.simulation_type == "deterministic":
        new_config_dct["model_info"] = new_config_dct["labels"]["bottom"][
            "model_info_det"
        ]
        if setup.plot_type == "affected_area_mono":
            new_config_dct["extend"] = "none"
            new_config_dct["n_levels"] = 1
    if setup.simulation_type == "ensemble":
        new_config_dct["model_info"] = new_config_dct["labels"]["bottom"][
            "model_info_ens"
        ]
        if setup.plot_type == "ens_thr_agrmt":
            new_config_dct.update(
                {
                    "extend": "min",
                    "n_levels": 7,
                    "d_level": 2,
                    "legend_rstrip_zeros": False,
                    "level_range_style": "int",
                    "level_ranges_align": "left",
                    "mark_field_max": False,
                    "legend_box_title": new_config_dct["labels"]["right_top"]["title"],
                    "levels_scale": "lin",
                }
            )
            new_config_dct["labels"]["top_right"]["tc"] = (
                f"{words['cloud']}: {symbols['geq']} {setup.ens_param_thr}"
                f" {mdata.format('variable_unit')}"
            )
        elif setup.plot_type == "ens_cloud_arrival_time":
            new_config_dct.update(
                {
                    "extend": "max",
                    "n_levels": 9,
                    "d_level": 3,
                    "legend_rstrip_zeros": False,
                    "level_range_style": "int",
                    "level_ranges_align": "left",
                    "mark_field_max": False,
                    "legend_box_title": new_config_dct["labels"]["right_top"]["title"],
                    "levels_scale": "lin",
                }
            )
            new_config_dct["labels"]["top_right"]["tc"] = (
                f"{words['cloud']}: {symbols['geq']} {setup.ens_param_thr}"
                f" {mdata.format('variable_unit')}"
            )
            new_config_dct["labels"]["top_right"]["bc"] = (
                f"{words['member', 'pl']}: {symbols['geq']}"
                f" {setup.ens_param_mem_min}"
            )
    # SR_TMP < TODO set default
    if "legend_box_title" not in new_config_dct:
        new_config_dct["legend_box_title"] = new_config_dct["labels"]["right_top"][
            "title_unit"
        ]
    # SR_TMP >
    return PlotConfig(**new_config_dct)


def colors_flexplot(n_levels: int, extend: str) -> List[Tuple[int, int, int]]:

    # color_under = [i/255.0 for i in (255, 155, 255)]
    color_under = [i / 255.0 for i in (200, 200, 200)]
    color_over = [i / 255.0 for i in (200, 200, 200)]

    colors_core_8 = (
        np.array(
            [
                (224, 196, 172),
                (221, 127, 215),
                (99, 0, 255),
                (100, 153, 199),
                (34, 139, 34),
                (93, 255, 2),
                (199, 255, 0),
                (255, 239, 57),
            ],
            float,
        )
        / 255
    ).tolist()

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


def colors_from_plot_config(plot_config: PlotConfig) -> List[Tuple[int, int, int]]:
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
    elif plot_config.setup.plot_type == "ens_cloud_arrival_time":
        assert plot_config.d_level is not None  # mypy
        return np.arange(0, plot_config.n_levels) * plot_config.d_level
    elif plot_config.setup.plot_type == "affected_area_mono":
        levels = _auto_levels_log10(n_levels=9, val_max=time_stats["max"])
        return np.array([levels[0], np.inf])
    else:
        return _auto_levels_log10(plot_config.n_levels, val_max=time_stats["max"])
