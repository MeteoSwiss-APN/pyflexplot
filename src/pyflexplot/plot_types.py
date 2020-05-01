# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap

# Local
from .data import Field
from .meta_data import MetaData
from .meta_data import format_unit
from .meta_data import get_integr_type
from .plot_lib import MapAxesConf
from .setup import InputSetup
from .utils import format_level_range
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS
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


# SR_TMP TODO Turn into dataclass or the like.
@summarizable
class PlotLabels:

    words: TranslatedWords = WORDS
    symbols: Words = SYMBOLS

    def __init__(self, lang: str, mdata: MetaData):
        """Create an instance of ``PlotLabels``."""
        self.mdata: MetaData = mdata
        self.setup: InputSetup = mdata.setup

        self.words.set_active_lang(lang)

        self.top_left: Dict[str, Any] = self._init_group(
            {
                "variable": (
                    f"{get_long_name(self.setup, self.words)}"
                    f"{format_level_label(self.mdata, self.words)}"
                ),
                "period": (
                    f"{format_integr_period(self.mdata, self.setup, self.words)} "
                    f"({self.words['since']} +{self.mdata.simulation_integr_start_rel})"
                ),
                "subtitle_thr_agrmt_fmt": (
                    f"Cloud: {self.symbols['geq']} {{thr}} "
                    f"{escape_format_keys(self.mdata.format('variable_unit'))}"
                ),
                "subtitle_cloud_arrival_time": (
                    f"Cloud: {self.symbols['geq']} {{thr}} "
                    f"{escape_format_keys(self.mdata.format('variable_unit'))}"
                    f"; members: {self.symbols['geq']} {{mem}}"
                ),
                "timestep": (
                    f"{self.mdata.simulation_now} (+{self.mdata.simulation_now_rel})"
                ),
                "time_since_release_start": (
                    f"{self.mdata.simulation_now_rel} "
                    f"{self.words['since']} "
                    f"{self.words['release_start']}"
                ),
            },
        )

        self.top_right: Dict[str, Any] = self._init_group(
            {
                "species": f"{self.mdata.species_name}",
                "site": f"{self.words['site']}: {self.mdata.release_site_name}",
            },
        )

        self.right_top: Dict[str, Any] = self._init_group(
            {
                "title": get_short_name(self.setup, self.words),
                "title_unit": (
                    f"{get_short_name(self.setup, self.words)} "
                    f"({self.mdata.format('variable_unit')})"
                ),
                "release_site": self.words["release_site"].s,
                "max": self.words["max"].s,
            },
        )

        self.right_bottom: Dict[str, Any] = self._init_group(
            {
                "title": self.words["release"].t,
                "start": self.words["start"].s,
                "end": self.words["end"].s,
                "latitude": self.words["latitude"].s,
                "longitude": self.words["longitude"].s,
                "lat_deg_fmt": format_coord_label("north", self.words, self.symbols),
                "lon_deg_fmt": format_coord_label("east", self.words, self.symbols),
                "height": self.words["height"].s,
                "rate": self.words["rate"].s,
                "mass": self.words["total_mass"].s,
                "site": self.words["site"].s,
                "release_site": self.words["release_site"].s,
                "max": self.words["max"].s,
                "name": self.words["substance"].s,
                "half_life": self.words["half_life"].s,
                "deposit_vel": self.words["deposition_velocity", "abbr"].s,
                "sediment_vel": self.words["sedimentation_velocity", "abbr"].s,
                "washout_coeff": self.words["washout_coeff"].s,
                "washout_exponent": self.words["washout_exponent"].s,
            },
        )

        # SR_TMP < TODO un-hardcode
        n_members = "X"  # SR_HC
        # ens_member_id = "{:03d}-{:03d}".format(0, 20)  # SR_HC
        ens_member_id = "XX-YY"  # SR_HC
        # SR_TMP >
        s_ens = (
            f" {self.words['ensemble']} "
            f"({n_members} {self.words['member', 'pl']}: {ens_member_id})"
        )
        info_fmt_base = (
            f"{self.words['flexpart']} {self.words['based_on']} "
            f"{self.mdata.simulation_model_name}{{ens}}, "
            f"{self.mdata.simulation_start}"
        )
        self.bottom: Dict[str, Any] = self._init_group(
            {
                "model_info_det": info_fmt_base.format(ens=""),
                "model_info_ens": info_fmt_base.format(ens=s_ens),
                "copyright": f"{self.symbols['copyright']}{self.words['meteoswiss']}",
            },
        )

    def _init_group(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and set group values."""
        for key, val in values.copy().items():
            if isinstance(val, str):
                # Capitalize first letter only (even if it's a space!)
                values[key] = list(val)[0].capitalize() + val[1:]
        return values


def create_plot_labels(setup: InputSetup, mdata: MetaData):
    return PlotLabels(setup.lang, mdata)


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
    if not unit:
        pass
    assert isinstance(unit, str)  # mypy
    level = format_level_range(
        mdata.variable_level_bot.value, mdata.variable_level_top.value, unit
    )
    if not level:
        return ""
    return f" {words['at', 'level']} {format_unit(level)}"


def format_integr_period(
    mdata: "MetaData", setup: InputSetup, words: TranslatedWords
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
    period = (now - start).total_seconds() / 3600
    return f"{operation} {period:g}$\\,$h"


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
def get_long_name(setup: InputSetup, words: TranslatedWords) -> str:
    if setup.plot_type and setup.plot_type.startswith("affected_area"):
        super_name = get_long_name(
            setup.derive({"variable": "deposition", "plot_type": "auto"}), words
        )
        return f"{words['affected_area']} ({super_name})"
    elif setup.plot_type == "ens_thr_agrmt":
        super_name = get_short_name(setup.derive({"variable": "deposition"}), words)
        return f"{words['threshold_agreement']} ({super_name})"
    elif setup.plot_type == "ens_cloud_arrival_time":
        return f"{words['cloud_arrival_time']}"
    if setup.variable == "deposition":
        if setup.deposition_type == "tot":
            word = "total"
        else:
            assert isinstance(setup.deposition_type, str)  # mypy
            word = setup.deposition_type
        dep = words[word, "f"].s
        if setup.plot_type == "ens_min":
            return f"{words['ensemble_minimum']} {dep} {words['surface_deposition']}"
        elif setup.plot_type == "ens_max":
            return f"{words['ensemble_maximum']} {dep} {words['surface_deposition']}"
        elif setup.plot_type == "ens_median":
            return f"{words['ensemble_median']} {dep} {words['surface_deposition']}"
        elif setup.plot_type == "ens_mean":
            return f"{words['ensemble_mean']} {dep} {words['surface_deposition']}"
        else:
            return f"{dep} {words['surface_deposition']}"
    if setup.variable == "concentration":
        if setup.plot_type == "ens_min":
            return f"{words['ensemble_minimum']} {words['concentration']}"
        elif setup.plot_type == "ens_max":
            return f"{words['ensemble_maximum']} {words['concentration']}"
        elif setup.plot_type == "ens_median":
            return f"{words['ensemble_median']} {words['concentration']}"
        elif setup.plot_type == "ens_mean":
            return f"{words['ensemble_mean']} {words['concentration']}"
        else:
            ctx = "abbr" if setup.integrate else "*"
            return words["activity_concentration", ctx].s
    raise NotImplementedError(
        f"long_name", setup.variable, setup.plot_type, setup.integrate
    )


def get_short_name(setup: InputSetup, words: TranslatedWords) -> str:
    if setup.variable == "concentration":
        if setup.plot_type == "ens_cloud_arrival_time":
            return f"{words['arrival'].c} ({words['hour', 'pl']})"
        else:
            if setup.integrate:
                return (
                    f"{words['integrated', 'abbr']} "
                    f"{words['concentration', 'abbr']}"
                )
            return words["concentration"].s
    if setup.variable == "deposition":
        if setup.plot_type == "ens_thr_agrmt":
            return f"{words['number_of', 'abbr'].c} " f"{words['member', 'pl']}"
        else:
            return words["deposition"].s
    raise NotImplementedError("short_name", setup.variable, setup.plot_type)


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class PlotConfig:
    setup: InputSetup  # SR_TODO consider removing this
    labels: PlotLabels  # SR_TODO consider removing this
    cmap: Union[str, Colormap] = "flexplot"
    d_level: Optional[int] = None  # SR_TODO sensible default
    draw_colors: bool = True
    draw_contours: bool = False
    extend: str = "max"
    # SR_NOTE Figure size may change when boxes etc. are added
    # SR_TODO Specify plot size in a robust way (what you want is what you get)
    fig_size: Tuple[float, float] = (12.5, 8.0)
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
    top_box_subtitle: str = ""

    @property
    def fig_aspect(self):
        return np.divide(*self.fig_size)


# SR_TODO Create dataclass with default values for test box setup
def create_plot_config(setup: InputSetup, labels: PlotLabels) -> "PlotConfig":
    new_config_dct: Dict[str, Any] = {}
    if setup.variable == "concentration":
        new_config_dct["n_levels"] = 8
    elif setup.variable == "deposition":
        new_config_dct["n_levels"] = 9
    if setup.simulation_type == "deterministic":
        new_config_dct["model_info"] = labels.bottom["model_info_det"]
        if setup.plot_type == "affected_area_mono":
            new_config_dct["extend"] = "none"
            new_config_dct["n_levels"] = 1
    if setup.simulation_type == "ensemble":
        new_config_dct["model_info"] = labels.bottom["model_info_ens"]
        if setup.plot_type == "ens_thr_agrmt":
            new_config_dct["extend"] = "min"
            new_config_dct["n_levels"] = 7
            new_config_dct["d_level"] = 2
            new_config_dct["legend_rstrip_zeros"] = False
            new_config_dct["level_range_style"] = "int"
            new_config_dct["level_ranges_align"] = "left"
            new_config_dct["mark_field_max"] = False
            new_config_dct["top_box_subtitle"] = labels.top_left[
                "subtitle_thr_agrmt_fmt"
            ].format(thr=setup.ens_param_thr)
            new_config_dct["legend_box_title"] = labels.right_top["title"]
            new_config_dct["levels_scale"] = "lin"
        elif setup.plot_type == "ens_cloud_arrival_time":
            new_config_dct["extend"] = "max"
            new_config_dct["n_levels"] = 9
            new_config_dct["d_level"] = 3
            new_config_dct["legend_rstrip_zeros"] = False
            new_config_dct["level_range_style"] = "int"
            new_config_dct["level_ranges_align"] = "left"
            new_config_dct["mark_field_max"] = False
            new_config_dct["top_box_subtitle"] = labels.top_left[
                "subtitle_cloud_arrival_time"
            ].format(thr=setup.ens_param_thr, mem=setup.ens_param_mem_min)
            new_config_dct["legend_box_title"] = labels.right_top["title"]
            new_config_dct["levels_scale"] = "lin"
    # SR_TMP < TODO set default
    if "legend_box_title" not in new_config_dct:
        new_config_dct["legend_box_title"] = labels.right_top["title_unit"]
    # SR_TMP >
    return PlotConfig(setup=setup, labels=labels, **new_config_dct)


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
