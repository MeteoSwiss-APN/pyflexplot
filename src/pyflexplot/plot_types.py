# -*- coding: utf-8 -*-
"""
Plot types.
"""
# Standard library
from dataclasses import dataclass
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
from .meta_data import format_integr_period
from .meta_data import format_level_range
from .plot_lib import MapAxesConf
from .setup import InputSetup
from .utils import summarizable
from .words import SYMBOLS
from .words import WORDS
from .words import TranslatedWords
from .words import Words


def create_map_conf(field: Field) -> MapAxesConf:
    domain = field.var_setups.collect_equal("domain")
    model = field.nc_meta_data["analysis"]["model"]

    conf_base = {"lang": field.var_setups.collect_equal("lang")}

    conf_model_ifs = {"geo_res": "110m"}
    conf_model_cosmo = {"geo_res": "10m"}

    conf_domain_global = {
        "geo_res_cities": "110m",
        "geo_res_rivers": "110m",
        "min_city_pop": 1_000_000,
        "zoom_fact": 1.02,
    }
    conf_domain_eu = {
        "geo_res_cities": "50m",
        "geo_res_rivers": "50m",
        "min_city_pop": 300_000,
        "zoom_fact": 1.02,
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
        conf = {**conf_base, **conf_model_ifs, **conf_domain_global}
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

        self.words.set_default_lang(lang)

        # Declare groups
        self.top_left: Dict[str, Any] = self._init_top_left()
        self.top_right: Dict[str, Any] = self._init_top_right()
        self.right_top: Dict[str, Any] = self._init_right_top()
        self.right_bottom: Dict[str, Any] = self._init_right_bottom()
        self.bottom: Dict[str, Any] = self._init_bottom()

    def _init_group(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and set group values."""
        for key, val in values.copy().items():
            if isinstance(val, str):
                # Capitalize first letter only (even if it's a space!)
                values[key] = list(val)[0].capitalize() + val[1:]
        return values

    def _init_top_left(self) -> Dict[str, Any]:
        value_bottom = self.mdata.variable_level_bot.value
        value_top = self.mdata.variable_level_top.value
        unit_bottom = self.mdata.variable_level_bot_unit.value
        unit_top = self.mdata.variable_level_top_unit.value
        assert isinstance(unit_bottom, str)  # mypy
        assert isinstance(unit_top, str)  # mypy
        level = format_level_range(value_bottom, value_top, unit_bottom, unit_top)
        if not level:
            s_level = ""
        else:
            s_level = f" {self.words['at', None, 'level']} {level}"
        assert isinstance(self.mdata.simulation_integr_type.value, str)  # mypy
        integr_op = self.words[
            {
                "sum": "summed_over",
                "mean": "averaged_over",
                "accum": "accumulated_over",
            }[self.mdata.simulation_integr_type.value]
        ].s
        time_rels = self.mdata.simulation_now_rel
        period = format_integr_period(self.mdata)
        start = self.mdata.simulation_integr_start_rel
        unit = self.mdata.variable_unit
        unit_escaped = str(unit).replace("{", "{{").replace("}", "}}")
        ts_abs = self.mdata.simulation_now
        ts_rel = self.mdata.simulation_now_rel
        return self._init_group(
            {
                "variable": f"{get_long_name(self.mdata.setup, self.words)}{s_level}",
                "period": f"{integr_op} {period} ({self.words['since']} +{start})",
                "subtitle_thr_agrmt_fmt": (
                    f"Cloud: {self.symbols['geq']} {{thr}} {unit_escaped}"
                ),
                "subtitle_cloud_arrival_time": (
                    f"Cloud: {self.symbols['geq']} {{thr}} {unit_escaped}; "
                    f"members: {self.symbols['geq']} {{mem}}"
                ),
                "timestep": f"{ts_abs} (+{ts_rel})",
                "time_since_release_start": (
                    f"{time_rels} {self.words['since']} {self.words['release_start']}"
                ),
            },
        )

    def _init_top_right(self) -> Dict[str, Any]:
        return self._init_group(
            {
                "species": f"{self.mdata.species_name}",
                "site": f"{self.words['site']}: {self.mdata.release_site_name.value}",
            },
        )

    def _init_right_top(self) -> Dict[str, Any]:
        name = get_short_name(self.mdata.setup, self.words)
        unit = self.mdata.variable_unit
        return self._init_group(
            {
                "title": f"{name}",
                "title_unit": f"{name} ({unit})",
                "release_site": self.words["release_site"].s,
                "max": self.words["max"].s,
            },
        )

    def _init_right_bottom(self) -> Dict[str, Any]:
        deg_ = f"{self.symbols['deg']}{self.symbols['short_space']}"
        north = f"{self.symbols['short_space']}{self.words['north', None, 'abbr']}"
        east = f"{self.symbols['short_space']}{self.words['east', None, 'abbr']}"
        return self._init_group(
            {
                "title": self.words["release"].t,
                "start": self.words["start"].s,
                "end": self.words["end"].s,
                "latitude": self.words["latitude"].s,
                "longitude": self.words["longitude"].s,
                "lat_deg_fmt": (
                    f"{{d}}{deg_}{{m}}'{north} ({{f:.4f}}{self.words['degN']})"
                ),
                "lon_deg_fmt": (
                    f"{{d}}{deg_}{{m}}'{east} ({{f:.4f}}{self.words['degE']})"
                ),
                "height": self.words["height"].s,
                "rate": self.words["rate"].s,
                "mass": self.words["total_mass"].s,
                "site": self.words["site"].s,
                "release_site": self.words["release_site"].s,
                "max": self.words["max"].s,
                "name": self.words["substance"].s,
                "half_life": self.words["half_life"].s,
                "deposit_vel": self.words["deposition_velocity", None, "abbr"].s,
                "sediment_vel": self.words["sedimentation_velocity", None, "abbr"].s,
                "washout_coeff": self.words["washout_coeff"].s,
                "washout_exponent": self.words["washout_exponent"].s,
            },
        )

    def _init_bottom(self) -> Dict[str, Any]:
        model = self.mdata.simulation_model_name.value
        n_members = 21  # SR_TMP SR_HC TODO un-hardcode
        ens_member_id = "{:03d}-{:03d}".format(0, 20)  # SR_TMP SR_HC TODO un-hardcode
        model_ens = (
            f"{model} {self.words['ensemble']} ({n_members} "
            f"{self.words['member', None, 'pl']}: {ens_member_id}"
        )
        start = self.mdata.simulation_start
        model_info_fmt = (
            f"{self.words['flexpart']} {self.words['based_on']} {model}, {start}"
        )
        return self._init_group(
            {
                "model_info_det": model_info_fmt.format(m=model),
                "model_info_ens": model_info_fmt.format(m=model_ens),
                "copyright": f"{self.symbols['copyright']}{self.words['meteoswiss']}",
            },
        )


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
        dep = words[word, None, "f"].s
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
            return words["activity_concentration", None, ctx].s
    raise NotImplementedError(
        f"long_name", setup.variable, setup.plot_type, setup.integrate
    )


def get_short_name(setup: InputSetup, words: TranslatedWords) -> str:
    if setup.variable == "concentration":
        if setup.plot_type == "ens_cloud_arrival_time":
            return f"{words['arrival'].c} ({words['hour', None, 'pl']}??)"
        else:
            if setup.integrate:
                return (
                    f"{words['integrated', None, 'abbr']} "
                    f"{words['concentration', None, 'abbr']}"
                )
            return words["concentration"].s
    if setup.variable == "deposition":
        if setup.plot_type == "ens_thr_agrmt":
            return (
                f"{words['number_of', None, 'abbr'].c} "
                f"{words['member', None, 'pl']}"
            )
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
    figsize: Tuple[float, float] = (12.0, 9.0)
    legend_box_title: str = ""  # SR_TODO sensible default
    level_ranges_align: str = "center"
    level_range_style: str = "base"
    levels_scale: str = "log"
    lw_frame: float = 1.0
    mark_field_max: bool = True
    mark_release_site: bool = True
    model_info: str = ""  # SR_TODO sensible default
    n_levels: Optional[int] = None  # SR_TODO sensible default
    reverse_legend: bool = False
    text_box_setup: Optional[Dict[str, float]] = None  # SR_TODO sensible default
    top_box_subtitle: str = ""

    @classmethod
    def create(cls, setup: InputSetup, labels: PlotLabels) -> "PlotConfig":
        new_config_dct: Dict[str, Any] = {}
        if setup.variable == "concentration":
            new_config_dct["n_levels"] = 8
        elif setup.variable == "deposition":
            new_config_dct["n_levels"] = 9
        if setup.simulation_type == "deterministic":
            new_config_dct["text_box_setup"] = {
                "h_rel_t": 0.1,
                "h_rel_b": 0.03,
                "w_rel_r": 0.25,
                "pad_hor_rel": 0.015,
                "h_rel_box_rt": 0.45,
            }
            new_config_dct["model_info"] = labels.bottom["model_info_det"]
            if setup.plot_type == "affected_area_mono":
                new_config_dct["extend"] = "none"
                new_config_dct["n_levels"] = 1
        if setup.simulation_type == "ensemble":
            new_config_dct["text_box_setup"] = {
                "h_rel_t": 0.14,
                "h_rel_b": 0.03,
                "w_rel_r": 0.25,
                "pad_hor_rel": 0.015,
                "h_rel_box_rt": 0.46,
            }
            new_config_dct["model_info"] = labels.bottom["model_info_ens"]
            if setup.plot_type == "ens_thr_agrmt":
                new_config_dct["extend"] = "min"
                new_config_dct["n_levels"] = 7
                new_config_dct["d_level"] = 2
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
                new_config_dct["n_levels"] = 5
                new_config_dct["d_level"] = 3
                new_config_dct["level_range_style"] = "int"
                new_config_dct["level_ranges_align"] = "left"
                new_config_dct["mark_field_max"] = False
                new_config_dct["top_box_subtitle"] = labels.top_left[
                    "subtitle_cloud_arrival_time"
                ].format(thr=setup.ens_param_thr, mem=setup.ens_param_mem_min)
                new_config_dct["legend_box_title"] = labels.right_top["title"]
                new_config_dct["levels_scale"] = "lin"
        # SR_TMP < TODO set default
        if "lebend_box_title" not in new_config_dct:
            new_config_dct["legend_box_title"] = labels.right_top["title_unit"]
        # SR_TMP >
        return cls(setup=setup, labels=labels, **new_config_dct)


def colors_flexplot(n_levels: int, extend: str) -> List[Tuple[int, int, int]]:

    # color_under = [i/255.0 for i in (255, 155, 255)]
    color_under = [i / 255.0 for i in (200, 200, 200)]
    color_over = [i / 255.0 for i in (200, 200, 200)]

    colors_core_8 = (
        np.array(
            [
                (224, 196, 172),  #
                (221, 127, 215),  #
                (99, 0, 255),  #
                (100, 153, 199),  #
                (34, 139, 34),  #
                (93, 255, 2),  #
                (199, 255, 0),  #
                (255, 239, 57),  #
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
