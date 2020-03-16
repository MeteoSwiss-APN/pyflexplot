#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test summarizing plots as defined in ``pyflexplot.plot``.

Plots can be summarized into dicts, which can be used to test that all plot
elements are in the right place with the right content etc.

(Summarization is similar to serialization, but without the ability to
reconstruct the objects from the summary.)

In this module, the summary dict of one plot type is thoroughly tested to
ensure that summarization works as expected. In order to isolate this process
as best as possible, several classes used to define labels etc. are mocked:
``TranslatedWord``, ``TranslatedWords``, ``Attr``.

While the mocks allows us to inject dummy strings as labels etc. and makes the
tests independent of the exact formation of those, the tests are still
dependent on the structure of ``Plot``, i.e., what elements there are
and where they go.

Not every failure thus implies that summarization is broken! There might just
have been minor adjustments to the plot layout.

The tests are organized in such a way that a comprehensive reference summary
dict is constructed near the bottom, and each test compares a subset of it
against the respective part(s) of the summary dict obtained from the tested
plot class.

All tests come in pairs: For each positive test (check that elements present in
both the result and solution summary dicts match) there is a negative test that
ensures failure in case of wrong content. This prevents false positive, i.e.,
passing tests because of missing elements in either dict.

Some dict elements are explicitly ignored in comparisons by aid of
``IgnoredElement``. This allows us to test, for instance that the correct
number of lebend labels are present without testing for the labels themselves,
or to avoid hard-coding lat/lon coordinates in the solution.

"""
# Standard library
from types import SimpleNamespace

# Third-party
import numpy as np

# First-party
from pyflexplot.data import Field
from pyflexplot.plot import Plot
from pyflexplot.plot import PlotLabels
from pyflexplot.plot import PlotSetup
from pyflexplot.setup import Setup
from srutils.testing import CheckFailedError
from srutils.testing import IgnoredElement
from srutils.testing import UnequalElement
from srutils.testing import check_summary_dict_is_subdict
from words import TranslatedWord
from words import TranslatedWords
from words import Word
from words import Words

# Classes for dummy input data


def dummy__str__(parent, word, lang, ctx):
    details = []
    if str(lang) != "None":
        details.append(lang)
    if str(ctx) != "None":
        details.append(ctx)
    if details:
        s_details = f"[{'|'.join(details)}]"
    else:
        s_details = ""
    return f"<{parent}.{word}{s_details}>"


class Dummy_Word(Word):
    def __str__(self):
        try:
            ctx = self.ctx
        except AttributeError:
            ctx = self._parent._pop_curr_ctx()
        return dummy__str__(self._parent._parent.name, self._s, self.lang, ctx)


class Dummy_TranslatedWord(TranslatedWord):
    """Wrapper for ``TranslatedWord`` class for testing."""

    @property
    def cls_word(self):
        class Wrapped_Dummy_Word(Dummy_Word):
            _parent = self

        return Wrapped_Dummy_Word

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._curr_ctx = None

    def __str__(self):
        ctx = self._pop_curr_ctx()
        return dummy__str__(self._parent.name, self.name, self.lang, ctx)

    def _pop_curr_ctx(self):
        ctx = self._curr_ctx
        self._curr_ctx = None
        return ctx

    def s(self):
        return str(self)

    @property
    def lang(self):
        return self._parent.lang

    def ctx(self, name):
        self._curr_ctx = name
        return self


class Dummy_Words(Words):
    @property
    def cls_word(self):
        class Wrapped_Dummy_TranslatedWord(Dummy_TranslatedWord):
            _parent = self

        return Wrapped_Dummy_TranslatedWord

    @classmethod
    def create(cls, name, words):
        self = cls(name, {word: word for word in words})
        return self


class Dummy_TranslatedWords(TranslatedWords):
    """Wrapper for ``TranslatedWords`` class for testing."""

    @property
    def cls_word(self):
        class Wrapped_Dummy_TranslatedWord(Dummy_TranslatedWord):
            _parent = self

        return Wrapped_Dummy_TranslatedWord

    @classmethod
    def create(cls, name, lang, words):
        words_langs = {w: {"en": w, "de": w} for w in words}
        self = cls(name, words_langs, default_lang=lang)
        self.lang = lang
        return self

    def get(self, name, lang=None, ctx=None, **kwargs):
        word = super().get(name, chainable=True)
        word._curr_ctx = ctx
        return super().get(name, lang, ctx, **kwargs)


class Dummy_Attr:
    """Replacement for ``Attr`` class for testing."""

    def __init__(self, name, value, lang):
        name = f"<{name}[{lang}]>"
        self._name = name
        self.value = value or name
        self.lang = lang
        self.unit = f"{name}.unit"

    # def __str__(self):
    #     return self._name

    def format(self, *args, **kwargs):
        return f"<{self._name[1:-1]}.format>"


# Prepare test data


def create_attrs(lang):

    # Note: Some values must be passed, otherwise plotting fails

    def DA(name, value=None):
        return Dummy_Attr(name, value, lang=lang)

    def fm(name, value=None):
        return DA(name, value).format

    return SimpleNamespace(
        summarize=lambda: {},
        grid=SimpleNamespace(
            north_pole_lat=DA("grid.north_pole_lat", 43.0),
            north_pole_lon=DA("grid.north_pole_lat", -170),
        ),
        release=SimpleNamespace(
            height=DA("release.height"),
            mass=DA("release.mass"),
            rate=DA("release.rate"),
            start=DA("release.start"),
            end=DA("release.end"),
            site_lat=DA("release.site_lat", 47.0),
            site_lon=DA("release.site_lon", 8.0),
            site_name=DA("release.site_name"),
        ),
        simulation=SimpleNamespace(
            end=DA("simulation.end"),
            fmt_integr_period=fm("simulation.fmt_integr_period"),
            integr_start=DA("simulation.integr_start"),
            integr_type=DA("simulation.integr_type", "mean"),
            model_name=DA("simulation.model_name"),
            now=DA("simulation.now"),
            start=DA("simulation.start"),
        ),
        species=SimpleNamespace(
            deposit_vel=DA("species.deposit_vel"),
            half_life=DA("species.half_life"),
            name=DA("species.name"),
            sediment_vel=DA("species.sediment_vel"),
            washout_coeff=DA("species.washout_coeff"),
            washout_exponent=DA("species.washout_exponent"),
        ),
        variable=SimpleNamespace(
            fmt_level_range=fm("variable.fmt_level_range"),
            long_name=DA("variable.long_name"),
            short_name=DA("variable.short_name"),
            unit=DA("variable.unit"),
        ),
    )


def create_field():
    dummy_field = Field(
        fld=np.array([[i] * 10 for i in range(10)], np.float32),
        rlat=np.arange(-5.0, 4.1, 1.0),
        rlon=np.arange(-6.0, 3.1, 1.0),
        fld_specs=None,
        time_stats={"max": 15},
    )
    return dummy_field


def create_setup(lang):
    infile = "dummy_infile.nc"
    outfile = "dummy_outfile.png"
    return Setup.create(
        {"infile": infile, "outfile": outfile, "lang": lang, "variable": "deposition"},
    )


def create_words(lang):

    w = Dummy_TranslatedWords.create(
        "words",
        lang,
        [
            "accumulated_over",
            "at",
            "averaged_over",
            "based_on",
            "degE",
            "degN",
            "deposition_velocity",
            "east",
            "end",
            "ensemble",
            "flexpart",
            "half_life",
            "height",
            "latitude",
            "longitude",
            "max",
            "member",
            "meteoswiss",
            "north",
            "rate",
            "release",
            "release_site",
            "release_start",
            "sedimentation_velocity",
            "since",
            "site",
            "start",
            "substance",
            "summed_up_over",
            "total_mass",
            "washout_coeff",
            "washout_exponent",
        ],
    )
    return w


def create_symbols():

    return Dummy_Words.create(
        "symbols", ["ae", "copyright", "deg", "geq", "oe", "ue", "short_space", "t0"]
    )


def create_labels(lang, attrs):
    PlotLabels.words = create_words(lang)
    PlotLabels.symbols = create_symbols()
    return PlotLabels(lang, attrs)


def create_plot_setup(lang):
    setup = create_setup(lang)
    attrs = create_attrs(lang)
    labels = create_labels(lang, attrs)
    return PlotSetup(setup, attrs, labels)


def create_map_conf(lang):
    return SimpleNamespace(
        zoom_fact=1.0,
        rel_offset=(0, 0),
        ref_dist_on=True,
        ref_dist_conf=SimpleNamespace(pos="bl", unit="km", dist=100),
        geo_res="10m",
        geo_res_rivers="50m",
        geo_res_cities="50m",
        min_city_pop=0,
        lang=lang,
        lw_frame=1,
    )


# Create result


def create_res(lang, _cache={}):
    """Create test result once, then return it from cache.

    Based on mostly dummy input data (defined above), a dispersion plot object
    is created and summarized into a dict. The same result dict is used in all
    test, but compared against solution dicts of varying detail. Each solution
    dict constitutes a subset of the results dict.

    A test passes if all elements in the solution dict are present and equal in
    the results dict, while all other elements in the results dict are ignored.

    """
    if lang not in _cache:
        field = create_field()
        plot_setup = create_plot_setup(lang)
        map_conf = create_map_conf(lang)
        plot = Plot(field, plot_setup, map_conf)
        _cache[lang] = plot.summarize()
    return _cache[lang]


# Run tests


class CreateTests:
    """Decorator to test a given results set in multiple languages.

    Each test is run for all combinations of ``langs`` (languages) and ``invs``
    (whether to run an inverse test; see below).

    By default (langs=['en', 'de], invs=[False, True]), four tests are created
    for the given argument setup:
        - a regular test (inv=False) in English (lang='en'),
        - an inverted test (inv=True) in English (lang='en'),
        - a regular test (inv=False) in German (lang='de'), and
        - an inverted test (inv=True) in German (lang='de').

    In an inverse test, the values of the tested attributes are replaced by
    instances of ``UnequalElement``, which always return False in comparisons.
    This means that for the test to pass, the result must differ from the
    solution. This ensures that the regular tests pass because of successful
    comparisons.

    """

    langs = ["en", "de"]
    invs = [False, True]
    args = [
        "base_attrs",
        "ax_map",
        "boxes",
        "boxes_text",
        "fig",
        "field",
    ]

    def __init__(self, langs=None, invs=None, args=None):
        if langs:
            self.langs = langs
        if invs:
            self.invs = invs
        if args:
            self.args = args

    def __call__(outer_self, cls):
        for inv in outer_self.invs:
            for lang in outer_self.langs:

                def f(inner_self, lang=lang, inverted=inv):
                    outer_self.run_test(inner_self, lang=lang, inverted=inverted)

                setattr(cls, f"test_{lang}{'_inv' if inv else ''}", f)
        return cls

    def run_test(outer_self, inner_self, lang, inverted):
        res = create_res(lang)
        kwargs = {s: getattr(inner_self, s, False) for s in outer_self.args}
        sol = create_sol(lang, inverted=inverted, **kwargs)
        try:
            check_summary_dict_is_subdict(
                superdict=res, subdict=sol, supername="result", subname="solution"
            )
        except CheckFailedError as e:
            if inverted:
                return
            raise AssertionError(e.args[0], *e.args[1:]) from e
        else:
            if inverted:
                raise AssertionError(f"inverted test passed")


@CreateTests()
class Test_BaseAttrs:
    base_attrs = True


@CreateTests()
class Test_Boxes:
    boxes = True


@CreateTests()
class Test_BoxesText:
    boxes = True
    boxes_text = True


@CreateTests()
class Test_Fig:
    fig = True


@CreateTests()
class Test_Field:
    field = True


@CreateTests()
class Test_Full:
    base_attrs = True
    ax_map = True
    boxes = True
    boxes_text = True
    fig = True
    field = True


# Create solutions


def create_sol(lang, inverted, **kwargs):
    return Solution(lang, inverted).create(**kwargs)


class Solution:
    def __init__(self, lang, inverted):
        self.lang = lang
        self.inverted = inverted

    def create(self, *, base_attrs, ax_map, boxes, boxes_text, fig, field):
        sol = {}
        if base_attrs:
            sol.update(self.base_attrs())
        if ax_map:
            sol.update(self.ax_map())
        if boxes:
            sol.update(self.boxes(check_text=boxes_text))
        if fig:
            sol.update(self.fig())
        if field:
            sol.update(self.field())
        return sol

    def element(self, e):
        if self.inverted:
            return UnequalElement(str(e))
        return e

    def base_attrs(self):
        e = self.element
        jdat = {
            "type": e("Plot"),
            # SR_TMP "extend": e("max"),
            # SR_TMP "level_range_style": e("base"),
            "draw_colors": e(True),
            "draw_contours": e(False),
            # SR_TMP "mark_field_max": e(True),
            "mark_release_site": e(True),
            "plot_setup": IgnoredElement("PlotSetupTMP"),  # SR_TMP TODO
            # SR_TMP "setup": IgnoredElement("Setup"),  # SR_TMP TODO
        }
        return jdat

    def ax_map(self):
        e = self.element
        jdat = {
            "type": e("MapAxesRotatedPole"),
        }
        return {"ax_map": jdat}

    def boxes(self, check_text=True):

        if not check_text:
            # Outer elements set up to fail in inverted test
            # Inner elements (text strings etc.) always ignored
            e1 = self.element
            e2 = IgnoredElement
        else:
            # Outer elements NOT set up to fail in inverted test
            # Inner elements set up to fail in inverted test
            e1 = lambda x: x  # noqa:E731
            e2 = self.element

        def txt(loc, s, **kwargs):
            """Shortcut for text box element."""
            return {
                "type": "TextBoxElement_Text",
                "loc": {"loc": loc},
                "s": e2(s),
                **kwargs,
            }

        def col(loc, fc, **kwargs):
            """Shortcut for color rect element."""
            return {
                "type": "TextBoxElement_ColorRect",
                "loc": {"loc": loc},
                "fc": e2(fc),
                **kwargs,
            }

        def mkr(loc, m, **kwargs):
            """Shortcut for marker element."""
            return {
                "type": "TextBoxElement_Marker",
                "loc": {"loc": loc},
                "m": e2(m),
                **kwargs,
            }

        sl = f"[{self.lang}]"

        box_top_left = {
            "type": "TextBoxAxes",
            "name": e1("top/left"),
            "elements": e1(
                [
                    txt(
                        "tl",
                        f"<variable.long_name{sl}> <words.at[{self.lang}|level]> "
                        f"<variable.fmt_level_range{sl}.format>",
                    ),
                    txt(
                        "bl",
                        f"<words.averaged_over{sl}> "
                        f"<simulation.fmt_integr_period{sl}.format> (<words.since{sl}> "
                        f"+<simulation.integr_start{sl}.format>)",
                    ),
                    txt(
                        "tr",
                        f"<simulation.now{sl}.format> (+<simulation.now{sl}.format>)",
                    ),
                    txt(
                        "br",
                        f"<simulation.now{sl}.format> <words.since{sl}> "
                        f"<words.release_start{sl}>",
                    ),
                ]
            ),
        }
        box_top_right = {
            "type": "TextBoxAxes",
            "name": e1("top/right"),
            "elements": e1(
                [
                    txt("tc", f"<species.name{sl}.format>"),
                    txt("bc", IgnoredElement("release site (may be truncated")),
                ]
            ),
        }
        box_right_top = {
            "type": "TextBoxAxes",
            "name": e1("right/top"),
            "elements": e1(
                [
                    txt(
                        "tc",
                        f"<variable.short_name{sl}.format> "
                        f"(<variable.unit{sl}.format>)",
                    ),
                    txt("bc", IgnoredElement("level range #0")),
                    txt("bc", IgnoredElement("level range #1")),
                    txt("bc", IgnoredElement("level range #2")),
                    txt("bc", IgnoredElement("level range #3")),
                    txt("bc", IgnoredElement("level range #4")),
                    txt("bc", IgnoredElement("level range #5")),
                    txt("bc", IgnoredElement("level range #6")),
                    txt("bc", IgnoredElement("level range #7")),
                    txt("bc", IgnoredElement("level range #8")),
                    col("bc", IgnoredElement("face color #0")),
                    col("bc", IgnoredElement("face color #1")),
                    col("bc", IgnoredElement("face color #2")),
                    col("bc", IgnoredElement("face color #3")),
                    col("bc", IgnoredElement("face color #4")),
                    col("bc", IgnoredElement("face color #5")),
                    col("bc", IgnoredElement("face color #6")),
                    col("bc", IgnoredElement("face color #7")),
                    col("bc", IgnoredElement("face color #8")),
                    mkr("bc", IgnoredElement("marker #0")),
                    txt("bc", IgnoredElement("marker label #0")),
                    mkr("bc", IgnoredElement("marker #1")),
                    txt("bc", IgnoredElement("marker label #1")),
                ]
            ),
        }
        box_right_bottom = {
            "type": "TextBoxAxes",
            "name": e1("right/bottom"),
            "elements": e1(
                [
                    txt("tc", f"<words.release{sl}>"),
                    txt("bl", f"<words.washout_exponent{sl}>:"),
                    txt("bl", f"<words.washout_coeff{sl}>:"),
                    txt("bl", f"<words.sedimentation_velocity{sl[:-1]}|abbr]>:"),
                    txt("bl", f"<words.deposition_velocity{sl[:-1]}|abbr]>:"),
                    txt("bl", f"<words.half_life{sl}>:"),
                    txt("bl", f"<words.substance{sl}>:"),
                    txt("bl", f"<words.total_mass{sl}>:"),
                    txt("bl", f"<words.rate{sl}>:"),
                    txt("bl", f"<words.end{sl}>:"),
                    txt("bl", f"<words.start{sl}>:"),
                    txt("bl", f"<words.height{sl}>:"),
                    txt("bl", f"<words.longitude{sl}>:"),
                    txt("bl", f"<words.latitude{sl}>:"),
                    txt("bl", f"<words.site{sl}>:"),
                    txt("br", f"<species.washout_exponent{sl}.format>"),
                    txt("br", f"<species.washout_coeff{sl}.format>"),
                    txt("br", f"<species.sediment_vel{sl}.format>"),
                    txt("br", f"<species.deposit_vel{sl}.format>"),
                    txt("br", f"<species.half_life{sl}.format>"),
                    txt("br", f"<species.name{sl}.format>"),
                    txt("br", f"<release.mass{sl}.format>"),
                    txt("br", f"<release.rate{sl}.format>"),
                    txt("br", f"<release.end{sl}.format>"),
                    txt("br", f"<release.start{sl}.format>"),
                    txt("br", f"<release.height{sl}.format>"),
                    txt("br", IgnoredElement("release longitude")),
                    txt("br", IgnoredElement("release latitude")),
                    txt("br", f"<release.site_name{sl}.format>"),
                ]
            ),
        }
        box_bottom = {
            "type": "TextBoxAxes",
            "name": e1("bottom"),
            "elements": e1(
                [
                    txt(
                        "tl",
                        f"<words.flexpart{sl}> <words.based_on{sl}> "
                        f"<simulation.model_name{sl}>, <simulation.start{sl}.format>",
                    ),
                    txt("tr", f"<symbols.copyright><words.meteoswiss{sl}>"),
                ]
            ),
        }
        jdat = {
            "method:Plot.fill_box_top_left": box_top_left,
            "method:Plot.fill_box_top_right": box_top_right,
            "method:Plot.fill_box_right_top": box_right_top,
            "method:Plot.fill_box_right_bottom": box_right_bottom,
            "method:Plot.fill_box_bottom": box_bottom,
        }

        return {"boxes": jdat}

    def fig(self):
        e = self.element

        jdat = {
            "type": "Figure",
            "bbox": {
                "type": "TransformedBbox",
                "bounds": e((0.0, 0.0, 1200.0, 900.0)),
            },
            "axes": [
                {"type": "GeoAxesSubplot", "bbox": IgnoredElement("axes[0]::bbox")},
                {"type": "Axes", "bbox": IgnoredElement("axes[1]::bbox")},
                {"type": "Axes", "bbox": IgnoredElement("axes[2]::bbox")},
                {"type": "Axes", "bbox": IgnoredElement("axes[3]::bbox")},
                {"type": "Axes", "bbox": IgnoredElement("axes[4]::bbox")},
                {"type": "Axes", "bbox": IgnoredElement("axes[5]::bbox")},
            ],
        }

        return {"fig": jdat}

    def field(self):
        e = self.element
        jdat = {
            "type": "Field",
            "fld_specs": None,  # SR_TMP
            "time_stats": {"max": e(15)},
            "fld": {
                "dtype": "float32",
                "shape": e((10, 10)),
                "nanmin": e(0.0),
                "nanmean": e(4.5),
                "nanmedian": e(4.5),
                "nanmax": e(9.0),
                "nanmin_nonzero": e(1.0),
                "nanmean_nonzero": e(5.0),
                "nanmedian_nonzero": e(5.0),
                "nanmax_nonzero": e(9.0),
                "n_nan": e(0),
                "n_zero": e(10),
            },
            "rlat": {
                "dtype": "float64",
                "shape": e((10,)),
                "min": e(-5.0),
                "max": e(4.0),
            },
            "rlon": {
                "dtype": "float64",
                "shape": e((10,)),
                "min": e(-6.0),
                "max": e(3.0),
            },
        }
        return {"field": jdat}
