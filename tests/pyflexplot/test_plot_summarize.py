#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test summarizing plots as defined in ``pyflexplot.plot``.

Plots can be summarized into dicts, which can be used to test that all
plot elements are in the right place with the right content etc.

(Summarization is similar to serialization, but without the ability to
reconstruct the objects from the summary.)

In this module, the summary dict of one plot type is thoroughly tested
to ensure that summarization works as expected. In order to isolate
this process as best as possible, several classes used to define labels
etc. are mocked: ``Word``, ``Words``, ``Attr``.

While the mocks allows us to inject dummy strings as labels etc. and
makes the tests independent of the exact formation of those, the tests
are still dependent on the structure of ``DispersionPlot``, i.e., what
elements there are and where they go.

Not every failure thus implies that summarization is broken! There
might just have been minor adjustments to the plot layout.

The tests are organized in such a way that a comprehensive reference
summary dict is constructed near the bottom, and each test compares a
subset of it against the respective part(s) of the summary dict
obtained from the tested plot class.

All tests come in pairs: For each positive test (check that elements
present in both the result and solution summary dicts match) there is
a negative test that ensures failure in case of wrong content. This
prevents false positive, i.e., passing tests because of missing
elements in either dict.

Some dict elements are explicitly ignored in comparisons by aid of
``IgnoredElement``. This allows us to test, for instance that the
correct number of lebend labels are present without testing for the
labels themselves, or to avoid hard-coding lat/lon coordinates in the
solution.
"""
import functools
import logging as log
import numpy as np
import pytest

from types import SimpleNamespace

from srutils.testing import assert_summary_dict_is_subdict
from srutils.testing import IgnoredElement
from srutils.testing import UnequalElement
from srutils.various import isiterable
from words import Word, Words

from pyflexplot.data import Field
from pyflexplot.plot import DispersionPlot

#======================================================================
# Classes for dummy input data
#======================================================================


class DummyWord(Word):
    """Wrapper for ``Word`` class for testing."""

    def __str__(self):
        s = f'{self._parent.name}.{self.name}'
        if self._lang:
            s += f'[{self._lang}]'
        return f'<{s}>'

    @property
    def _lang(self):
        return self._parent._lang

    def ctx(self, name):
        return f'<{str(self)[1:-1]}[{name}]>'.replace(r'][', '|')


class DummyWords(Words):
    """Wrapper for ``Words`` class for testing."""

    cls_word = DummyWord

    @classmethod
    def create(cls, name, lang, words):
        self = cls(
            name=name,
            default_lang=lang,
            **{w: {
                'en': w,
                'de': w
            } for w in words})
        self._lang = lang
        for word in self._words.values():
            word._parent = self
        return self


class DummyAttr:
    """Replacement for ``Attr`` class for testing."""

    def __init__(self, name, value, lang):
        name = f'<{name}[{lang}]>'
        self._name = name
        self.value = value or name
        self.lang = lang
        self.unit = f'{name}.unit'

    def format(self, *args, **kwargs):
        return f'<{self._name[1:-1]}.format>'


#======================================================================
# Prepare test data
#======================================================================


def create_dummy_attrs(lang):

    # Note: Some values must be passed, otherwise plotting fails

    DA = lambda name, value=None: DummyAttr(name, value, lang=lang)
    fm = lambda name, value=None: DA(name, value).format

    # yapf: disable

    return SimpleNamespace(
        summarize=lambda: {},

        grid=SimpleNamespace(
            north_pole_lat      = DA('grid.north_pole_lat', 43.0),
            north_pole_lon      = DA('grid.north_pole_lat', -170),
        ),
        release=SimpleNamespace(
            lon                 = DA('release.lon', 8.0),
            lat                 = DA('release.lat', 47.0),
            site_name           = DA('release.site_name'),
            height              = DA('release.height'),
            rate                = DA('release.rate'),
            mass                = DA('release.mass'),
        ),
        variable=SimpleNamespace(
            long_name           = DA('variable.long_name'),
            fmt_level_range     = fm('variable.fmt_level_range'),
            short_name          = DA('variable.short_name'),
            unit                = DA('variable.unit'),
        ),
        species=SimpleNamespace(
            name                = DA('species.name'),
            half_life           = DA('species.half_life'),
            deposit_vel         = DA('species.deposit_vel'),
            sediment_vel        = DA('species.sediment_vel'),
            washout_coeff       = DA('species.washout_coeff'),
            washout_exponent    = DA('species.washout_exponent'),
        ),
        simulation=SimpleNamespace(
            now                 = DA('simulation.now'),
            fmt_integr_period   = fm('simulation.fmt_integr_period'),
            integr_start        = DA('simulation.integr_start'),
            integr_type         = DA('simulation.integr_type', 'mean'),
            start               = DA('simulation.start'),
            end                 = DA('simulation.end'),
            model_name          = DA('simulation.model_name'),
        ),
    )
    # yapf: enable


def create_dummy_field(attrs):
    dummy_field = Field(
        fld=np.array([[i]*10 for i in range(10)], np.float32),
        rlat=np.arange(-5.0, 4.1, 1.0),
        rlon=np.arange(-6.0, 3.1, 1.0),
        attrs=attrs,
        field_specs=None,
        time_stats={'max': 15},
    )
    return dummy_field


def create_dummy_words(lang):

    w = DummyWords.create(
        'words', lang, [
            'accumulated_over',
            'at',
            'averaged_over',
            'based_on',
            'deposit_vel',
            'end',
            'flexpart',
            'half_life',
            'height',
            'latitude',
            'longitude',
            'max',
            'mch',
            'rate',
            'release',
            'release_site',
            'sediment_vel',
            'since',
            'site',
            'start',
            'substance',
            'summed_up_over',
            'total_mass',
            'washout_coeff',
            'washout_exponent',
        ])
    w.symbols = create_dummy_symbols()
    return w


def create_dummy_symbols():

    return DummyWords.create(
        'symbols', None, [
            'ae',
            'copyright',
            'oe',
            't0',
            'ue',
        ])


def create_dummy_labels(lang):
    dummy_words = create_dummy_words(lang)
    from pyflexplot.plot import DispersionPlotLabels
    return DispersionPlotLabels(lang, dummy_words)


#======================================================================
# Create result
#======================================================================


def create_res(lang, _cache={}):
    """Create test result once, then return it from cache.

    Based on mostly dummy input data (defined above), a dispersion plot
    object is created and summarized into a dict. The same result dict
    is used in all test, but compared against solution dicts of varying
    detail. Each solution dict constitutes a subset of the results dict.

    A test passes if all elements in the solution dict are present and
    equal in the results dict, while all other elements in the results
    dict are ignored.

    """
    if lang not in _cache:
        attrs = create_dummy_attrs(lang)
        field = create_dummy_field(attrs)
        labels = create_dummy_labels(lang)
        plot = DispersionPlot(
            field, dpi=100, figsize=(12, 9), lang=lang, labels=labels)
        _cache[lang] = plot.summarize()
    return _cache[lang]


#======================================================================
# Run tests
#======================================================================


class CreateTests:
    """Decorator to test a given results set in multiple languages.

    Each test is run for all combinations of ``langs`` (languages) and
    ``invs`` (whether to run an inverse test; see below).

    By default (langs=['en', 'de], invs=[False, True]), four tests are
    created for the given argument setup:
        - a regular test (inv=False) in English (lang='en'),
        - an inverted test (inv=True) in English (lang='en'),
        - a regular test (inv=False) in German (lang='de'), and
        - an inverted test (inv=True) in German (lang='de').

    In an inverse test, the values of the tested attributes are
    replaced by instances of ``UnequalElement``, which always return
    False in comparisons. This means that for the test to pass, the
    result must differ from the solution. This ensures that the regular
    tests pass because of successful comparisons.

    """

    langs = ['en', 'de']
    invs = [False, True]
    args = [
        'base_attrs', 'dim_attrs', 'labels', 'ax_map', 'boxes', 'boxes_text',
        'fig', 'field'
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
                    outer_self.run_test(
                        inner_self, lang=lang, inverted=inverted)

                setattr(cls, f"test_{lang}{'_inv' if inv else ''}", f)
        return cls

    def run_test(outer_self, inner_self, lang, inverted):
        res = create_res(lang)
        kwargs = {s: getattr(inner_self, s, False) for s in outer_self.args}
        sol = create_sol(lang, inverted=inverted, **kwargs)
        try:
            assert_summary_dict_is_subdict(
                superdict=res,
                subdict=sol,
                supername='result',
                subname='solution')
        except AssertionError:
            if inverted:
                return
            raise
        else:
            if inverted:
                raise AssertionError(f"inverted test passed")


#======================================================================


@CreateTests()
class Test_BaseAttrs:
    base_attrs = True


@CreateTests()
class Test_DimAttrs:
    dim_attrs = True


@CreateTests()
class Test_Labels:
    labels = True


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
    dim_attrs = True
    labels = True
    ax_map = True
    boxes = True
    boxes_text = True
    fig = True
    field = True


#======================================================================
# Create solutions
#======================================================================


def create_sol(lang, inverted, **kwargs):
    return Solution(lang, inverted).create(**kwargs)


class Solution:

    def __init__(self, lang, inverted):
        self.lang = lang
        self.inverted = inverted

    def create(
            self, *, base_attrs, dim_attrs, labels, ax_map, boxes, boxes_text,
            fig, field):
        sol = {}
        if base_attrs:
            sol.update(self.base_attrs())
        if dim_attrs:
            sol.update(self.dim_attrs())
        if labels:
            sol.update(self.labels())
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
            'type': e('DispersionPlot'),
            'lang': e(self.lang),
            'extend': e('max'),
            'level_range_style': e('base'),
            'draw_colors': e(True),
            'draw_contours': e(False),
            'mark_field_max': e(True),
            'mark_release_site': e(True),
        }
        return jdat

    def dim_attrs(self):
        e = self.element
        jdat = {
            'dpi': e(100.0),
            'figsize': e((12.0, 9.0)),
            'text_box_setup': {
                'h_rel_t': e(0.1),
                'h_rel_b': e(0.03),
                'w_rel_r': e(0.25),
                'pad_hor_rel': e(0.015),
                'h_rel_box_rt': e(0.45),
            },
        }
        return jdat

    def labels(self):
        e = self.element
        jdat = e({})  #SR_TMP
        return {'labels': jdat}

    def ax_map(self):
        e = self.element
        jdat = {
            'type': e('AxesMap'),
        }
        return {'ax_map': jdat}

    def boxes(self, check_text=True):
        # yapf: disable

        if not check_text:
            # Outer elements set up to fail in inverted test
            # Inner elements (text strings etc.) always ignored
            e1 = self.element
            e2 = IgnoredElement
        else:
            # Outer elements NOT set up to fail in inverted test
            # Inner elements set up to fail in inverted test
            e1 = lambda x: x
            e2 = self.element

        def txt(loc, s, **kwargs):
            """Shortcut for text box element."""
            return {
                'type': 'TextBoxElement_Text',
                'loc': {'loc': loc},
                's': e2(s),
                **kwargs}

        def col(loc, fc, **kwargs):
            """Shortcut for color rect element."""
            return {
                'type': 'TextBoxElement_ColorRect',
                'loc': {'loc': loc},
                'fc': e2(fc),
                **kwargs}

        def mkr(loc, m, **kwargs):
            """Shortcut for marker element."""
            return {
                'type': 'TextBoxElement_Marker',
                'loc': {'loc': loc},
                'm': e2(m),
                **kwargs}

        sl = f'[{self.lang}]'

        jdat = [
            {
                'type': 'TextBoxAxes',
                'name': e1('top'),
                'elements': e1([
                    txt('tl', f'<variable.long_name{sl}> '
                              f'<words.at[{self.lang}|level]> '
                              f'<variable.fmt_level_range{sl}.format>') ,
                    txt('bl', f'<words.averaged_over{sl}> '
                              f'<simulation.fmt_integr_period{sl}.format> '
                              f'(<words.since{sl}> '
                              f'<simulation.integr_start{sl}.format>)') ,
                    txt('tc', f'<species.name{sl}.format>'),
                    txt('bc', f'<words.release_site{sl}>: '
                              f'<release.site_name{sl}>'),
                    txt('tr', f'<simulation.now{sl}.format>'),
                    txt('br', f'<simulation.now{sl}.format>'),
                ]),
            },
            {
                'type': 'TextBoxAxes',
                'name': e1('right/top'),
                'elements': e1([
                    txt('tc', (f'<variable.short_name{sl}.format> '
                               f'(<variable.unit{sl}.format>)')),
                    txt('bc', IgnoredElement('level range #0')),
                    txt('bc', IgnoredElement('level range #1')),
                    txt('bc', IgnoredElement('level range #2')),
                    txt('bc', IgnoredElement('level range #3')),
                    txt('bc', IgnoredElement('level range #4')),
                    txt('bc', IgnoredElement('level range #5')),
                    txt('bc', IgnoredElement('level range #6')),
                    txt('bc', IgnoredElement('level range #7')),
                    txt('bc', IgnoredElement('level range #8')),
                    col('bc', IgnoredElement('face color #0')),
                    col('bc', IgnoredElement('face color #1')),
                    col('bc', IgnoredElement('face color #2')),
                    col('bc', IgnoredElement('face color #3')),
                    col('bc', IgnoredElement('face color #4')),
                    col('bc', IgnoredElement('face color #5')),
                    col('bc', IgnoredElement('face color #6')),
                    col('bc', IgnoredElement('face color #7')),
                    col('bc', IgnoredElement('face color #8')),
                    mkr('bc', IgnoredElement('marker #0')),
                    txt('bc', IgnoredElement('marker label #0')),
                    mkr('bc', IgnoredElement('marker #1')),
                    txt('bc', IgnoredElement('marker label #1')),
                ]),
            },
            {
                'type': 'TextBoxAxes',
                'name': e1('right/bottom'),
                'elements': e1([
                    txt('tc', f'<words.release{sl}>'),
                    txt('bl', f'<words.washout_exponent{sl}>:'),
                    txt('bl', f'<words.washout_coeff{sl}>:'),
                    txt('bl', f'<words.sediment_vel{sl}>:'),
                    txt('bl', f'<words.deposit_vel{sl}>:'),
                    txt('bl', f'<words.half_life{sl}>:'),
                    txt('bl', f'<words.substance{sl}>:'),
                    txt('bl', f'<words.total_mass{sl}>:'),
                    txt('bl', f'<words.rate{sl}>:'),
                    txt('bl', f'<words.end{sl}>:'),
                    txt('bl', f'<words.start{sl}> (<symbols.t0>):'),
                    txt('bl', f'<words.height{sl}>:'),
                    txt('bl', f'<words.longitude{sl}>:'),
                    txt('bl', f'<words.latitude{sl}>:'),
                    txt('bl', f'<words.site{sl}>:'),
                    txt('br', f'<species.washout_exponent{sl}.format>'),
                    txt('br', f'<species.washout_coeff{sl}.format>'),
                    txt('br', f'<species.sediment_vel{sl}.format>'),
                    txt('br', f'<species.deposit_vel{sl}.format>'),
                    txt('br', f'<species.half_life{sl}.format>'),
                    txt('br', f'<species.name{sl}.format>'),
                    txt('br', f'<release.mass{sl}.format>'),
                    txt('br', f'<release.rate{sl}.format>'),
                    txt('br', f'<simulation.end{sl}.format>'),
                    txt('br', f'<simulation.start{sl}.format>'),
                    txt('br', f'<release.height{sl}.format>'),
                    txt('br', IgnoredElement('release longitude')),
                    txt('br', IgnoredElement('release latitude')),
                    txt('br', f'<release.site_name{sl}.format>'),
                ]),
            },
            {
                'type': 'TextBoxAxes',
                'name': e1('bottom'),
                'elements': e1([
                    txt('tl', f'<words.flexpart{sl}> '
                              f'<words.based_on{sl}> '
                              f'<simulation.model_name{sl}>, '
                              f'<simulation.start{sl}.format>'),
                    txt('tr', f'<symbols.copyright><words.mch{sl}>'),
                ]),
            },
        ]
        # yapf: enable
        return {'boxes': jdat}

    def fig(self):
        e = self.element
        # yapf: disable
        jdat = {
            'type': 'Figure',
            'dpi': e(100.0),
            'bbox': {
                'type': 'TransformedBbox',
                'bounds': e((0.0, 0.0, 1200.0, 900.0)),
            },
            'axes': [
                {
                    'type': 'GeoAxesSubplot',
                    'bbox': IgnoredElement('axes[0]::bbox'),
                },
                {
                    'type': 'Axes',
                    'bbox': IgnoredElement('axes[1]::bbox'),
                },
                {
                    'type': 'Axes',
                    'bbox': IgnoredElement('axes[2]::bbox'),
                },
                {
                    'type': 'Axes',
                    'bbox': IgnoredElement('axes[3]::bbox'),
                },
                {
                    'type': 'Axes',
                    'bbox': IgnoredElement('axes[4]::bbox'),
                },
            ]
        }
        # yapf: enable
        return {'fig': jdat}

    def field(self):
        e = self.element
        jdat = {
            'type': 'Field',
            'attrs': {},  #SR_TMP
            'field_specs': None,  #SR_TMP
            'time_stats': {
                'max': e(15),
            },
            'fld': {
                'dtype': 'float32',
                'shape': e((10, 10)),
                'nanmin': e(0.0),
                'nanmean': e(4.5),
                'nanmedian': e(4.5),
                'nanmax': e(9.0),
                'nanmin_nonzero': e(1.0),
                'nanmean_nonzero': e(5.0),
                'nanmedian_nonzero': e(5.0),
                'nanmax_nonzero': e(9.0),
                'n_nan': e(0),
                'n_zero': e(10),
            },
            'rlat': {
                'dtype': 'float64',
                'shape': e((10,)),
                'min': e(-5.0),
                'max': e(4.0),
            },
            'rlon': {
                'dtype': 'float64',
                'shape': e((10,)),
                'min': e(-6.0),
                'max': e(3.0),
            },
        }
        return {'field': jdat}
