#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot``."""
import logging as log
import numpy as np
import pytest

from types import SimpleNamespace

from pyflexplot.plot import DispersionPlot

from utils import assert_summary_dict_is_subdict, IgnoredElement


#----------------------------------------------------------------------
# Classes for dummy input data
#----------------------------------------------------------------------

class DummyWord:

    def __init__(self, name, parent):
        self._name = name
        self._lang = None
        self._parent = parent

    @property
    def lang(self):
        return self._parent.lang_

    def __str__(self):
        s = f'{self._parent._name}.{self._name}'
        if self.lang:
            s += f'[{self.lang}]'
        return f'<{s}>'

    def ctx(self, ctx):
        return f'<{str(self)[1:-1]}[{ctx}]>'.replace(r'][', '|')


class DummyWords:

    def __init__(self, name, words, lang=None):
        self._name = name
        for word in words:
            setattr(self, word, DummyWord(word, self))
        self.lang_ = None
        self.lang_ = lang

    def set_default_(self, lang):
        self.lang_ = lang


class DummyAttr:

    def __init__(self, name, value, lang):
        name = f'<{name}[{lang}]>'
        self._name = name
        self.value = value or name
        self.lang = lang
        self.unit = f'{name}.unit'

    def format(self, *args, **kwargs):
        return f'<{self._name[1:-1]}.format>'


#----------------------------------------------------------------------
# Prepare test data
#----------------------------------------------------------------------

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
    return SimpleNamespace(
        time_stats={'max': 15},
        fld=np.array([[i]*10 for i in range(10)], np.float32),
        rlat=np.arange(-5.0, 4.1, 1.0),
        rlon=np.arange(-6.0, 3.1, 1.0),
        attrs=attrs,
        summarize=lambda: {},
        scale=lambda f: None,
    )

def create_dummy_words():

    w = DummyWords(
        'words', [
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

    return DummyWords('symbols', [
        'ae',
        'copyright',
        'oe',
        't0',
        'ue',
    ])

def create_dummy_labels(lang):
    dummy_words = create_dummy_words()
    from pyflexplot.plot import DispersionPlotLabels
    return DispersionPlotLabels(lang, dummy_words)

#----------------------------------------------------------------------
# Create result
#----------------------------------------------------------------------

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

#----------------------------------------------------------------------
# Run tests
#----------------------------------------------------------------------

def _test_full(lang):
    res = create_res(lang)
    sol = create_sol(lang, base_attrs=True, dim_attrs=True, labels=True,
                     ax_map=True, boxes=True, fig=True)
    assert_summary_dict_is_subdict(
        superdict=res, subdict=sol, supername='result', subname='solution')

def test_full_en():
    _test_full('en')

def test_full_de():
    _test_full('de')

#----------------------------------------------------------------------
# Create solutions
#----------------------------------------------------------------------

def create_sol(lang, base_attrs=False, dim_attrs=False, labels=False,
               ax_map=False, boxes=False, fig=False):
    sol = {}
    if base_attrs:
        sol.update(create_sol_base_attrs(lang))
    if dim_attrs:
        sol.update(create_sol_dim_attrs())
    if labels:
        sol.update(create_sol_labels())
    if ax_map:
        sol.update(create_sol_ax_map())
    if boxes:
        sol.update(create_sol_boxes(lang))
    if fig:
        sol.update(create_sol_fig())
    return sol

def create_sol_base_attrs(lang):
    base_attrs = {
        'type': 'DispersionPlot',
        'lang': lang,
        'extend': 'max',
        'level_range_style': 'base',
        'draw_colors': True,
        'draw_contours': False,
        'mark_field_max': True,
        'mark_release_site': True,
    }
    return base_attrs

def create_sol_dim_attrs():
    dim_attrs = {
        'dpi': 100.0,
        'figsize': (12.0, 9.0),
        'text_box_setup': {
            'h_rel_t': 0.1,
            'h_rel_b': 0.03,
            'w_rel_r': 0.25,
            'pad_hor_rel': 0.015,
            'h_rel_box_rt': 0.45,
        },
    }
    return dim_attrs

def create_sol_labels():
    labels = {}
    return {'labels': labels}

def create_sol_ax_map():
    ax_map = {
        'type': 'AxesMap',
    }
    return {'ax_map': ax_map}

def create_sol_boxes(lang, check_text=True):
    # yapf: disable

    def txt(loc, s, **kwargs):
        """Shortcut for text box element."""
        return {
            'type': 'TextBoxElement_Text',
            'loc': {'loc': loc},
            's': s if check_text else SkipElement(s),
            **kwargs}

    def col(loc, fc, **kwargs):
        """Shortcut for color rect element."""
        return {
            'type': 'TextBoxElement_ColorRect',
            'loc': {'loc': loc},
            'fc': fc,
            **kwargs}

    def mkr(loc, m, **kwargs):
        """Shortcut for marker element."""
        return {
            'type': 'TextBoxElement_Marker',
            'loc': {'loc': loc},
            'm': m,
            **kwargs}

    boxes = [
        {
            'type': 'TextBoxAxes',
            'name': 'top',
            'elements': [
                txt('tl', f'<variable.long_name[{lang}]> '
                          f'<words.at[{lang}|level]> '
                          f'<variable.fmt_level_range[{lang}].format>') ,
                txt('bl', f'<words.averaged_over[{lang}]> '
                          f'<simulation.fmt_integr_period[{lang}].format> '
                          f'(<words.since[{lang}]> '
                          f'<simulation.integr_start[{lang}].format>)') ,
                txt('tc', f'<species.name[{lang}].format>'),
                txt('bc', f'<release.site_name[{lang}]>'),
                txt('tr', f'<simulation.now[{lang}].format>'),
                txt('br', f'<simulation.now[{lang}].format>'),
            ],
        },
        {
            'type': 'TextBoxAxes',
            'name': 'right/top',
            'elements': [
                txt('tc', (f'<variable.short_name[{lang}].format> '
                           f'(<variable.unit[{lang}].format>)')),
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
            ],
        },
        {
            'type': 'TextBoxAxes',
            'name': 'right/bottom',
            'elements': [
                txt('tc', f'<words.release[{lang}]>'),
                txt('bl', f'<words.washout_exponent[{lang}]>:'),
                txt('bl', f'<words.washout_coeff[{lang}]>:'),
                txt('bl', f'<words.sediment_vel[{lang}]>:'),
                txt('bl', f'<words.deposit_vel[{lang}]>:'),
                txt('bl', f'<words.half_life[{lang}]>:'),
                txt('bl', f'<words.substance[{lang}]>:'),
                txt('bl', f'<words.total_mass[{lang}]>:'),
                txt('bl', f'<words.rate[{lang}]>:'),
                txt('bl', f'<words.end[{lang}]>:'),
                txt('bl', f'<words.start[{lang}]> (<symbols.t0>):'),
                txt('bl', f'<words.height[{lang}]>:'),
                txt('bl', f'<words.longitude[{lang}]>:'),
                txt('bl', f'<words.latitude[{lang}]>:'),
                txt('bl', f'<words.site[{lang}]>:'),
                txt('br', f'<species.washout_exponent[{lang}].format>'),
                txt('br', f'<species.washout_coeff[{lang}].format>'),
                txt('br', f'<species.sediment_vel[{lang}].format>'),
                txt('br', f'<species.deposit_vel[{lang}].format>'),
                txt('br', f'<species.half_life[{lang}].format>'),
                txt('br', f'<species.name[{lang}].format>'),
                txt('br', f'<release.mass[{lang}].format>'),
                txt('br', f'<release.rate[{lang}].format>'),
                txt('br', f'<simulation.end[{lang}].format>'),
                txt('br', f'<simulation.start[{lang}].format>'),
                txt('br', f'<release.height[{lang}].format>'),
                txt('br', IgnoredElement('release longitude')),
                txt('br', IgnoredElement('release latitude')),
                txt('br', f'<release.site_name[{lang}].format>'),
            ],
        },
        {
            'type': 'TextBoxAxes',
            'name': 'bottom',
            'elements': [
                txt('tl', f'<words.flexpart[{lang}]> '
                          f'<words.based_on[{lang}]> '
                          f'<simulation.model_name[{lang}]>, '
                          f'<simulation.start[{lang}].format>'),
                txt('tr', f'<symbols.copyright><words.mch[{lang}]>'),
            ],
        },
    ]
    # yapf: enable
    return {'boxes': boxes}

def create_sol_fig():
    # yapf: disable
    fig = {
        'type': 'Figure',
        'dpi': 100.0,
        'bbox': {
            'type': 'TransformedBbox',
            'bounds': (0.0, 0.0, 1200.0, 900.0),
        },
        'axes': [
            {
                'type': 'GeoAxesSubplot'
            },
            {
                'type': 'Axes'
            },
            {
                'type': 'Axes'
            },
            {
                'type': 'Axes'
            },
            {
                'type': 'Axes'
            },
        ]
    }
    # yapf: enable
    return {'fig': fig}
