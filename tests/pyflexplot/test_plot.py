#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot``."""
import logging as log
import numpy as np
import pytest

from types import SimpleNamespace

from pyflexplot.plot import DispersionPlot

from utils import assert_summary_dict_is_subdict, IgnoredElement

AE = r'$\mathrm{\"a}$'
OE = r'$\mathrm{\"o}$'
UE = r'$\mathrm{\"u}$'


class DummyWord:

    def __init__(self, name, parent):
        self._name = name
        self._lang = None
        self._parent = parent

    @property
    def lang(self):
        return self._parent.lang_

    def __str__(self):
        return f"words.{self._name}[{self.lang or ''}]"

    def ctx(self, ctx):
        return f"{str(self)[:-1]}/{ctx}]"


class DummyWords:

    def __init__(self, words, lang='en'):
        for word in words:
            setattr(self, word, DummyWord(word, self))
        self.lang_ = None
        self.lang_ = lang

    def set_default_(self, lang):
        self.lang_ = lang


class DummyAttr:

    def __init__(self, name, value, lang):
        name = f'{name}[{lang}]'
        self._name = name
        self.value = value or name
        self.lang = lang
        self.unit = f'{name}.unit'

    def format(self, *args, **kwargs):
        return f'{self._name}.format'


class Test_PlotDispersion_Summarize:

    def create_dummy_attrs(self, lang):

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
                washout_exponent    = DA('species.washout_coeff'),
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

    def create_dummy_field(self, attrs):
        return SimpleNamespace(
            time_stats={'max': 15},
            fld=np.array([[i]*10 for i in range(10)], np.float32),
            rlat=np.arange(-5.0, 4.1, 1.0),
            rlon=np.arange(-6.0, 3.1, 1.0),
            attrs=attrs,
            summarize=lambda: {},
            scale=lambda f: None,
        )

    def create_dummy_words(self):

        return DummyWords([
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

    def create_dummy_labels(self, lang):
        dummy_words = self.create_dummy_words()
        from pyflexplot.plot import DispersionPlotLabels  #SR_TMP
        return DispersionPlotLabels(lang, dummy_words)  #SR_TMP

    #------------------------------------------------------------------

    def test(self):
        lang = 'de'

        attrs = self.create_dummy_attrs(lang)
        field = self.create_dummy_field(attrs)
        labels = self.create_dummy_labels(lang)
        plot = DispersionPlot(
            field, dpi=100, figsize=(12, 9), lang=lang, labels=labels)
        res = plot.summarize()

        txt = lambda l, s, **kwargs: dict(
            type='TextBoxElement_Text', loc={'loc': l}, s=s, **kwargs)
        col = lambda l, fc, **kwargs: dict(
            type='TextBoxElement_ColorRect', loc={'loc': l}, fc=fc, **kwargs)
        mkr = lambda l, m, **kwargs: dict(
            type='TextBoxElement_Marker', loc={'loc': l}, m=m, **kwargs)

        # yapf: disable
        sol_boxes = [
            # Top box
            {'type': 'TextBoxAxes', 'elements': [
                txt('tl',(f'variable.long_name[{lang}] '
                          f'words.at[{lang}/level] '
                          f'variable.fmt_level_range[{lang}].format')),
                txt('bl',(f'Words.averaged_over[{lang}] '  #SR_TMP< TODO proper capitalization
                          f'simulation.fmt_integr_period[{lang}].format '
                          f'(words.since[{lang}] '
                          f'simulation.integr_start[{lang}].format)')),
                txt('tc', f'species.name[{lang}].format'),
                txt('bc', f'release.site_name[{lang}]'),
                txt('tr', f'simulation.now[{lang}].format'),
                txt('br', f'simulation.now[{lang}].format'),
            ]},
            # Right-top box
            {'type': 'TextBoxAxes', 'elements': [
                txt('tc', (f'variable.short_name[{lang}].format '
                           f'(variable.unit[{lang}].format)')),
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
            ]},
            # Right-bottom box
            {'type': 'TextBoxAxes'},
            # Bottom box
            {'type': 'TextBoxAxes'},
        ]
        sol = {
            'type': 'DispersionPlot',
            'dpi': 100.0,
            'figsize': (12.0, 9.0),
            'lang': 'de',
            'extend': 'max',
            'level_range_style': 'base',
            'draw_colors': True,
            'draw_contours': False,
            'mark_field_max': True,
            'mark_release_site': True,
            'text_box_setup': {
                'h_rel_t': 0.1,
                'h_rel_b': 0.03,
                'w_rel_r': 0.25,
                'pad_hor_rel': 0.015,
                'h_rel_box_rt': 0.45,
            },
            'labels': {},
            'ax_map': {
                'type': 'AxesMap',
            },
            'boxes': sol_boxes,
            'fig': {
                'type': 'Figure',
                'dpi': 100.0,
                'bbox': {
                    'type': 'TransformedBbox',
                    'bounds': (0.0, 0.0, 1200.0, 900.0),
                },
                'axes': [
                    {'type': 'GeoAxesSubplot'},
                    {'type': 'Axes'},
                    {'type': 'Axes'},
                    {'type': 'Axes'},
                    {'type': 'Axes'},
                ]
            },
        }
        # yapf: enable

        assert_summary_dict_is_subdict(
            superdict=res, subdict=sol, supername='result', subname='solution')
