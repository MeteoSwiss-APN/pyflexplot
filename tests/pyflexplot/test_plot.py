#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot``."""
import logging as log
import numpy as np
import pytest
from types import SimpleNamespace

from pyflexplot.plot import DispersionPlot

from pyflexplot.utils_dev import ipython  #SR_DEV

from utils import assert_summary_dict_is_subdict


class Test_PlotDispersion_Summarize:

    def create_dummy_attrs(self):

        def f_fmt(name):

            def fmt(*args, **kwargs):
                return f'{name}'

            return fmt

        def attr(name, value=None):
            return SimpleNamespace(
                value=name if value is None else value,
                unit=f'{name}.unit',
                format=f_fmt(f'{name}.format'),
            )

        return SimpleNamespace(
            summarize=lambda: {},
            grid=SimpleNamespace(
                north_pole_lat=attr('grid.north_pole_lat', 43.0),
                north_pole_lon=attr('grid.north_pole_lat', -170.0),
            ),
            release=SimpleNamespace(
                lon=attr('release.lon', 8.0),
                lat=attr('release.lat', 47.0),
                site_name=attr('release.site_name'),
                height=attr('release.height'),
                rate=attr('release.rate'),
                mass=attr('release.mass'),
            ),
            variable=SimpleNamespace(
                long_name=attr('variable.long_name'),
                format_level_range=f_fmt('variable.format_level_range'),
                short_name=attr('variable.short_name'),
                unit=attr('variable.unit'),
            ),
            species=SimpleNamespace(
                name=attr('species.name'),
                half_life=attr('species.half_life'),
                deposit_vel=attr('species.deposit_vel'),
                sediment_vel=attr('species.sediment_vel'),
                washout_coeff=attr('species.washout_coeff'),
                washout_exponent=attr('species.washout_coeff'),
            ),
            simulation=SimpleNamespace(
                now=attr('simulation.now'),
                format_integr_period=f_fmt('simulation.format_integr_period'),
                integr_start=attr('simulation.integr_start'),
                start=attr('simulation.start'),
                end=attr('simulation.end'),
                model_name=attr('simulation.model_name')),
        )

    def create_dummy_field(self):
        return SimpleNamespace(
            time_stats={'max': 15},
            fld=np.array([[i]*10 for i in range(10)], np.float32),
            rlat=np.arange(-5.0, 4.1, 1.0),
            rlon=np.arange(-6.0, 3.1, 1.0),
            attrs=self.create_dummy_attrs(),
            summarize=lambda: {},
        )

    #------------------------------------------------------------------

    def test(self):

        field = self.create_dummy_field()
        plot = DispersionPlot(field, dpi=100, figsize=(12, 9), lang='de')
        res = plot.summarize()

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
                'h_rel_box_rt': 0.46,
            },
            'labels': {},
            'ax_map': {
                'type': 'AxesMap',
            },
            'boxes': [
                {
                    'type':
                        'TextBoxAxes',
                    'elements': [
                        {
                            'type': 'TextBoxElement_Text',
                            's': 'variable.long_name',
                            'loc': {
                                'loc': 'tl'
                            },
                        },
                        {
                            'type': 'TextBoxElement_Text',
                            's': 'species.name.format',
                            'loc': {
                                'loc': 'tc'
                            },
                        },
                        {
                            'type': 'TextBoxElement_Text',
                            's': 'simulation.now.format',
                            'loc': {
                                'loc': 'tr'
                            },
                        },
                        {
                            'type': 'TextBoxElement_Text',
                            #'s': '...',
                            'loc': {
                                'loc': 'bl'
                            },
                        },
                        {
                            'type': 'TextBoxElement_Text',
                            's': 'release.site_name',
                            'loc': {
                                'loc': 'bc'
                            },
                        },
                        {
                            'type': 'TextBoxElement_Text',
                            's': 'simulation.now.format',
                            'loc': {
                                'loc': 'br'
                            },
                        },
                    ],
                },
                {
                    'type': 'TextBoxAxes'
                },
                {
                    'type': 'TextBoxAxes'
                },
                {
                    'type': 'TextBoxAxes'
                },
            ],
            'fig': {
                'type':
                    'Figure',
                'dpi':
                    100.0,
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
            },
        }

        assert_summary_dict_is_subdict(superdict=res, subdict=sol)
