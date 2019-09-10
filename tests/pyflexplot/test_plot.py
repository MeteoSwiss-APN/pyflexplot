#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot``."""
import logging as log
import numpy as np
import pytest
from types import SimpleNamespace

from pyflexplot.plot import Plot_Dispersion

from pyflexplot.utils_dev import ipython  #SR_DEV


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
                site_name=attr('release.site_name', 'Goesgen'),
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

    def test_FOO(self):

        field = self.create_dummy_field()
        plot = Plot_Dispersion(field)
        res = plot.summarize()

        #ipython(globals(), locals())
