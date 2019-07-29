# -*- coding: utf-8 -*-
"""
Data structures.
"""
import datetime
import logging as log

from .utils_dev import ipython  #SR_DEV


class FlexFieldRotPole:
    """FLEXPART data on rotated-pole grid."""

    def __init__(self, rlat, rlon, fld, attrs, field_specs, time_stats):
        """Create an instance of ``FlexFieldRotPole``.

        Args:
            rlat (ndarray[float]): Rotated latitude array (1D).

            rlon (ndarray[float]): Rotated longitude array (1D).

            fld (ndarray[float, float]): Field array (2D).

            attrs (FlexAttrGroupCollection): Attributes collection.

            field_specs (FlexFieldSpecs): Input field specifications.

            time_stats (dict): Some statistics across all time steps.

        """
        self.rlat = rlat
        self.rlon = rlon
        self.fld = fld
        self.attrs = attrs
        self.field_specs = field_specs
        self.time_stats = time_stats
