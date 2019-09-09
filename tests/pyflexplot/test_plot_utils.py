#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot_utils``."""
import logging as log
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt

from pyflexplot.plot_utils import AxesTextBox

from pyflexplot.utils_dev import ipython  #SR_DEV


class TestAxesTextBox_Serialize:
    """Serialize all relevant information about an ``AxesTextBox``."""

    def test_TODO(self):

        fig, ax_ref = plt.subplots()
        rect_lbwh = [1, 1, 3, 2]
        box = AxesTextBox(fig, ax_ref, rect_lbwh)
        box.text('bl', 'lower-left')

        res = box.summarize()

        sol = {}

        assert res['type'] == 'AxesTextBox'
