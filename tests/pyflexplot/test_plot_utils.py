#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot_utils``."""
import logging as log
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt

from pprint import pformat

from srutils.testing import assert_summary_dict_is_subdict
from srutils.various import isiterable

from pyflexplot.plot_utils import TextBoxAxes


class Test_TextBoxAxes_Summarize:
    """Summarize all relevant information about an ``TextBoxAxes``."""

    def create_text_box(self, **kwargs):
        self.fig, self.ax_ref = plt.subplots(figsize=(1, 1), dpi=100)
        self.rect_lbwh = [1, 1, 3, 2]
        self.box = TextBoxAxes(self.fig, self.ax_ref, self.rect_lbwh, **kwargs)
        return self.box

    kwargs_summarize = {"add": ["fig"]}

    sol_base = {
        "type": "TextBoxAxes",
        "rect": [1.0, 1.0, 3.0, 2.0],
        "fig": {
            "type": "Figure",
            "dpi": 100,
            "bbox": {"type": "TransformedBbox", "bounds": (0.0, 0.0, 100.0, 100.0),},
            "axes": [
                {"type": "AxesSubplot"},
                {
                    "type": "Axes",
                    "bbox": {
                        "type": "TransformedBbox",
                        "bounds": (100.0, 100.0, 300.0, 200.0),
                    },
                },
            ],
        },
    }

    def test_text_line(self):
        box = self.create_text_box(show_border=False)
        box.text("bl", "lower-left", dx=1.6, dy=0.8)
        res = box.summarize(**self.kwargs_summarize)
        sol = {
            **self.sol_base,
            "show_border": False,
            "elements": [
                {
                    "type": "TextBoxElement_Text",
                    "s": "lower-left",
                    "loc": {
                        "type": "TextBoxLocation",
                        "va": "baseline",
                        "ha": "left",
                        "dx": 1.6,
                        "dy": 0.8,
                    },
                }
            ],
        }
        assert_summary_dict_is_subdict(superdict=res, subdict=sol)

    def test_text_block(self):
        box = self.create_text_box()
        box.text_block("mc", [("foo", "bar"), ("hello", "world")])
        res = box.summarize(**self.kwargs_summarize)
        sol = {
            **self.sol_base,
            "show_border": True,
            "elements": [
                {"type": "TextBoxElement_Text", "s": ("foo", "bar"),},
                {"type": "TextBoxElement_Text", "s": ("hello", "world"),},
            ],
        }
        assert_summary_dict_is_subdict(superdict=res, subdict=sol)

    def test_color_rect(self):
        box = self.create_text_box(show_border=True)
        box.color_rect("tr", "red", "black")
        res = box.summarize(**self.kwargs_summarize)
        sol = {
            **self.sol_base,
            "show_border": True,
            "elements": [
                {"type": "TextBoxElement_ColorRect", "fc": "red", "ec": "black",}
            ],
        }
        assert_summary_dict_is_subdict(superdict=res, subdict=sol)
