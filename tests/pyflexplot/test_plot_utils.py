#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.plot_utils``."""
import logging as log
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt

from pprint import pformat

from pyflexplot.plot_utils import TextBoxAxes
from pyflexplot.utils import isiterable

from pyflexplot.utils_dev import ipython  #SR_DEV

#======================================================================


def assert_summary_dict_is_subdict(subdict, superdict):
    """Check that one summary dict is a subdict of another."""
    for key, val_sub in subdict.items():
        val_super = get_dict_element(
            superdict, key, 'superdict', AssertionError)
        assert_summary_dict_element_is_subelement(val_sub, val_super)


def assert_summary_dict_element_is_subelement(obj_sub, obj_super):

    if obj_sub == obj_super:
        return

    elif isinstance(obj_sub, dict):
        assert_summary_dict_is_subdict(obj_sub, obj_super)

    elif isiterable(obj_sub):

        if not isiterable(obj_super):
            raise AssertionError(
                f"superdict element not iterable:\n\n{pformat(obj_super)}")

        if len(obj_sub) != len(obj_super):
            raise AssertionError(
                f"iterable elements differ in size: {len(obj_sub)} != "
                f"{len(obj_super)}\n\nsuper:\n{obj_super}\n\nsub:\n{obj_sub}")

        for subobj_sub, subobj_super in zip(obj_sub, obj_super):
            assert_summary_dict_element_is_subelement(subobj_sub, subobj_super)

    else:
        raise AssertionError(f"elements differ: {obj_sub} != {obj_super}")


def get_dict_element(dict_, key, name='dict', exception_type=ValueError):
    """Get an element from a dict, raising an exception otherwise."""
    try:
        return dict_[key]
    except KeyError:
        err = f"key missing in {name}: {key}"
    except TypeError:
        err = f"{name} has wrong type: {type(dict_)}"
    raise exception_type(f"{err}\n\n{name}:\n{pformat(dict_)}")


#======================================================================


class Test_TextBoxAxes_Summarize:
    """Summarize all relevant information about an ``TextBoxAxes``."""

    def create_text_box(self, **kwargs):
        self.fig, self.ax_ref = plt.subplots(figsize=(1, 1), dpi=100)
        self.rect_lbwh = [1, 1, 3, 2]
        self.box = TextBoxAxes(self.fig, self.ax_ref, self.rect_lbwh, **kwargs)
        return self.box

    kwargs_summarize = {'include_fig': True}

    sol_base = {
        'type': 'TextBoxAxes',
        'rect': [1.0, 1.0, 3.0, 2.0],
        'fig': {
            'type':
                'Figure',
            'dpi':
                100,
            'bbox': {
                'type': 'TransformedBbox',
                'bounds': (0.0, 0.0, 100.0, 100.0),
            },
            'axes': [{
                'type': 'AxesSubplot'
            }, {
                'type': 'Axes',
                'bbox': {
                    'type': 'TransformedBbox',
                    'bounds': (100.0, 100.0, 300.0, 200.0),
                }
            }]
        }
    }

    def test_text_line(self):
        box = self.create_text_box(show_border=False)
        box.text('bl', 'lower-left', dx=1.6, dy=0.8)
        res = box.summarize(**self.kwargs_summarize)
        sol = {
            **self.sol_base,
            'show_border':
                False,
            'elements': [{
                'type': 'TextBoxElement_Text',
                's': 'lower-left',
                'loc': {
                    'type': 'BoxLocation',
                    'va': 'baseline',
                    'ha': 'left',
                    'dx': 1.6,
                    'dy': 0.8,
                },
            }],
        }
        assert_summary_dict_is_subdict(superdict=res, subdict=sol)

    def test_text_block(self):
        box = self.create_text_box()
        box.text_block('mc', [('foo', 'bar'), ('hello', 'world')])
        res = box.summarize(**self.kwargs_summarize)
        sol = {
            **self.sol_base,
            'show_border':
                True,
            'elements': [
                {
                    'type': 'TextBoxElement_Text',
                    's': ('foo', 'bar'),
                },
                {
                    'type': 'TextBoxElement_Text',
                    's': ('hello', 'world'),
                },
            ],
        }
        assert_summary_dict_is_subdict(superdict=res, subdict=sol)

    def test_color_rect(self):
        box = self.create_text_box(show_border=True)
        box.color_rect('tr', 'red', 'black')
        res = box.summarize(**self.kwargs_summarize)
        sol = {
            **self.sol_base,
            'show_border':
                True,
            'elements': [{
                'type': 'TextBoxElement_ColorRect',
                'fc': 'red',
                'ec': 'black',
            }],
        }
        assert_summary_dict_is_subdict(superdict=res, subdict=sol)
