#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``to_varname`` in module ``srutils``.
"""
import pytest

from srutils.str import to_varname


def test_unchanged():
    assert to_varname("foo_bar") == "foo_bar"


# Spaces are turned into underscores


def test_single_space():
    assert to_varname("foo bar") == "foo_bar"


def test_multiple_spaces():
    assert to_varname("foo bar baz") == "foo_bar_baz"


def test_repeated_spaces():
    assert to_varname("foo   bar  baz") == "foo___bar__baz"


def test_leading_space():
    assert to_varname(" foo") == "_foo"


def test_trailing_space():
    assert to_varname("bar ") == "bar_"


# Dashes are turned into underscores


def test_single_dash():
    assert to_varname("foo-bar") == "foo_bar"


def test_multiple_dashes():
    assert to_varname("-foo--bar-baz---") == "_foo__bar_baz___"


# So are leading numbers


def test_leading_number():
    assert to_varname("1foo") == "_foo"
    assert to_varname(" 1foo") == "_1foo"


def test_numbers():
    assert to_varname("foo1bar2baz3") == "foo1bar2baz3"


# So are all other special characters


def test_periods():
    assert to_varname("foo. bar.baz.") == "foo__bar_baz_"


def test_various():
    assert to_varname("#foo@ +bar/-baz 123$") == "_foo___bar__baz_123_"
