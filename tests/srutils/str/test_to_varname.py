#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``to_varname`` in module ``srutils``.
"""
import pytest

from srutils.str import to_varname


def test_unchanged():
    assert to_varname('foo_bar') == 'foo_bar'


# Spaces are turned into underscores


def test_single_space():
    assert to_varname('foo bar') == 'foo_bar'


def test_multiple_spaces():
    assert to_varname('foo bar baz') == 'foo_bar_baz'


def test_repeated_spaces():
    assert to_varname('foo   bar  baz') == 'foo___bar__baz'


def test_leading_space():
    assert to_varname(' foo') == '_foo'


def test_trailing_space():
    assert to_varname('bar ') == 'bar_'


# Dashes are turned into underscores


def test_single_dash():
    assert to_varname('foo-bar') == 'foo_bar'


def test_multiple_dashes():
    assert to_varname('-foo--bar-baz---') == '_foo__bar_baz___'


# Numbers are retained, unless in the beginning


def test_numbers():
    assert to_varname('foo1bar2baz3') == 'foo1bar2baz3'


def test_fail_leading_number():
    with pytest.raises(ValueError):
        to_varname('1foo')


# Other special chars are dropped


def test_periods():
    assert to_varname('foo. bar.baz.') == 'foo_barbaz'


def test_various():
    assert to_varname('#foo@ +bar/-baz 123$') == 'foo_bar_baz_123'


def test_fail_leading_number_concealed():
    with pytest.raises(ValueError):
        to_varname('@#1foo')


# Optionally, all letters can be lowercased


def test_lower_default():
    assert to_varname(' Hello! ') != to_varname(' Hello! ', lower=False)
    assert to_varname(' Hello! ') == to_varname(' Hello! ', lower=True)


def test_lower_false():
    assert to_varname(' Hello World! ', lower=False) == '_Hello_World_'


def test_lower_true():
    assert to_varname(' Hello World! ', lower=True) == '_hello_world_'