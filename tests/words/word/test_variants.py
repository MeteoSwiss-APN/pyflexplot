#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for class ``ContextWord`` in module ``words.word``.
"""

import pytest

from words.word import ContextWord


def test_single_nolang():
    w = ContextWord(default_context='foo', foo='bar')
    assert w.lang is None
    assert w == 'bar'
    assert w.default_context == 'foo'


def test_single_lang():
    w = ContextWord(foo='Zug', lang='de')
    assert w.lang == 'de'
    assert w == 'Zug'
    assert w.default_context == 'foo'


def test_multiple():
    wf = ContextWord('de', place='bei', time='um', level='auf')
    assert wf.ctx('place') == 'bei'
    assert wf.ctx('time') == 'um'
    assert wf.ctx('level') == 'auf'
