#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for class ``WordVariants`` in module ``words.word``.
"""

import pytest

from words.word import WordVariants


def test_single_nolang():
    wf = WordVariants(default='train')
    assert wf.lang is None
    assert str(wf) == 'train'
    assert wf.default == 'default'


def test_single_lang():
    wf = WordVariants(foo='Zug', lang_='de')
    assert wf.lang == 'de'
    assert str(wf) == 'Zug'
    assert wf.default == 'foo'


def test_multiple():
    wf = WordVariants('de', place='bei', time='um', level='auf')
    assert wf.ctx('place') == 'bei'
    assert wf.ctx('time') == 'um'
    assert wf.ctx('level') == 'auf'


def test_context_lang():
    """Language  specifier ``lang_`` vs. a context named ``lang``."""
    wf = WordVariants(lang='foo', lang_='test')
    assert wf.lang == 'test'
    assert wf.ctx('lang') == 'foo'
