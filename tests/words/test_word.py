#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for class ``Word`` in module ``words.word``.
"""

import pytest

from words import Word


class Test_Basic:
    """Test only basic behavior of a simple word."""

    w = Word(en='train', de='Zug')

    def test_str_default(self):
        assert self.w.default == 'en'
        assert str(self.w) == 'train'

    def test_lang_method(self):
        assert str(self.w.in_('en')) == 'train'
        assert str(self.w.in_('de')) == 'Zug'

    def test_lang_property(self):
        assert str(self.w.en) == 'train'
        assert str(self.w.de) == 'Zug'

    def test_langs(self):
        assert self.w.langs == ['en', 'de']


class Test_Simple:
    """Test all behavior of a simple word."""

    w = Word('high_school', en='high school', de='Mittelschule')

    def test_basics(self):
        assert self.w.default == 'en'
        assert str(self.w) == 'high school'
        assert str(self.w.en) == 'high school'
        assert str(self.w.de) == 'Mittelschule'
        assert self.w.in_('en') == self.w.en
        assert self.w.in_('de') == self.w.de
        assert self.w.langs == ['en', 'de']

    def test_key(self):
        assert self.w.key == 'high_school'

    def test_default(self):
        assert self.w.default == 'en'
        assert str(self.w) == 'high school'
        self.w.set_default('de')
        assert self.w.default == 'de'
        assert str(self.w) == 'Mittelschule'


class Test_Context:
    """Test a complex word with context-dependent translation."""

    w = Word(
        en='at', default='de', de=dict(place='bei', time='um', level='auf'))

    def test_default(self):
        assert str(self.w) == 'bei'

    def test_contexts_explicit(self):
        assert str(self.w.de.ctx('place')) == 'bei'
        assert str(self.w.de.ctx('time')) == 'um'
        assert str(self.w.de.ctx('level')) == 'auf'

    def test_contexts_default(self):
        assert str(self.w.ctx('place')) == 'bei'
        assert str(self.w.ctx('time')) == 'um'
        assert str(self.w.ctx('level')) == 'auf'


class Test_Creation:
    """Test creation of a ``Word`` instance."""

    def test_fail_noargs(self):
        with pytest.raises(ValueError):
            Word()

    def test_fail_nolangs(self):
        with pytest.raises(ValueError):
            Word('train')

    def test_nokey(self):
        Word(en='train', de='Zug')

    def test_fail_nokey(self):
        with pytest.raises(ValueError):
            Word(en='high school', de='Mittelschule')

    def test_key(self):
        Word('high_school', en='high school', de='Mittelschule')

    def test_fail_key(self):
        with pytest.raises(ValueError):
            Word('0', en='zero', de='Null')