#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for class ``Word`` in module ``words.word``.
"""

import pytest

from attr import attrs, attrib
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

    def test_name(self):
        assert self.w.name == 'high_school'

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

    def test_pseudo_contexts(self):
        assert str(self.w.en.ctx('place')) == 'at'
        assert str(self.w.en.ctx('time')) == 'at'
        assert str(self.w.en.ctx('level')) == 'at'


@attrs
class AttrHolder:
    _val = attrib(default=None)

    def set(self, val):
        self._val = val

    def get(self):
        return self._val


class Test_QueryDefault:
    """Test querying the default language on-the-fly."""

    w = Word(en='train', de='Zug')

    def test_noquery(self):
        """Pass hard-coded default language."""
        self.w.set_default(lang='de', query=None)
        assert str(self.w) == 'Zug'
        self.w.set_default(lang='en', query=None)
        assert str(self.w) == 'train'

    def test_query_hardcoded(self):
        """Query default language from hard-coded lamba."""
        self.w.set_default(lang=None, query=lambda: 'en')
        assert str(self.w) == 'train'
        self.w.set_default(lang=None, query=lambda: 'de')
        assert str(self.w) == 'Zug'

    def test_query_dynamic(self):
        """Query default language dynamically from external object."""
        default = AttrHolder()
        self.w.set_default(lang=None, query=default.get)
        default.set('de')
        self.w.set_default(lang=None, query=lambda: 'de')
        default.set('en')
        self.w.set_default(lang=None, query=lambda: 'en')

    def test_query_dynamic_precedence(self):
        """Ensure precedence of query over hard-coded default."""
        default = AttrHolder()
        self.w.set_default(lang='de', query=default.get)
        default.set('de')
        self.w.set_default(lang=None, query=lambda: 'de')
        default.set('en')
        self.w.set_default(lang=None, query=lambda: 'en')


class Test_Creation:
    """Test creation of a ``Word`` instance."""

    def test_fail_noargs(self):
        with pytest.raises(ValueError):
            Word()

    def test_fail_nolangs(self):
        with pytest.raises(ValueError):
            Word('train')

    def test_pass_noname(self):
        Word(en='train', de='Zug')

    def test_pass_fail_noname(self):
        with pytest.raises(ValueError):
            Word(en='high school', de='Mittelschule')

    def test_pass_name_implicit(self):
        Word('high_school', en='high school', de='Mittelschule')

    def test_pass_name_explicit(self):
        Word(name='high_school', en='high school', de='Mittelschule')

    def test_fail_name_implicit(self):
        with pytest.raises(ValueError):
            Word('0', en='zero', de='Null')

    def test_fail_name_explicit(self):
        with pytest.raises(ValueError):
            Word(name='0', en='zero', de='Null')

    def test_fail_inconsistent_contexts(self):
        with pytest.raises(ValueError):
            Word(
                en={
                    'foo': 'bar',
                    'hello': 'world'
                },
                de={
                    'bar': 'baz',
                    'hello': 'world'
                })
