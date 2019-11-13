#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for class ``Word`` in module ``words.word``.
"""
import functools
import pytest

from attr import attrs, attrib
from words import Word

from utils import property_obj


property_word = functools.partial(property_obj, Word)


class Test_Basic:
    """Basic behavior of a simple word."""

    w = property_word(en='train', de='Zug')

    def test_str_default(self):
        assert self.w.default == 'en'
        assert self.w == 'train'

    def test_lang_method(self):
        assert self.w.in_('en') == 'train'
        assert self.w.in_('de') == 'Zug'

    def test_lang_property(self):
        assert self.w.en == 'train'
        assert self.w.de == 'Zug'

    def test_langs(self):
        assert self.w.langs == ['en', 'de']


class Test_Simple:
    """All behavior of a simple word."""

    w = property_word('high_school', en='high school', de='Mittelschule')

    def test_basics(self):
        assert self.w.default == 'en'
        assert self.w == 'high school'
        assert self.w.en == 'high school'
        assert self.w.de == 'Mittelschule'
        assert self.w.in_('en') == self.w.en
        assert self.w.in_('de') == self.w.de
        assert self.w.langs == ['en', 'de']

    def test_name(self):
        assert self.w.name == 'high_school'

    def test_default(self):
        w = self.w
        assert w.default == 'en'
        assert w == 'high school'
        w.set_default('de')
        assert w.default == 'de'
        assert w == 'Mittelschule'


class Test_Context_OneMany:
    """A word depending on context in one language."""

    w = property_word(
        en='at',
        de=dict(place='bei', time='um', level='auf'),
        default='de')

    def test_name(self):
        assert self.w.name == 'at'

    def test_default_context(self):
        assert self.w == 'bei'

    def test_explicit_contexts_explicit_lang(self):
        assert self.w.de.ctx('place') == 'bei'
        assert self.w.de.ctx('time') == 'um'
        assert self.w.de.ctx('level') == 'auf'

    def test_explicit_contexts_default_lang(self):
        assert self.w.ctx('place') == 'bei'
        assert self.w.ctx('time') == 'um'
        assert self.w.ctx('level') == 'auf'

    def test_explicit_contexts_other_lang(self):
        assert self.w.en.ctx('place') == 'at'
        assert self.w.en.ctx('time') == 'at'
        assert self.w.en.ctx('level') == 'at'


class Test_Context_ManyMany_Same:
    """A word depending on the same contexts in two languages."""

    w = property_word(
        en={'m': 'Mr.', 'f': 'Ms.'},
        de={'m': 'Herr', 'f': 'Frau'})

    def test_name(self):
        assert self.w.name == 'mr'

    def test_default_context(self):
        assert self.w.en == 'Mr.'
        assert self.w.de == 'Herr'

    def test_explicit_contexts(self):
        assert self.w.en.ctx('m') == 'Mr.'
        assert self.w.en.ctx('f') == 'Ms.'
        assert self.w.de.ctx('m') == 'Herr'
        assert self.w.de.ctx('f') == 'Frau'


class Test_Context_ManyMany_Diff:
    """A word depending on different contextx in two languages."""

    def test_fail_no_default_context(self):
        with pytest.raises(ValueError):
            Word(en={'abbr': 'int.'}, de={'abbr': 'int.', 'f': 'integrierte'})

    w = property_word(
        en={'*': 'integrated', 'abbr': 'int.'},
        de={'*': 'integriert', 'abbr': 'int.', 'f': 'integrierte'})

    def test_default_context(self):
        assert self.w.en == 'integrated'
        assert self.w.de == 'integriert'

    def test_shared_contexts(self):
        assert self.w.en.ctx('*') == 'integrated'
        assert self.w.de.ctx('*') == 'integriert'
        assert self.w.en.ctx('abbr') == 'int.'
        assert self.w.de.ctx('abbr') == 'int.'

    def test_unshared_context_defined(self):
        assert self.w.de.ctx('f') == 'integrierte'

    def test_unshared_context_undefined(self):
        assert self.w.en.ctx('f') == self.w.en.ctx('*')

@attrs
class AttrHolder:
    _val = attrib(default=None)

    def set(self, val):
        self._val = val

    def get(self):
        return self._val


class Test_QueryDefault:
    """Querying the default language on-the-fly."""

    w = property_word(en='train', de='Zug')

    def test_noquery(self):
        """Pass hard-coded default language."""
        w = self.w
        w.set_default(lang='de', query=None)
        assert w == 'Zug'
        w.set_default(lang='en', query=None)
        assert w == 'train'

    def test_query_hardcoded(self):
        """Query default language from hard-coded lamba."""
        w = self.w
        w.set_default(lang=None, query=lambda: 'en')
        assert w == 'train'
        w.set_default(lang=None, query=lambda: 'de')
        assert w == 'Zug'

    def test_query_dynamic(self):
        """Query default language dynamically from external object."""
        w = self.w
        default = AttrHolder()
        w.set_default(lang=None, query=default.get)
        default.set('de')
        w.set_default(lang=None, query=lambda: 'de')
        default.set('en')
        w.set_default(lang=None, query=lambda: 'en')

    def test_query_dynamic_precedence(self):
        """Ensure precedence of query over hard-coded default."""
        w = self.w
        default = AttrHolder()
        w.set_default(lang='de', query=default.get)
        default.set('de')
        w.set_default(lang=None, query=lambda: 'de')
        default.set('en')
        w.set_default(lang=None, query=lambda: 'en')


class Test_Creation:
    """Creation of a ``Word`` instance."""

    def test_fail_noargs(self):
        with pytest.raises(ValueError):
            Word()

    def test_fail_nolangs(self):
        with pytest.raises(ValueError):
            Word('train')

    def test_pass_noname(self):
        Word(en='train', de='Zug')

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
