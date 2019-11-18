#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``words.word``."""
import functools
import pytest

from words import Words

from srutils.testing import property_obj

property_words = functools.partial(property_obj, Words)


class Test_Basic:
    """Test for simple words."""

    ws = property_words(
        train={
            'en': 'train',
            'de': 'Zug',
        },
        high_school={
            'en': 'high school',
            'de': 'Mittelschule',
        },
    )

    def test_default(self):
        assert self.ws.get('train') == 'train'
        assert self.ws.get('high_school') == 'high school'

    def test_langs(self):
        assert self.ws.get('train').get_in('de') == 'Zug'
        assert self.ws.get('high_school').get_in('de') == 'Mittelschule'


class Test_Basic_BracketsInterface:
    """Test simple words with brackets interface."""

    ws = Test_Basic.ws

    def test_default(self):
        assert self.ws['train'] == 'train'
        assert self.ws['high_school'] == 'high school'

    def test_default_lang_none_get(self):
        assert self.ws['train'].get_in(None) == 'train'
        assert self.ws['high_school'].get_in(None) == 'high school'

    def test_default_lang_none_bracket(self):
        assert self.ws['train', None] == 'train'
        assert self.ws['high_school', None] == 'high school'

    def test_lang_get(self):
        assert self.ws['train'].get_in('de') == 'Zug'
        assert self.ws['high_school'].get_in('de') == 'Mittelschule'

    def test_lang_bracket(self):
        assert self.ws['train', 'de'] == 'Zug'
        assert self.ws['high_school', 'de'] == 'Mittelschule'


class Test_ContextDependent_OneToMany:
    """Test words with context-dependency in one language."""

    ws = property_words(
        default_lang='de',
        at={
            'en': 'at',
            'de': {
                'place': 'bei',
                'level': 'auf',
            },
        })

    def test_change_default_lang(self):
        ws = self.ws
        assert ws.default_lang == 'de'
        assert ws.get('at') == 'bei'
        ws.set_default_lang('en')
        assert ws.default_lang == 'en'
        assert ws.get('at') == 'at'

    def test_default_context(self):
        assert self.ws.get('at') == 'bei'
        assert self.ws.get('at').get_in('de') == 'bei'
        assert self.ws.get('at').get_in('en') == 'at'
        assert self.ws.get('at', 'de') == 'bei'
        assert self.ws.get('at', 'en') == 'at'

    def test_explicit_context__arg(self):
        assert self.ws.get('at').ctx('place') == 'bei'
        assert self.ws.get('at').ctx('level') == 'auf'
        assert self.ws.get('at').get_in('de').ctx('place') == 'bei'
        assert self.ws.get('at').get_in('de').ctx('level') == 'auf'
        assert self.ws.get('at').get_in('en').ctx('place') == 'at'
        assert self.ws.get('at').get_in('en').ctx('level') == 'at'

    def test_explicit_context_one_args(self):
        assert self.ws.get('at', None).ctx('place') == 'bei'
        assert self.ws.get('at', None).ctx('level') == 'auf'
        assert self.ws.get('at', 'de').ctx('place') == 'bei'
        assert self.ws.get('at', 'de').ctx('level') == 'auf'
        assert self.ws.get('at', 'en').ctx('place') == 'at'
        assert self.ws.get('at', 'en').ctx('level') == 'at'

    def test_explicit_context_three_args(self):
        assert self.ws.get('at', None, 'place') == 'bei'
        assert self.ws.get('at', None, 'level') == 'auf'
        assert self.ws.get('at', 'de', 'place') == 'bei'
        assert self.ws.get('at', 'de', 'level') == 'auf'
        assert self.ws.get('at', 'en', 'place') == 'at'
        assert self.ws.get('at', 'en', 'level') == 'at'


class Test_ContextDependent_OneToMany_BracketInterface:
    """Test bracket interface for one-to-many context-dep. words."""

    ws = Test_ContextDependent_OneToMany.ws

    def test_default_lang_default_context(self):
        assert self.ws['at'] == 'bei'
        assert self.ws['at'].get_in(None) == 'bei'
        assert self.ws['at'][None] == 'bei'
        assert self.ws['at', None] == 'bei'
        assert self.ws['at'].ctx(None) == 'bei'
        assert self.ws['at'].get_in(None).ctx(None) == 'bei'
        assert self.ws['at'][None].ctx(None) == 'bei'
        assert self.ws['at', None].ctx(None) == 'bei'
        assert self.ws['at', None, None] == 'bei'
        assert self.ws.get('at')[None] == 'bei'
        assert self.ws.get('at')[None].ctx(None) == 'bei'

    def test_explicit_lang_default_context(self):
        assert self.ws['at'].get_in('en') == 'at'
        assert self.ws['at']['en'] == 'at'
        assert self.ws['at', 'en'] == 'at'
        assert self.ws['at'].get_in('en').ctx(None) == 'at'
        assert self.ws['at']['en'].ctx(None) == 'at'
        assert self.ws['at', 'en'].ctx(None) == 'at'
        assert self.ws['at', 'en', None] == 'at'
        assert self.ws.get('at')['en'] == 'at'
        assert self.ws.get('at')['en'].ctx(None) == 'at'

    def test_default_lang_explicit_context(self):
        assert self.ws['at'].ctx('level') == 'auf'
        assert self.ws['at'].get_in(None).ctx('level') == 'auf'
        assert self.ws['at'][None].ctx('level') == 'auf'
        assert self.ws['at', None].ctx('level') == 'auf'
        assert self.ws['at', None, 'level'] == 'auf'
        assert self.ws.get('at')[None].ctx('level') == 'auf'

    def test_explicit_lang_explicit_context(self):
        assert self.ws['at'].get_in('en').ctx('level') == 'at'
        assert self.ws['at']['en'].ctx('level') == 'at'
        assert self.ws['at', 'en'].ctx('level') == 'at'
        assert self.ws['at', 'en', 'level'] == 'at'
        assert self.ws.get('at')['en'].ctx('level') == 'at'


class Test_ContextDependent_ManyToMany:
    """Test words with context-dependency in both languages."""

    ws = property_words(
        default_lang='de',
        integrated={
            'en': {
                '*': 'integrated',
                'abbr': 'int.',
            },
            'de': {
                'f': 'integrierte',
                '*': 'integriert',
                'abbr': 'int.',
            },
        })

    def test_change_default_lang(self):
        ws = self.ws
        assert ws.default_lang == 'de'
        assert ws.get('integrated') == 'integrierte'
        ws.set_default_lang('en')
        assert ws.default_lang == 'en'
        assert ws.get('integrated') == 'integrated'

    def test_default_context(self):
        assert self.ws.get('integrated') == 'integrierte'
        assert self.ws.get('integrated').get_in('de') == 'integrierte'
        assert self.ws.get('integrated').get_in('en') == 'integrated'
        assert self.ws.get('integrated', 'de') == 'integrierte'
        assert self.ws.get('integrated', 'en') == 'integrated'

    def test_explicit_context_one_arg(self):
        assert self.ws.get('integrated').ctx('*') == 'integriert'
        assert self.ws.get('integrated').ctx('f') == 'integrierte'
        assert self.ws.get('integrated').ctx('abbr') == 'int.'
        assert self.ws.get('integrated').get_in('de').ctx('*') == 'integriert'
        assert self.ws.get('integrated').get_in('de').ctx('f') == 'integrierte'
        assert self.ws.get('integrated').get_in('de').ctx('abbr') == 'int.'
        assert self.ws.get('integrated').get_in('en').ctx('*') == 'integrated'
        assert self.ws.get('integrated').get_in('en').ctx('f') == 'integrated'
        assert self.ws.get('integrated').get_in('en').ctx('abbr') == 'int.'

    def test_explicit_context_two_args(self):
        assert self.ws.get('integrated', None).ctx('*') == 'integriert'
        assert self.ws.get('integrated', None).ctx('f') == 'integrierte'
        assert self.ws.get('integrated', None).ctx('abbr') == 'int.'
        assert self.ws.get('integrated', 'de').ctx('*') == 'integriert'
        assert self.ws.get('integrated', 'de').ctx('f') == 'integrierte'
        assert self.ws.get('integrated', 'de').ctx('abbr') == 'int.'
        assert self.ws.get('integrated', 'en').ctx('*') == 'integrated'
        assert self.ws.get('integrated', 'en').ctx('f') == 'integrated'
        assert self.ws.get('integrated', 'en').ctx('abbr') == 'int.'

    def test_explicit_context_three_args(self):
        assert self.ws.get('integrated', None, '*') == 'integriert'
        assert self.ws.get('integrated', None, 'f') == 'integrierte'
        assert self.ws.get('integrated', None, 'abbr') == 'int.'
        assert self.ws.get('integrated', 'de', '*') == 'integriert'
        assert self.ws.get('integrated', 'de', 'f') == 'integrierte'
        assert self.ws.get('integrated', 'de', 'abbr') == 'int.'
        assert self.ws.get('integrated', 'en', '*') == 'integrated'
        assert self.ws.get('integrated', 'en', 'f') == 'integrated'
        assert self.ws.get('integrated', 'en', 'abbr') == 'int.'


class Test_ContextDependent_ManyToMany_BracketInterface:
    """Test bracket interface for many-to-many context-dep. words."""

    ws = Test_ContextDependent_ManyToMany.ws

    def test_default_lang_default_context(self):
        assert self.ws['integrated'] == 'integrierte'
        assert self.ws['integrated'].get_in(None) == 'integrierte'
        assert self.ws['integrated'][None] == 'integrierte'
        assert self.ws['integrated', None] == 'integrierte'
        assert self.ws['integrated'].ctx(None) == 'integrierte'
        assert self.ws['integrated'].get_in(None).ctx(None) == 'integrierte'
        assert self.ws['integrated'][None].ctx(None) == 'integrierte'
        assert self.ws['integrated', None].ctx(None) == 'integrierte'
        assert self.ws['integrated', None, None] == 'integrierte'
        assert self.ws.get('integrated')[None] == 'integrierte'
        assert self.ws.get('integrated')[None].ctx(None) == 'integrierte'

    def test_explicit_lang_default_context(self):
        assert self.ws['integrated'].get_in('en') == 'integrated'
        assert self.ws['integrated']['en'] == 'integrated'
        assert self.ws['integrated', 'en'] == 'integrated'
        assert self.ws['integrated'].get_in('en').ctx(None) == 'integrated'
        assert self.ws['integrated']['en'].ctx(None) == 'integrated'
        assert self.ws['integrated', 'en'].ctx(None) == 'integrated'
        assert self.ws['integrated', 'en', None] == 'integrated'
        assert self.ws.get('integrated')['en'] == 'integrated'
        assert self.ws.get('integrated')['en'].ctx(None) == 'integrated'

    def test_default_lang_explicit_context(self):
        assert self.ws['integrated'].ctx('*') == 'integriert'
        assert self.ws['integrated'].ctx('f') == 'integrierte'
        assert self.ws['integrated'].get_in(None).ctx('*') == 'integriert'
        assert self.ws['integrated'].get_in(None).ctx('f') == 'integrierte'
        assert self.ws['integrated'][None].ctx('*') == 'integriert'
        assert self.ws['integrated'][None].ctx('f') == 'integrierte'
        assert self.ws['integrated', None].ctx('*') == 'integriert'
        assert self.ws['integrated', None].ctx('f') == 'integrierte'
        assert self.ws['integrated', None, '*'] == 'integriert'
        assert self.ws['integrated', None, 'f'] == 'integrierte'
        assert self.ws.get('integrated')[None].ctx('*') == 'integriert'
        assert self.ws.get('integrated')[None].ctx('f') == 'integrierte'

    def test_explicit_lang_explicit_context(self):
        assert self.ws['integrated'].get_in('en').ctx('*') == 'integrated'
        assert self.ws['integrated'].get_in('en').ctx('f') == 'integrated'
        assert self.ws['integrated']['en'].ctx('*') == 'integrated'
        assert self.ws['integrated']['en'].ctx('f') == 'integrated'
        assert self.ws['integrated', 'en'].ctx('*') == 'integrated'
        assert self.ws['integrated', 'en'].ctx('f') == 'integrated'
        assert self.ws['integrated', 'en', '*'] == 'integrated'
        assert self.ws['integrated', 'en', 'f'] == 'integrated'
        assert self.ws.get('integrated')['en'].ctx('*') == 'integrated'
        assert self.ws.get('integrated')['en'].ctx('f') == 'integrated'
