#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``words.word``."""
import pytest

from words import Words


class Test_Basic:
    """Test basic functionality for a few simple words."""

    ws = Words(
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
        assert str(self.ws.train) == 'train'
        assert str(self.ws.high_school) == 'high school'

    def test_langs(self):
        assert str(self.ws.train.de) == 'Zug'
        assert str(self.ws.high_school.de) == 'Mittelschule'


class Test_Complex:
    """Test all functionality for a more complex set of words."""

    ws = Words(
        default_='de',
        train={
            'en': 'train',
            'de': 'Zug',
        },
        high_school={
            'en': 'high school',
            'de': 'Mittelschule',
        },
        at={
            'en': 'at',
            'de': {
                'place': 'bei',
                'time': 'um',
                'level': 'auf',
            },
        },
    )

    def test_default(self):
        assert str(self.ws.train) == 'Zug'
        assert str(self.ws.high_school) == 'Mittelschule'
        assert str(self.ws.at) == 'bei'

    def test_change_default(self):
        assert self.ws.default_ == 'de'
        assert str(self.ws.train) == 'Zug'
        self.ws.set_default('en')
        assert self.ws.default_ == 'en'
        assert str(self.ws.train) == 'train'
