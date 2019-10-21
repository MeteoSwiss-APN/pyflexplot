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
