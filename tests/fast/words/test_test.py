#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``words.testing``.
"""
# Standard library
import functools

# First-party
from srutils.testing import property_obj
from words.test import TranslatedTestWord

property_word = functools.partial(property_obj, TranslatedTestWord)
property_words = functools.partial(property_obj, TranslatedTestWord)


class _Test_Basic:

    ws = property_words(["foo", "bar", "baz"])

    def test_all_implicit(self):
        assert self.ws["foo"] == "foo"
        assert self.ws["bar"] == "bar"
        assert self.ws["baz"] == "baz"
