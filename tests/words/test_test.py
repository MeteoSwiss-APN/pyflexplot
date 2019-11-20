#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for module ``words.testing``.
"""
import functools

from words.test import TestWord
from words.test import TestWords

from srutils.testing import property_obj

property_word = functools.partial(property_obj, TestWord)
property_words = functools.partial(property_obj, TestWord)


class _Test_Basic():

    ws = property_words(['foo', 'bar', 'baz'])

    def test_all_implicit(self):
        assert self.ws['foo'] == 'foo'
        assert self.ws['bar'] == 'bar'
        assert self.ws['baz'] == 'baz'
