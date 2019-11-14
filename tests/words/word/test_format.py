#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test formatting methods of ``Word`` class.
"""
import pytest

from words import Word

def test_string():
    w = Word(en='hello world', de='Hallo Welt')
    assert w['en'].s == w['en']
    assert w['de'].s == w['de']
    assert isinstance(w['en'].s, str)
    assert isinstance(w['de'].s, str)

def test_capitalize_method():
    w = Word(en='the quick brown fox', de='der flinke braune Fuchs')
    assert w['en'].cap() == 'The quick brown fox'
    assert w['en'].cap(all=False) == 'The quick brown fox'
    assert w['en'].cap(all=True) == 'The Quick Brown Fox'
    assert w['de'].cap() == 'Der flinke braune Fuchs'
    assert w['de'].cap(all=False) == 'Der flinke braune Fuchs'
    assert w['de'].cap(all=True) == 'Der Flinke Braune Fuchs'
    assert isinstance(w['en'].cap(), str)
    assert isinstance(w['de'].cap(), str)
    assert isinstance(w['en'].cap(all=False), str)
    assert isinstance(w['de'].cap(all=False), str)
    assert isinstance(w['en'].cap(all=True), str)
    assert isinstance(w['de'].cap(all=True), str)

def test_capitalize_property():
    w = Word(en='the quick brown fox', de='der flinke braune Fuchs')
    assert w['en'].c == 'The quick brown fox'
    assert w['en'].C == 'The Quick Brown Fox'
    assert w['de'].c == 'Der flinke braune Fuchs'
    assert w['de'].C == 'Der Flinke Braune Fuchs'
    assert isinstance(w['en'].c, str)
    assert isinstance(w['de'].c, str)
    assert isinstance(w['en'].C, str)
    assert isinstance(w['de'].C, str)
