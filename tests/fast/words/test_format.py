#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test formatting methods of ``TranslatedWord`` class.
"""
import pytest

from words import TranslatedWord


def test_string():
    w = TranslatedWord(en="hello world", de="Hallo Welt")
    assert w["en"].s == w["en"]
    assert w["de"].s == w["de"]
    assert isinstance(w["en"].s, str)
    assert isinstance(w["de"].s, str)


def test_capitalize_method():
    w = TranslatedWord(en="the quick brown FOX", de="der flinke braune FUCHS")
    assert w["en"].capital() == "The quick brown FOX"
    assert w["en"].capital(all=False) == "The quick brown FOX"
    assert w["en"].capital(all=True) == "The Quick Brown FOX"
    assert w["de"].capital() == "Der flinke braune FUCHS"
    assert w["de"].capital(all=False) == "Der flinke braune FUCHS"
    assert w["de"].capital(all=True) == "Der Flinke Braune FUCHS"
    assert w["en"].capital(all=False, preserve=False) == "The quick brown fox"
    assert w["en"].capital(all=True, preserve=False) == "The Quick Brown Fox"
    assert w["de"].capital(all=False, preserve=False) == "Der flinke braune fuchs"
    assert w["de"].capital(all=True, preserve=False) == "Der Flinke Braune Fuchs"
    assert isinstance(w["en"].capital(), str)
    assert isinstance(w["de"].capital(), str)
    assert isinstance(w["en"].capital(all=False), str)
    assert isinstance(w["de"].capital(all=False), str)
    assert isinstance(w["en"].capital(all=True), str)
    assert isinstance(w["de"].capital(all=True), str)


def test_capitalize_property():
    w = TranslatedWord(en="the quick brown fox", de="der flinke braune Fuchs")
    assert w["en"].c == "The quick brown fox"
    assert w["en"].C == "The Quick Brown Fox"
    assert w["de"].c == "Der flinke braune Fuchs"
    assert w["de"].C == "Der Flinke Braune Fuchs"
    assert isinstance(w["en"].c, str)
    assert isinstance(w["de"].c, str)
    assert isinstance(w["en"].C, str)
    assert isinstance(w["de"].C, str)


def test_title_method():
    w = TranslatedWord(
        en="the VITAMINS are IN my fresh california raisins",
        de="Die Vitamine sind in meinen frischen Kalifornischen Rosinen",
    )
    assert w["en"].title() == "The VITAMINS Are IN My Fresh California Raisins"
    assert (
        w["en"].title(preserve=True)
        == "The VITAMINS Are IN My Fresh California Raisins"
    )
    assert (
        w["en"].title(preserve=False)
        == "The Vitamins Are in My Fresh California Raisins"
    )
    assert (
        w["de"].title() == "Die Vitamine sind in meinen frischen Kalifornischen Rosinen"
    )
    assert (
        w["de"].title(preserve=True)
        == "Die Vitamine sind in meinen frischen Kalifornischen Rosinen"
    )
    assert (
        w["de"].title(preserve=False)
        == "Die vitamine sind in meinen frischen kalifornischen rosinen"
    )


def test_title_property():
    w = TranslatedWord(
        en="the VITAMINS are IN my fresh california raisins",
        de="Die Vitamine sind in meinen frischen Kalifornischen Rosinen",
    )
    assert w["en"].t == "The VITAMINS Are IN My Fresh California Raisins"
    assert w["de"].t == "Die Vitamine sind in meinen frischen Kalifornischen Rosinen"
