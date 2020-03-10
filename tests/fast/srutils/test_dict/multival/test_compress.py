#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.compress_multival_dicts``.
"""
# First-party
from srutils.dict import compress_multival_dicts


def test_simple():
    dcts = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 2, "c": 3},
    ]
    res = compress_multival_dicts(dcts)
    sol = {"a": [1, 4], "b": 2, "c": 3}
    assert res == sol


def test_extend_first():
    dcts = [
        {"a": [1, 2], "b": 3},
        {"a": 4, "b": 5},
    ]
    res = compress_multival_dicts(dcts)
    sol = {"a": [1, 2, 4], "b": [3, 5]}
    assert res == sol


def test_extend_other():
    dcts = [
        {"a": 1, "b": 3},
        {"a": [2, 4], "b": 5},
    ]
    res = compress_multival_dicts(dcts)
    sol = {"a": [1, 2, 4], "b": [3, 5]}
    assert res == sol


def test_extend_both():
    dcts = [
        {"a": [1, 2], "b": 3},
        {"a": 4, "b": 5},
        {"a": [2, 6], "b": [5, 7]},
    ]
    res = compress_multival_dicts(dcts)
    sol = {"a": [1, 2, 4, 6], "b": [3, 5, 7]}
    assert res == sol
