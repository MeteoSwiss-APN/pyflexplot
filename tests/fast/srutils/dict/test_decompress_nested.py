#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.decompress_nested_dict``.
"""
import pytest

from srutils.dict import decompress_nested_dict


@pytest.mark.parametrize(
    "dct, sol",
    [
        ({}, {}),
        ({"a": 1}, {"a": 1}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test(dct, sol):
    assert decompress_nested_dict(dct) == sol
