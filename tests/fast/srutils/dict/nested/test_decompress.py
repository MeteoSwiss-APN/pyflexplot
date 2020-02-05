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
        ({}, [{}]),
        ({"a": 1}, [{"a": 1}]),
        ({"a": 1, "b": 2}, [{"a": 1, "b": 2}]),
        #
        ({"a": 1, "foo": {"a": 1}}, [{"a": 1}]),
        (
            {
                "a": 1,
                "b": 2,
                "foo": {"c": 3, "bar": {"a": 4, "d": 5}, "baz": {"d": 6}},
                "bar": {"d": 7},
            },
            [
                {"a": 4, "b": 2, "c": 3, "d": 5},
                {"a": 1, "b": 2, "c": 3, "d": 6},
                {"a": 1, "b": 2, "d": 7},
            ],
        ),
    ],
)
def test(dct, sol):
    assert decompress_nested_dict(dct) == sol
