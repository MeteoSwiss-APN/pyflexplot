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


def test_return_paths():
    dct = {
        "a": 1,
        "b": 2,
        "foo": {"c": 3, "bar": {"a": 4, "d": 5}, "baz": {"d": 6}},
        "bar": {"d": 7},
    }
    sol_values = [
        {"a": 4, "b": 2, "c": 3, "d": 5,},
        {"a": 1, "b": 2, "c": 3, "d": 6,},
        {"a": 1, "b": 2, "d": 7,},
    ]
    sol_paths = [
        {"a": ("foo", "bar"), "b": (), "c": ("foo",), "d": ("foo", "bar"),},
        {"a": (), "b": (), "c": ("foo",), "d": ("foo", "baz"),},
        {"a": (), "b": (), "d": ("bar",),},
    ]
    values, paths = decompress_nested_dict(dct, return_paths=True)
    assert values == sol_values
    assert paths == sol_paths
