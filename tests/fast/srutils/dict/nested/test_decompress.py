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


def test_retain_keys():
    dct = {
        "a": 1,
        "b": 2,
        "foo": {"c": 3, "bar": {"a": 4, "d": 5}, "baz": {"d": 6}},
        "bar": {"d": 7},
    }
    sol = [
        {
            "a": {"value": 4, "keys": ("foo", "bar")},
            "b": {"value": 2, "keys": tuple()},
            "c": {"value": 3, "keys": ("foo",)},
            "d": {"value": 5, "keys": ("foo", "bar")},
        },
        {
            "a": {"value": 1, "keys": tuple()},
            "b": {"value": 2, "keys": tuple()},
            "c": {"value": 3, "keys": ("foo",)},
            "d": {"value": 6, "keys": ("foo", "baz")},
        },
        {
            "a": {"value": 1, "keys": tuple()},
            "b": {"value": 2, "keys": tuple()},
            "d": {"value": 7, "keys": ("bar",)},
        },
    ]
    assert decompress_nested_dict(dct, retain_keys=True) == sol
