#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.linearize_nested_dict``.
"""
import pytest

from srutils.dict import linearize_nested_dict


@pytest.mark.parametrize(
    "dct, sol",
    [
        ({}, [{}]),
        ({"a": 1}, [{"a": 1}]),
        ({"a": 1, "b": 2}, [{"a": 1, "b": 2}]),
        #
        ({"a": 1, "foo": {"a": 1}}, [{"a": 1, "foo": {"a": 1}}]),
    ],
)
def test_linear(dct, sol):
    assert linearize_nested_dict(dct) == sol


@pytest.mark.parametrize(
    "dct, sol",
    [
        ({"foo": {"a": 1}, "bar": {"a": 2}}, [{"foo": {"a": 1}}, {"bar": {"a": 2}}],),
        (
            {"a": 1, "foo": {"b": 2}, "bar": {"b": 3}},
            [{"a": 1, "foo": {"b": 2}}, {"a": 1, "bar": {"b": 3}}],
        ),
        (
            {
                "a": 1,
                "b": 2,
                "foo": {"c": 3, "bar": {"a": 4, "d": 5}, "baz": {"d": 6}},
                "bar": {"d": 7},
            },
            [
                {"a": 1, "b": 2, "foo": {"c": 3, "bar": {"a": 4, "d": 5}}},
                {"a": 1, "b": 2, "foo": {"c": 3, "baz": {"d": 6}}},
                {"a": 1, "b": 2, "bar": {"d": 7}},
            ],
        ),
    ],
)
def test_branched(dct, sol):
    assert linearize_nested_dict(dct) == sol
