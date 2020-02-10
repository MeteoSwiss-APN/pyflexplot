#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.flatten_nested_dict``.
"""
import pytest

from srutils.dict import flatten_nested_dict
from srutils.exceptions import KeyConflictError


@pytest.mark.parametrize(
    "dct, sol",
    [
        # Flat, str keys
        ({}, {}),
        ({"a": 1}, {"a": 1}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        # 3x nested, one non-dict element each, str keys
        ({"a": 1, "foo": {"a": 1, "bar": {"a": 1}}}, {"a": 1}),
        ({"a": 1, "foo": {"a": 2, "bar": {"a": 3}}}, {"a": 3}),
        ({"a": 1, "foo": {"b": 2, "bar": {"c": 3}}}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1, "foo": {"b": 2, "bar": {"a": 3}}}, {"a": 3, "b": 2}),
        # 3x nested, two non-dict elements each, str keys
        (
            {"a": 1, "b": 2, "foo": {"a": 1, "b": 2, "baz": {"a": 1, "b": 2}}},
            {"a": 1, "b": 2},
        ),
        (
            {"a": 1, "b": 2, "foo": {"a": 3, "b": 4, "baz": {"a": 5, "b": 6}}},
            {"a": 5, "b": 6},
        ),
        (
            {"a": 1, "b": 2, "foo": {"c": 3, "d": 4, "baz": {"e": 5, "f": 6}}},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        ),
        (
            {"a": 1, "b": 2, "foo": {"a": 3, "c": 4, "baz": {"b": 5, "d": 6}}},
            {"a": 3, "b": 5, "c": 4, "d": 6},
        ),
        # 3x nested, increasing no. non-dict elements, str keys
        (
            {"a": 1, "foo": {"a": 1, "b": 2, "bar": {"a": 1, "b": 2, "c": 3}}},
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            {"a": 1, "foo": {"a": 2, "b": 3, "bar": {"a": 4, "b": 5, "c": 6}}},
            {"a": 4, "b": 5, "c": 6},
        ),
        (
            {"a": 1, "foo": {"b": 2, "c": 3, "bar": {"d": 4, "e": 5, "f": 6}}},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        ),
        (
            {"a": 1, "foo": {"b": 2, "c": 3, "bar": {"a": 4, "b": 5, "d": 6}}},
            {"a": 4, "b": 5, "c": 3, "d": 6},
        ),
        # 3x nested, decreasing no. non-dict elements, str keys
        (
            {"a": 1, "b": 2, "c": 3, "foo": {"b": 2, "c": 3, "bar": {"c": 3}}},
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            {"a": 1, "b": 2, "c": 3, "foo": {"b": 4, "c": 5, "bar": {"c": 6}}},
            {"a": 1, "b": 4, "c": 6},
        ),
        (
            {"a": 1, "b": 2, "c": 3, "foo": {"d": 4, "e": 5, "bar": {"f": 6}}},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        ),
        (
            {"a": 1, "b": 2, "c": 3, "foo": {"a": 4, "b": 5, "bar": {"c": 6}}},
            {"a": 4, "b": 5, "c": 6},
        ),
        # 3x, mixed no. non-dict elements, str keys
        (
            {"a": 1, "b": 2, "foo": {"c": 3, "bar": {"a": 1, "b": 2, "c": 3}}},
            {"a": 1, "b": 2, "c": 3},
        ),
        (
            {"a": 1, "b": 2, "foo": {"c": 3, "bar": {"a": 4, "b": 5, "c": 6}}},
            {"a": 4, "b": 5, "c": 6},
        ),
        (
            {"a": 1, "b": 2, "foo": {"c": 3, "bar": {"d": 4, "e": 5, "f": 6}}},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        ),
        (
            {"a": 1, "b": 2, "foo": {"c": 3, "bar": {"a": 4, "c": 5, "d": 6}}},
            {"a": 4, "b": 2, "c": 5, "d": 6},
        ),
    ],
)
def test_linear_nesting(dct, sol):
    """
    Basic functionality for linearly nested dicts.

    In case of conflicting keys, the value at the deeper nesting level wins.
    """
    assert flatten_nested_dict(dct) == sol


@pytest.mark.parametrize(
    "dct, sol",
    [
        # No key conflict
        ({"foo": {"a": 1}, "bar": {"baz": {"b": 2}}}, {"a": 1, "b": 2}),
        # Key conflict, but at different levels
        ({"foo": {"a": 1}, "bar": {"baz": {"a": 2}}}, {"a": 2}),
    ],
)
def test_branched_notie(dct, sol):
    """
    Branched nesting without a tie breaker.
    """
    assert flatten_nested_dict(dct) == sol


@pytest.mark.parametrize(
    "dct, sol", [({"foo": {"a": 1}, "bar": {"a": 2}}, Exception),],
)
def test_branched_notie_fail(dct, sol):
    """
    Failure without tie breaker in case of key conflict at same nesting level.
    """
    with pytest.raises(sol):
        flatten_nested_dict(dct)


class TestReturnPathsDepths:
    """Return the path and/or depth for all values as separate dicts."""

    dct = {"a": 1, "b": 2, "foo": {"c": 3, "bar": {"a": 4, "c": 5, "d": 6}}}
    values = {"a": 4, "b": 2, "c": 5, "d": 6}
    paths = {"a": ("foo", "bar"), "b": (), "c": ("foo", "bar"), "d": ("foo", "bar")}
    depths = {"a": 2, "b": 0, "c": 2, "d": 2}

    def test_paths(self):
        res = flatten_nested_dict(self.dct, return_paths=True)
        assert res.values == self.values
        assert res.paths == self.paths

    def test_depth(self):
        res = flatten_nested_dict(self.dct, return_depths=True)
        assert res.values == self.values
        assert res.depths == self.depths

    def test_return_path_and_depth(self):
        res = flatten_nested_dict(self.dct, return_paths=True, return_depths=True)
        assert res.values == self.values
        assert res.paths == self.paths
        assert res.depths == self.depths
