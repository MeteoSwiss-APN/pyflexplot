#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.decompress_multival_dict``.
"""
import pytest

from srutils.dict import decompress_multival_dict


def assert_dict_lists_equal(res, sol):
    assert isinstance(res, list)
    assert len(res) == len(sol)
    assert res == sol


# Some shortcuts
N = None
F = False
T = True


@pytest.mark.parametrize(
    "depth,sol",
    [(1, [{"a": 1, "b": 2}]), (2, [[{"a": 1, "b": 2}]]), (3, [[[{"a": 1, "b": 2}]]]),],
)
def test_singval(depth, sol):
    """Only single-value dict elements."""
    dct = {"a": 1, "b": 2}
    res = decompress_multival_dict(dct, depth=depth)
    assert_dict_lists_equal(res, sol)


@pytest.mark.parametrize(
    "depth,sol",
    [
        (1, [{"a": 1, "b": 3}, {"a": 2, "b": 3}]),
        (2, [[{"a": 1, "b": 3}], [{"a": 2, "b": 3}]]),
        (3, [[[{"a": 1, "b": 3}]], [[{"a": 2, "b": 3}]]]),
        (4, [[[[{"a": 1, "b": 3}]]], [[[{"a": 2, "b": 3}]]]]),
    ],
)
def test_multival_sing_shallow(depth, sol):
    """Single multi-value element (simple list)."""
    dct = {"a": [1, 2], "b": 3}
    res = decompress_multival_dict(dct, depth=depth)
    assert_dict_lists_equal(res, sol)


@pytest.mark.parametrize(
    "depth,sol",
    [
        (
            1,
            [
                {"a": 1, "b": 3, "c": 4},
                {"a": 1, "b": 3, "c": 5},
                {"a": 1, "b": 3, "c": 6},
                {"a": 2, "b": 3, "c": 4},
                {"a": 2, "b": 3, "c": 5},
                {"a": 2, "b": 3, "c": 6},
            ],
        ),
        (
            2,
            [
                [{"a": 1, "b": 3, "c": 4}],
                [{"a": 1, "b": 3, "c": 5}],
                [{"a": 1, "b": 3, "c": 6}],
                [{"a": 2, "b": 3, "c": 4}],
                [{"a": 2, "b": 3, "c": 5}],
                [{"a": 2, "b": 3, "c": 6}],
            ],
        ),
    ],
)
def test_multival_mult_shallow(depth, sol):
    """Multiple shallow multi-value dict elements (simple lists)."""
    dct = {"a": [1, 2], "b": 3, "c": [4, 5, 6]}
    res = decompress_multival_dict(dct, depth=depth)
    assert_dict_lists_equal(res, sol)


@pytest.mark.parametrize(
    "depth, sol",
    [
        (1, [{"a": [1, 2], "b": 4}, {"a": 3, "b": 4}]),
        (2, [[{"a": 1, "b": 4}, {"a": 2, "b": 4}], [{"a": 3, "b": 4}]]),
        (3, [[[{"a": 1, "b": 4}], [{"a": 2, "b": 4}]], [[{"a": 3, "b": 4}]]]),
        (4, [[[[{"a": 1, "b": 4}]], [[{"a": 2, "b": 4}]]], [[[{"a": 3, "b": 4}]]]]),
    ],
)
def test_multival_sing_deep(depth, sol):
    """Single deep multi-value dict elements (nested lists)."""
    dct = {"a": [[1, 2], 3], "b": 4}
    res = decompress_multival_dict(dct, depth=depth)
    assert_dict_lists_equal(res, sol)


@pytest.mark.parametrize(
    "depth, sol",
    [
        (
            1,
            [
                {"a": [1, 2], "b": 4, "c": 5},
                {"a": [1, 2], "b": 4, "c": [6, 7]},
                {"a": [3], "b": 4, "c": 5},
                {"a": [3], "b": 4, "c": [6, 7]},
            ],
        ),
        (
            2,
            [
                [{"a": 1, "b": 4, "c": 5}, {"a": 2, "b": 4, "c": 5}],
                [
                    {"a": 1, "b": 4, "c": 6},
                    {"a": 1, "b": 4, "c": 7},
                    {"a": 2, "b": 4, "c": 6},
                    {"a": 2, "b": 4, "c": 7},
                ],
                [{"a": 3, "b": 4, "c": 5},],
                [{"a": 3, "b": 4, "c": 6}, {"a": 3, "b": 4, "c": 7},],
            ],
        ),
        (
            3,
            [
                [[{"a": 1, "b": 4, "c": 5}], [{"a": 2, "b": 4, "c": 5}]],
                [
                    [{"a": 1, "b": 4, "c": 6}],
                    [{"a": 1, "b": 4, "c": 7}],
                    [{"a": 2, "b": 4, "c": 6}],
                    [{"a": 2, "b": 4, "c": 7}],
                ],
                [[{"a": 3, "b": 4, "c": 5}]],
                [[{"a": 3, "b": 4, "c": 6}], [{"a": 3, "b": 4, "c": 7}]],
            ],
        ),
    ],
)
def test_multival_mult_deep(depth, sol):
    """Multiple deep multi-value dict elements (nestsed lists)."""
    dct = {"a": [[1, 2], [3]], "b": 4, "c": [5, [6, 7]]}
    res = decompress_multival_dict(dct, depth=depth)
    assert_dict_lists_equal(res, sol)
