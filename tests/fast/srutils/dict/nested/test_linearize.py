#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.linearize_nested_dict``.
"""
# Third-party
import pytest

# First-party
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


class Test_MatchEnd:
    """
    Create a separate branch ending at each key matching a criterion.
    """

    dct = {
        "_u": {
            "a": 1,
            "v": {
                "b": 2,
                "_w": {"c": 3, "x": {"d": 4}, "y": {"a": 5, "d": 6}},
                "x": {"c": 7, "y": {"d": 8}},
            },
            "_w": {"b": 9, "x": {"d": 10}, "y": {"z": {"a": 11, "e": 12}}},
        }
    }

    sol_control = [
        {"_u": {"a": 1, "v": {"b": 2, "_w": {"c": 3, "x": {"d": 4}}}}},  # 1
        {"_u": {"a": 1, "v": {"b": 2, "_w": {"c": 3, "y": {"a": 5, "d": 6}}}}},  # 2
        {"_u": {"a": 1, "v": {"b": 2, "x": {"c": 7, "y": {"d": 8}}}}},  # 4
        {"_u": {"a": 1, "_w": {"b": 9, "x": {"d": 10}}}},  # 5
        {"_u": {"a": 1, "_w": {"b": 9, "y": {"z": {"a": 11, "e": 12}}}}},  # 7
    ]

    sol_match = [
        {"_u": {"a": 1, "v": {"b": 2}}},  # 0
        {"_u": {"a": 1, "v": {"b": 2, "_w": {"c": 3, "x": {"d": 4}}}}},  # 1
        {"_u": {"a": 1, "v": {"b": 2, "_w": {"c": 3, "y": {"a": 5, "d": 6}}}}},  # 2
        {"_u": {"a": 1, "v": {"b": 2, "x": {"c": 7}}}},  # 3
        {"_u": {"a": 1, "v": {"b": 2, "x": {"c": 7, "y": {"d": 8}}}}},  # 4
        {"_u": {"a": 1, "_w": {"b": 9, "x": {"d": 10}}}},  # 5
        {"_u": {"a": 1, "_w": {"b": 9, "y": {}}}},  # 6
        {"_u": {"a": 1, "_w": {"b": 9, "y": {"z": {"a": 11, "e": 12}}}}},  # 7
    ]

    @staticmethod
    def match_end(key):
        return not key.startswith("_")

    def test_control(self):
        """
        Control test w/o matching function, returning only complete branches.
        """
        res = linearize_nested_dict(self.dct, match_end=None)
        assert res == self.sol_control

    def test_match(self):
        """
        Return an additional partial branch for each key not starting with "_".
        """
        res = linearize_nested_dict(self.dct, match_end=self.match_end)
        assert res == self.sol_match
