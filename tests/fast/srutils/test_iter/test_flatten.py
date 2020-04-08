# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.iter.flatten``.
"""

# First-party
from srutils.iter import flatten


def test_empty():
    assert flatten([]) == []
    assert flatten(tuple()) == []


def test_flat():
    assert flatten([1, 2, 3]) == [1, 2, 3]
    assert flatten((1, 2, 3)) == [1, 2, 3]
    assert flatten(range(1, 4)) == [1, 2, 3]


def test_nested_flat_regular():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_nested_flat_irregular():
    assert flatten([1, [2, 3], [4]]) == [1, 2, 3, 4]


def test_nested_deep_regular():
    res = flatten(([([1], [2]), ([3], [4])], [([5], [6]), ([7], [8])]))
    assert res == [1, 2, 3, 4, 5, 6, 7, 8]


def test_nested_deep_irregular():
    res = flatten([1, [2, [3, [4, [5]], [6]]], [7, [8]]])
    assert res == [1, 2, 3, 4, 5, 6, 7, 8]


def test_max_depth():
    lst = [1, [2, [3, [4, [5]], [6]]], [7, [8]]]
    assert flatten(lst, max_depth=0) == [1, [2, [3, [4, [5]], [6]]], [7, [8]]]
    assert flatten(lst, max_depth=1) == [1, 2, [3, [4, [5]], [6]], 7, [8]]
    assert flatten(lst, max_depth=2) == [1, 2, 3, [4, [5]], [6], 7, 8]
    assert flatten(lst, max_depth=3) == [1, 2, 3, 4, [5], 6, 7, 8]
    assert flatten(lst, max_depth=4) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert flatten(lst, max_depth=-1) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert flatten(lst, max_depth=None) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_nested_select_cls():
    lst = [[[[1], [2]], [(3,), (4,)]], (((5,), [6]), ([7], (8,)))]
    tup = ([[[1], [2]], [(3,), (4,)]], (((5,), [6]), ([7], (8,))))
    assert flatten(lst, cls=list) == [1, 2, (3,), (4,), (((5,), [6]), ([7], (8,)))]
    assert flatten(tup, cls=tuple) == ([[[1], [2]], [(3,), (4,)]], 5, [6], [7], 8)
