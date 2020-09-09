"""
Tests for function ``srutils.testing.assert_nested_equal``.
"""
# Third-party
import numpy as np
import pytest  # type: ignore

# First-party
from srutils.testing import assert_nested_equal


def test_flat_list():
    assert_nested_equal([1, 2, "hello", "world"], [1, 2, "hello", "world"])


def test_nested_list():
    assert_nested_equal([[1, "foo"], "bar", [[2]]], [[1, "foo"], "bar", [[2]]])


def test_mixed_list_tuple():
    assert_nested_equal([(1, "foo"), "bar", ([2],)], ([1, "foo"], "bar", [(2,)]))


def test_array_list_tuple():
    assert_nested_equal(
        [np.array([1, 2, 3]), (4, [5, 6])], [[1, 2, 3], [4, np.array([5, 6])]],
    )


def test_flat_dict():
    assert_nested_equal(
        {"a": 1, "b": "c", 2: "d", 3: 4}, {3: 4, "a": 1, "b": "c", 2: "d"},
    )


def test_nested_dict():
    assert_nested_equal(
        {1: {2: "a", "b": 3}, "c": "d", 4: 5}, {"c": "d", 1: {2: "a", "b": 3}, 4: 5},
    )


def test_flat_set():
    assert_nested_equal({1, 2, "a", "b"}, {"b", 1, 2, "a"})


def test_float_close_flat():
    obj1, obj2 = [1, 2, 3 + 1e-6], [1, 2, 3]
    with pytest.raises(AssertionError):
        assert_nested_equal(obj1, obj2)
    assert_nested_equal(obj1, obj2, float_close_ok=True)


def test_string():
    assert_nested_equal(["a"], ["a"])
    with pytest.raises(AssertionError):
        assert_nested_equal(["a"], ["b"])
