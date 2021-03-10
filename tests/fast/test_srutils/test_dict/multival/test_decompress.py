"""Test function ``srutils.dict.decompress_multival_dict``."""
# Third-party
import pytest  # type: ignore

# First-party
from srutils.dict import decompress_multival_dict
from srutils.exceptions import UnexpandableValueError


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
    [(1, [{"a": 1, "b": 2}]), (2, [[{"a": 1, "b": 2}]]), (3, [[[{"a": 1, "b": 2}]]])],
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
                [{"a": 3, "b": 4, "c": 5}],
                [{"a": 3, "b": 4, "c": 6}, {"a": 3, "b": 4, "c": 7}],
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


def test_skip():
    dct = {"a": ["foo", "bar"], "b": 1, "c": [2, 3], "d": (4, 5)}
    res = decompress_multival_dict(dct, skip=["a"])
    sol = [
        {"a": ["foo", "bar"], "b": 1, "c": 2, "d": 4},
        {"a": ["foo", "bar"], "b": 1, "c": 2, "d": 5},
        {"a": ["foo", "bar"], "b": 1, "c": 3, "d": 4},
        {"a": ["foo", "bar"], "b": 1, "c": 3, "d": 5},
    ]
    assert_dict_lists_equal(res, sol)


class Test_UnexpandableOk:
    dct = {"a": 1, "b": [2, 3], "c": {"d": 4}}

    def test_a_ok(self):
        try:
            decompress_multival_dict(self.dct, select="a")
            decompress_multival_dict(self.dct, select="a", unexpandable_ok=True)
        except UnexpandableValueError as e:
            raise AssertionError(f"unexpected exception: {repr(e)}") from e

    def test_a_fail(self):
        with pytest.raises(UnexpandableValueError):
            decompress_multival_dict(self.dct, select="a", unexpandable_ok=False)

    def test_b_ok(self):
        try:
            decompress_multival_dict(self.dct, select="b")
            decompress_multival_dict(self.dct, select="b", unexpandable_ok=True)
            decompress_multival_dict(self.dct, select="b", unexpandable_ok=False)
        except UnexpandableValueError as e:
            raise AssertionError(f"unexpected exception: {repr(e)}") from e

    def test_c_ok(self):
        try:
            decompress_multival_dict(self.dct, select="c")
            decompress_multival_dict(self.dct, select="c", unexpandable_ok=True)
        except UnexpandableValueError as e:
            raise AssertionError(f"unexpected exception: {repr(e)}") from e

    def test_c_fail(self):
        with pytest.raises(UnexpandableValueError):
            decompress_multival_dict(self.dct, select="c", unexpandable_ok=False)

    def test_d_ok(self):
        try:
            decompress_multival_dict(self.dct, select="d")
            decompress_multival_dict(self.dct, select="d", unexpandable_ok=True)
            decompress_multival_dict(self.dct, select="d", unexpandable_ok=False)
        except UnexpandableValueError as e:
            raise AssertionError(f"unexpected exception: {repr(e)}") from e
