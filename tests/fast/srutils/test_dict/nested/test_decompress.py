"""
Tests for function ``srutils.dict.decompress_nested_dict``.
"""
# Third-party
import pytest  # type: ignore

# First-party
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
def test_basic(dct, sol):
    assert decompress_nested_dict(dct) == sol


def test_return_paths():
    dct = {
        "a": 1,
        "b": 2,
        "foo": {"c": 3, "bar": {"a": 4, "d": 5}, "baz": {"d": 6}},
        "bar": {"d": 7},
    }
    sol_values = [
        {"a": 4, "b": 2, "c": 3, "d": 5},
        {"a": 1, "b": 2, "c": 3, "d": 6},
        {"a": 1, "b": 2, "d": 7},
    ]
    sol_paths = [
        {"a": ("foo", "bar"), "b": (), "c": ("foo",), "d": ("foo", "bar")},
        {"a": (), "b": (), "c": ("foo",), "d": ("foo", "baz")},
        {"a": (), "b": (), "d": ("bar",)},
    ]
    values, paths = decompress_nested_dict(dct, return_paths=True)
    assert values == sol_values
    assert paths == sol_paths


class Test_MatchEnd:

    dct = {
        "_zz": {
            "a": 1,
            "b": 2,
            "yy": {"c": 3, "xx": {"d": 4}},
            "_xx": {"c": 5, "d": 6, "ww": {"e": 7}, "vv": {"e": 8, "uu": {"a": 9}}},
        },
    }

    sol_control = [
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 1, "b": 2, "c": 5, "d": 6, "e": 7},
        {"a": 9, "b": 2, "c": 5, "d": 6, "e": 8},
    ]

    sol_match = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 1, "b": 2, "c": 5, "d": 6, "e": 7},
        {"a": 1, "b": 2, "c": 5, "d": 6, "e": 8},
        {"a": 9, "b": 2, "c": 5, "d": 6, "e": 8},
    ]

    sol_paths = [
        {"a": ("_zz",), "b": ("_zz",), "c": ("_zz", "yy")},
        {"a": ("_zz",), "b": ("_zz",), "c": ("_zz", "yy"), "d": ("_zz", "yy", "xx")},
        {
            "a": ("_zz",),
            "b": ("_zz",),
            "c": ("_zz", "_xx"),
            "d": ("_zz", "_xx"),
            "e": ("_zz", "_xx", "ww"),
        },
        {
            "a": ("_zz",),
            "b": ("_zz",),
            "c": ("_zz", "_xx"),
            "d": ("_zz", "_xx"),
            "e": ("_zz", "_xx", "vv"),
        },
        {
            "a": ("_zz", "_xx", "vv", "uu"),
            "b": ("_zz",),
            "c": ("_zz", "_xx"),
            "d": ("_zz", "_xx"),
            "e": ("_zz", "_xx", "vv"),
        },
    ]

    @staticmethod
    def branch_end_criterion(key):
        return not key.startswith("_")

    def test_control(self):
        res = decompress_nested_dict(self.dct, branch_end_criterion=None)
        assert res == self.sol_control

    def test_match_end(self):
        res = decompress_nested_dict(
            self.dct, branch_end_criterion=self.branch_end_criterion
        )
        assert res == self.sol_match

    def test_return_paths(self):
        res_values, res_paths = decompress_nested_dict(
            self.dct, branch_end_criterion=self.branch_end_criterion, return_paths=True,
        )
        assert res_values == self.sol_match
        assert res_paths == self.sol_paths
