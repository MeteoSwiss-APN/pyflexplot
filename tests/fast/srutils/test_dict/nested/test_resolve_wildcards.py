# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.nested_dict_resolve_wildcards``.
"""
# First-party
from srutils.dict import nested_dict_resolve_wildcards


def test_single_star_flat():
    """Update dicts at same level with contents of single-starred dict."""
    dct = {"foo": {"a": 0, "b": 1}, "bar": {"a": 2, "b": 3}, "*": {"c": 4}}
    sol = {"foo": {"a": 0, "b": 1, "c": 4}, "bar": {"a": 2, "b": 3, "c": 4}}
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_single_star_nested():
    """Update dicts at same level with contents of single-starred dict."""
    dct = {
        "foo": {
            "a": 0,
            "bar": {"b": 1, "baz": {"c": 2}},
            "zab": {"b": 3, "c": 4},
            "*": {"d": 5},
        },
        "asdf": {"a": 6, "fdsa": {"b": 7}, "*": {"c": 8}},
        "*": {"e": 9},
    }
    sol = {
        "foo": {
            "a": 0,
            "e": 9,
            "bar": {"b": 1, "d": 5, "baz": {"c": 2}},
            "zab": {"b": 3, "c": 4, "d": 5},
        },
        "asdf": {"a": 6, "e": 9, "fdsa": {"b": 7, "c": 8}},
    }
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_double_star_linear():
    """Update all regular dicts with the contents of double-starred dicts."""
    dct = {"foo": {"a": 0, "bar": {"b": 1, "baz": {"c": 2}}}, "**": {"d": 3}}
    res = nested_dict_resolve_wildcards(dct)
    sol = {"foo": {"a": 0, "d": 3, "bar": {"b": 1, "d": 3, "baz": {"c": 2, "d": 3}}}}
    assert res == sol


def test_double_star_linear_ends_only():
    """Update end-of-branch dicts with the contents of double-starred dicts."""
    dct = {"foo": {"a": 0, "bar": {"b": 1, "baz": {"c": 2}}}, "**": {"d": 3}}
    sol = {"foo": {"a": 0, "bar": {"b": 1, "baz": {"c": 2, "d": 3}}}}
    res = nested_dict_resolve_wildcards(dct, double_only_to_ends=True)
    assert res == sol


def test_double_star_nested():
    """Update all regular dicts with the contents of double-starred dicts."""
    dct = {
        "foo": {
            "a": 0,
            "bar": {"b": 1, "baz": {"c": 2}},
            "zab": {"b": 3, "c": 4},
            "**": {"d": 5},
        },
        "asdf": {"a": 6, "fdsa": {"b": 7}, "**": {"c": 8}},
        "**": {"e": 9},
    }
    sol = {
        "foo": {
            "a": 0,
            "e": 9,
            "bar": {"b": 1, "d": 5, "e": 9, "baz": {"c": 2, "d": 5, "e": 9}},
            "zab": {"b": 3, "c": 4, "d": 5, "e": 9},
        },
        "asdf": {"a": 6, "e": 9, "fdsa": {"b": 7, "c": 8, "e": 9}},
    }
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_double_star_nested_linear_ends_only():
    """Update end-of-branch dicts with the contents of double-starred dicts."""
    dct = {
        "foo": {
            "a": 0,
            "bar": {"b": 1, "baz": {"c": 2}},
            "zab": {"b": 3, "c": 4},
            "**": {"d": 5},
        },
        "asdf": {"a": 6, "fdsa": {"b": 7}, "**": {"c": 8}},
        "**": {"e": 9},
    }
    sol = {
        "foo": {
            "a": 0,
            "bar": {"b": 1, "baz": {"c": 2, "d": 5, "e": 9}},
            "zab": {"b": 3, "c": 4, "d": 5, "e": 9},
        },
        "asdf": {"a": 6, "fdsa": {"b": 7, "c": 8, "e": 9}},
    }
    res = nested_dict_resolve_wildcards(dct, double_only_to_ends=True)
    assert res == sol


def test_mixed_stars():
    """Mixed single- and double-star wildcards; based on 'real' Setup case."""
    dct = {
        "foo": {
            "a": 0,
            "bar": {"b": 1},
            "baz": {"b": 2},
            "*": {"zab": {"c": 3, "d": 5}, "rab": {"c": 4, "d": 6}},
        },
        "**": {"asdf": {"e": 7}, "fdsa": {"e": 8}},
    }
    sol = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "zab": {"c": 3, "d": 5, "asdf": {"e": 7}, "fdsa": {"e": 8}},
                "rab": {"c": 4, "d": 6, "asdf": {"e": 7}, "fdsa": {"e": 8}},
                "asdf": {"e": 7},
                "fdsa": {"e": 8},
            },
            "baz": {
                "b": 2,
                "zab": {"c": 3, "d": 5, "asdf": {"e": 7}, "fdsa": {"e": 8}},
                "rab": {"c": 4, "d": 6, "asdf": {"e": 7}, "fdsa": {"e": 8}},
                "asdf": {"e": 7},
                "fdsa": {"e": 8},
            },
            "asdf": {"e": 7},
            "fdsa": {"e": 8},
        },
    }
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_mixed_stars_ends_only():
    """Mixed single- and double-star wildcards; based on 'real' Setup case."""
    dct = {
        "foo": {
            "a": 0,
            "bar": {"b": 1},
            "baz": {"b": 2},
            "*": {"zab": {"c": 3, "d": 5}, "rab": {"c": 4, "d": 6}},
        },
        "**": {"asdf": {"e": 7}, "fdsa": {"e": 8}},
    }
    sol = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "zab": {"c": 3, "d": 5, "asdf": {"e": 7}, "fdsa": {"e": 8}},
                "rab": {"c": 4, "d": 6, "asdf": {"e": 7}, "fdsa": {"e": 8}},
            },
            "baz": {
                "b": 2,
                "zab": {"c": 3, "d": 5, "asdf": {"e": 7}, "fdsa": {"e": 8}},
                "rab": {"c": 4, "d": 6, "asdf": {"e": 7}, "fdsa": {"e": 8}},
            },
        },
    }
    res = nested_dict_resolve_wildcards(dct, double_only_to_ends=True)
    assert res == sol
