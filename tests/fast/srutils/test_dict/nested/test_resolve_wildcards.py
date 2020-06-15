# -*- coding: utf-8 -*-
"""
Tests for function ``srutils.dict.nested_dict_resolve_wildcards``.
"""
# First-party
from srutils.dict import nested_dict_resolve_wildcards


def test_single_star_flat():
    """Update dicts at same level with contents of single-star dict."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "b": 1,
        },
        "bar": {
            "a": 2,
            "b": 3,
        },
        "*": {
            "c": 4,
        },
    }
    sol = {
        "foo": {
            "a": 0,
            "b": 1,
            "c": 4,  # "*"
        },
        "bar": {
            "a": 2,
            "b": 3,
            "c": 4,  # "*"
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_single_star_nested_flat():
    """Update dicts at same level with contents of flat single-star dict."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "baz": {
                    "c": 2,
                },
            },
            "zab": {
                "b": 3,
                "c": 4,
            },
            "*": {
                "d": 5,
            },
        },
        "asdf": {
            "a": 6,
            "fdsa": {
                "b": 7,
            },
            "*": {
                "c": 8,
            },
        },
        "*": {
            "e": 9,
        },
    }
    sol = {
        "foo": {
            "a": 0,
            "e": 9,
            "bar": {
                "b": 1,
                "d": 5,  # "*"
                "baz": {
                    "c": 2,
                },
            },
            "zab": {
                "b": 3,
                "c": 4,
                "d": 5,  # "*"
            },
        },
        "asdf": {
            "a": 6,
            "e": 9,  # "*"
            "fdsa": {
                "b": 7,
                "c": 8,  # "*"
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_single_star_nested_nested():
    """Update dicts at same level with contents of nested single-star dict."""
    # fmt: off
    dct = {
        "dog": {
            "a": 0,
            "cat": {
                "b": 1,
                "tree": {
                    "c": 2,
                },
            },
        },
        "wolf": {
            "d": 3,
        },
        "bird": {
            "cat": {
                "e": 4,
            }
        },
        "*": {
            "e": 5,
            "cat": {
                "b": 6,
                "tree": {
                    "c": 7,
                },
            },
        },
    }
    sol = {
        "dog": {
            "a": 0,
            "e": 5,  # "*"
            "cat": {
                "b": 6,  # "*"
                "tree": {
                    "c": 7,  # "*"
                },
            },
        },
        "wolf": {
            "d": 3,
            "e": 5,  # "*"
            "cat": {  # "*"
                "b": 6,
                "tree": {
                    "c": 7,
                },
            },
        },
        "bird": {
            "e": 5,
            "cat": {
                "b": 6,  # "*"
                "e": 4,
                "tree": {  # "*"
                    "c": 7,
                },
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_double_star_linear_flat():
    """Update all regular dicts with the contents of flat double-star dicts."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "baz": {
                    "c": 2,
                },
            },
        },
        "**": {
            "d": 3,
        },
    }
    sol = {
        "foo": {
            "a": 0,
            "d": 3,  # "**"
            "bar": {
                "b": 1,
                "d": 3,  # "**"
                "baz": {
                    "c": 2,
                    "d": 3,  # "**"
                },
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_double_star_linear_flat_with_criterion():
    """Update certain dicts with the contents of flat double-star dicts."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "baz": {
                    "c": 2,
                },
            },
        },
        "**": {
            "d": 3,
        },
    }
    sol = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "d": 3,  # "**"
                "baz": {
                    "c": 2,
                    "d": 3,  # "**"
                },
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(
        dct, double_criterion=lambda key: key.startswith("b")
    )
    assert res == sol


def test_double_star_nested_flat():
    """Update all regular dicts with the contents of flat double-star dicts."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "baz": {
                    "c": 2,
                },
            },
            "zab": {
                "b": 3,
                "c": 4,
            },
            "**": {
                "d": 5,
            },
        },
        "asdf": {
            "a": 6,
            "fdsa": {
                "b": 7,
            },
            "**": {
                "c": 8,
            },
        },
        "**": {
            "e": 9,
        } ,
    }
    sol = {
        "foo": {
            "a": 0,
            "e": 9,  # "**"
            "bar": {
                "b": 1,
                "d": 5,  # "**"
                "e": 9,  # "**"
                "baz": {
                    "c": 2,
                    "d": 5,  # "**"
                    "e": 9,  # "**"
                },
            },
            "zab": {
                "b": 3,
                "c": 4,
                "d": 5,  # "**"
                "e": 9,  # "**"
            },
        },
        "asdf": {
            "a": 6,
            "e": 9,  # "**"
            "fdsa": {
                "b": 7,
                "c": 8,  # "**"
                "e": 9,  # "**"
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_double_star_nested_linear_flat_with_criterion():
    """Update certain dicts with the contents of flat double-star dicts."""
    # fmt: off
    dct = {
        "_foo": {
            "a": 0,
            "_bar": {
                "b": 1,
                "baz": {
                    "c": 2,
                },
            },
            "zab": {
                "b": 3,
                "c": 4,
            },
            "**": {
                "d": 5,
            },
        },
        "asdf": {
            "a": 6,
            "_fdsa": {
                "b": 7,
            },
            "**": {
                "c": 8,
            },
        },
        "**": {
            "e": 9,
        },
    }
    sol = {
        "_foo": {
            "a": 0,
            "_bar": {
                "b": 1,
                "baz": {
                    "c": 2,
                    "d": 5,  # "**"
                    "e": 9,  # "**"
                },
            },
            "zab": {
                "b": 3,
                "c": 4,
                "d": 5,  # "**"
                "e": 9,  # "**"
            },
        },
        "asdf": {
            "a": 6,
            "e": 9,  # "**"
            "_fdsa": {
                "b": 7,
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(
        dct, double_criterion=lambda key: not key.startswith("_")
    )
    assert res == sol


def test_mixed_stars_flat():
    """Mixed flat single- and double-star wildcards."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
            },
            "baz": {
                "b": 2,
            },
            "*": {
                "zab": {
                    "c": 3,
                    "d": 5,
                },
                "rab": {
                    "c": 4,
                    "d": 6,
                },
            },
        },
        "**": {
            "asdf": {
                "e": 7,
            },
            "fdsa": {
                "e": 8,
            },
        },
    }
    sol = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "zab": {  # "*"
                    "c": 3,
                    "d": 5,
                    "asdf": {  # "**"
                        "e": 7,
                    },
                    "fdsa": {  # "**"
                        "e": 8,
                    },
                },
                "rab": {  # "*"
                    "c": 4,
                    "d": 6,
                    "asdf": {  # "**"
                        "e": 7,
                    },
                    "fdsa": {  # "**"
                        "e": 8,
                    },
                },
                "asdf": {  # "**"
                    "e": 7,
                },
                "fdsa": {  # "**"
                    "e": 8,
                },
            },
            "baz": {
                "b": 2,
                "zab": {  # "*"
                    "c": 3,
                    "d": 5,
                    "asdf": {  # "**"
                        "e": 7,
                    },
                    "fdsa": {  # "**"
                        "e": 8,
                    },
                },
                "rab": {  # "*"
                    "c": 4,
                    "d": 6,
                    "asdf": {  # "**"
                        "e": 7,
                    },
                    "fdsa": {  # "**"
                        "e": 8,
                    },
                },
                "asdf": {  # "**"
                    "e": 7,
                },
                "fdsa": {  # "**"
                    "e": 8,
                },
            },
            "asdf": {  # "**"
                "e": 7,
            },
            "fdsa": {  # "**"
                "e": 8,
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(dct)
    assert res == sol


def test_mixed_stars_with_criterion():
    """Mixed single- and double-star wildcards with double-star criterion."""
    # fmt: off
    dct = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
            },
            "baz+": {
                "b": 2,
            },
            "*": {
                "zab+": {
                    "c": 3,
                    "d": 5,
                },
                "rab": {
                    "c": 4,
                    "d": 6,
                },
            },
        },
        "**": {
            "asdf": {
                "e": 7,
            },
            "fdsa": {
                "e": 8,
            },
        },
    }
    sol = {
        "foo": {
            "a": 0,
            "bar": {
                "b": 1,
                "zab+": {  # "*"
                    "c": 3,
                    "d": 5,
                    "asdf": {  # "**"
                        "e": 7,
                    },
                    "fdsa": {  # "**"
                        "e": 8,
                    },
                },
                "rab": {  # "*"
                    "c": 4,
                    "d": 6,
                },
            },
            "baz+": {
                "b": 2,
                "zab+": {  # "*"
                    "c": 3,
                    "d": 5,
                    "asdf": {  # "**"
                        "e": 7,
                    },
                    "fdsa": {  # "**"
                        "e": 8,
                    },
                },
                "rab": {  # "*"
                    "c": 4,
                    "d": 6,
                },
                "asdf": {  # "**"
                    "e": 7,
                },
                "fdsa": {  # "**"
                    "e": 8,
                },
            },
        },
    }
    # fmt: on
    res = nested_dict_resolve_wildcards(
        dct, double_criterion=lambda key: key.endswith("+")
    )
    assert res == sol
