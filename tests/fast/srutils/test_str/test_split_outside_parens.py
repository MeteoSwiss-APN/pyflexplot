"""Test function ``srutils.str.split_outside_parens``."""
# Third-party
import pytest

# First-party
from srutils.str import split_outside_parens
from srutils.testing import not_raises


def test_no_parens():
    assert split_outside_parens("hello world") == ["hello", "world"]
    assert split_outside_parens("hello  world") == ["hello", "world"]
    assert split_outside_parens("foo, bar and baz", ",") == ["foo", " bar and baz"]


def test_parens():
    assert split_outside_parens("foo (bar baz) oof") == ["foo", "(bar baz)", "oof"]
    assert split_outside_parens("foo  (bar  baz)  oof") == ["foo", "(bar  baz)", "oof"]


def test_brackets():
    def f(s):
        return split_outside_parens(s, parens="[]")

    assert f("foo [bar baz] oof") == ["foo", "[bar baz]", "oof"]
    assert f("foo  [bar  baz]  oof") == ["foo", "[bar  baz]", "oof"]
    assert f("foo  ([bar  baz]  oof)") == ["foo", "([bar  baz]", "oof)"]


def test_mixed():
    def f(s):
        return split_outside_parens(s, parens=("([", ")]"))

    assert f("foo [bar baz] oof") == ["foo", "[bar baz]", "oof"]
    assert f("foo  (bar  baz)  oof") == ["foo", "(bar  baz)", "oof"]
    assert f("foo  ([bar  baz]  oof)") == ["foo", "([bar  baz]  oof)"]


def test_maxsplit():
    def f(s):
        return split_outside_parens(s, parens=("([", ")]"), maxsplit=1)

    assert f("foo [bar baz] oof") == ["foo", "[bar baz] oof"]
    assert f("foo  (bar  baz)  oof") == ["foo", "(bar  baz)  oof"]
    assert f("foo  ([bar  baz]  oof)") == ["foo", "([bar  baz]  oof)"]


def test_arg_parens_ok():
    s = "..."
    with not_raises(ValueError):
        split_outside_parens(s, parens="()")
        split_outside_parens(s, parens="[}")
        split_outside_parens(s, parens=("(", ")"))
        split_outside_parens(s, parens=("([{", ")]}"))


def test_arg_parens_fail():
    s = "..."
    with pytest.raises(ValueError):
        split_outside_parens(s, parens="")
        split_outside_parens(s, parens="[")
        split_outside_parens(s, parens="[])")
        split_outside_parens(s, parens="[])")
        split_outside_parens(s, parens=("[])",))
        split_outside_parens(s, parens=("[", ""))
        split_outside_parens(s, parens=("", "]"))
        split_outside_parens(s, parens=("[", "]", ")"))
