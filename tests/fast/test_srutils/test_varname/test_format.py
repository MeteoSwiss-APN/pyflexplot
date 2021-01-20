"""Test method ``srutils.str.VariableName.format``."""
# Third-party
import pytest  # type: ignore

# First-party
from srutils.varname import VariableName


class Test_Default:
    """By default, all invalid characters are replaced by underscores."""

    def test_unchanged(self):
        assert VariableName("foo_bar").format() == "foo_bar"

    class Test_Spaces:
        """Test spaces."""

        def test_single_space(self):
            assert VariableName("foo bar").format() == "foo_bar"

        def test_multiple_spaces(self):
            assert VariableName("foo bar baz").format() == "foo_bar_baz"

        def test_repeated_spaces(self):
            assert VariableName("foo   bar  baz").format() == "foo___bar__baz"

        def test_leading_space(self):
            assert VariableName(" foo").format() == "_foo"

        def test_trailing_space(self):
            assert VariableName("bar ").format() == "bar_"

    class Test_Dashes:
        """Test dashes."""

        def test_single_dash(self):
            assert VariableName("foo-bar").format() == "foo_bar"

        def test_multiple_dashes(self):
            assert VariableName("-foo--bar-baz---").format() == "_foo__bar_baz___"

    class Test_Numbers:
        """Test numbers."""

        def test_leading_number(self):
            assert VariableName("1foo").format() == "_foo"
            assert VariableName(" 1foo").format() == "_1foo"

        def test_internal_numbers(self):
            assert VariableName("foo1bar2baz3").format() == "foo1bar2baz3"

    class Test_Others:
        """Test others."""

        def test_periods(self):
            assert VariableName("foo. bar.baz.").format() == "foo__bar_baz_"

        def test_various(self):
            res = VariableName("#foo@ +bar/-baz 123$").format()
            assert res == "_foo___bar__baz_123_"


class Test_FilterInvalid:
    """Pass a custom filter function to replace invalid characters."""

    def test_default_none(self):
        for s in ["foo", "hello world", "1-foo_2-bar@baz66_+"]:
            assert VariableName(s).format(c_filter=None) == VariableName(s).format()

    class Test_Success:
        """Successful conversions."""

        @staticmethod
        def run(s, f):
            return VariableName(s).format(c_filter=f)

        def test_drop_all(self):
            f = lambda c: ""  # noqa:E731
            assert self.run(" hello world! ", f) == "helloworld"
            assert self.run("1-foo_2-bar@baz66_+", f) == "foo_2barbaz66_"

        def test_keep_spaces_dashes(self):
            f = lambda c: "_" if c in "- " else ""  # noqa:E731
            assert self.run(" hello world ", f) == "_hello_world_"
            assert self.run("1-foo_2-bar@baz66_+", f) == "_foo_2_barbaz66_"

        def test_keep_leading_number(self):
            f = lambda c: "_" if c in "0123456789" else ""  # noqa:E731
            assert self.run(" hello world ", f) == "helloworld"
            assert self.run("1-foo_2-bar@baz66_+", f) == "_foo_2barbaz66_"

    class Test_Failure:
        """Failed conversions."""

        @staticmethod
        def run(E, f):
            with pytest.raises(E):
                VariableName("1-Hello World! 0").format(c_filter=f)

        def test_replacement_none(self):
            self.run(ValueError, lambda c: None)

        def test_replacement_number(self):
            self.run(ValueError, lambda c: 3.14)

        def test_replacement_invalid(self):
            self.run(ValueError, lambda c: "@")

        def test_no_args(self):
            self.run(ValueError, lambda: "")

        def test_too_many_args(self):
            self.run(ValueError, lambda a, b, c: "")
