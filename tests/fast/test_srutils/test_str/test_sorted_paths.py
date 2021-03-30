"""Test function ``srutils.str.sorted_paths``."""
# Standard library
import dataclasses as dc
from typing import List
from typing import Optional

# Third-party
import pytest

# First-party
from srutils.str import sorted_paths


def test_nodup():
    paths = ["foo.py", "bar.c", "baz.json"]
    assert sorted_paths(paths) == sorted(paths)
    assert sorted_paths(paths, reverse=True) == sorted(paths, reverse=True)
    assert sorted_paths(paths, key=len) == sorted(paths, key=len)


@dc.dataclass
class _TestCase:
    description: str
    paths: List[str]  # in same order as expected after ``sorted_paths(...)``
    dup_sep: Optional[str] = "-"
    builtin_ok: bool = False

    def sorted(self) -> List[str]:
        return sorted_paths(sorted(self.paths), self.dup_sep)

    def err_msg(self, msg: str, p1: List[str], p2: List[str]) -> str:
        return f"{self.description}:\n{msg}:\n  {p1}\n  {p2}"

    def test(self) -> None:
        self._test_builtin()
        p1 = self.sorted()
        p2 = self.paths
        assert p1 == p2, self.err_msg("sorted_paths != paths", p1, p2)

    def _test_builtin(self) -> None:
        """Compare ``sorted_paths`` and the builtin ``sorted``."""
        p1 = self.sorted()
        p2 = sorted(self.paths)
        if self.builtin_ok:
            assert p1 == p2, self.err_msg("sorted_paths != sorted", p1, p2)
        else:
            assert p1 != p2, self.err_msg("sorted_paths == sorted", p1, p2)


def test_TestCase():
    case = _TestCase("test", ["foo.png", "bar.png", "bar-1.png", "bar-10.png"])
    assert sorted_paths(case.paths) == case.sorted()


@pytest.mark.parametrize(
    "case",
    [
        _TestCase(
            "one duplicate type, all numbered, all same magnitude",
            ["foo-1.png", "foo-2.png", "foo-3.png"],
            builtin_ok=True,
        ),
        _TestCase(
            "one duplicate type, first unnumbered, all same magnitude",
            ["foo.png", "foo-1.png", "foo-2.png"],
        ),
        _TestCase(
            "one duplicate type, different magnitudes",
            ["foo.png"] + list(map("foo-{}.png".format, range(1, 21))),
        ),
        _TestCase(
            "two duplicate types, plus a non-duplicate",
            ["bar.c", "bar-1.c", "bar-10.c", "baz.c", "foo-1.c", "foo-10.c"],
        ),
        _TestCase(
            "different separator",
            ["bar.f", "bar.1.f", "bar.10.f", "foo-1.c", "foo-10.c", "foo.c"],
            dup_sep=".",
        ),
        _TestCase(
            "multiple separators",
            ["bar.f", "bar.1.f", "bar.10.f", "foo.c", "foo-1.c", "foo-10.c"],
            dup_sep="-.",
        ),
    ],
)
def test_sorted_paths(case):
    case.test()
