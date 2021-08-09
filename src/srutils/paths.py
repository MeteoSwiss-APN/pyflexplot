"""Manipulate paths."""
from __future__ import annotations

# Standard library
import re
from typing import Optional
from typing import Sequence

# Local
from .format import format_numbers_range
from .str import sorted_paths


class PathsOrganizer:
    """Organize paths that share some characteristics."""

    def __init__(self, suffix: str = "", dup_sep: str = ".") -> None:
        """Create an instance of ``PathOrganizer``.

        Args:
            suffix (optional): Suffix of files; leading period is optional.

            dup_sep (optional): Separator between the base of the path and the
                number that distinguishes paths with the same base (which comes at
                the end before the optional suffix).

        """
        self._suffix = f".{suffix.lstrip('.')}"
        self._dup_sep = str(dup_sep)

    def group_related(self, paths: Sequence[str]) -> list[list[str]]:
        """Group paths by shared base name.

        Args:
            paths: Paths to be grouped.

        Example:
            >>> org = PathsOrganizer(suffix="x", dup_sep=".")
            >>> org.group_related(["bar.0.x", "foo.1.x", "foo.x", "bar.1.x"])
            [["foo.x", "foo.1.x"], ["bar.0.x", "bar.1.x"]]

        """
        rx_numbered = re.compile(
            f"{re.escape(self._dup_sep)}[0-9]+{re.escape(self._suffix)}$"
        )
        paths = list(paths)
        grouped_pdf_paths: list[list[str]] = []
        for path in list(paths):
            if path not in paths:
                # Already handled
                continue
            if rx_numbered.search(path):
                # Numbered, so not the first
                continue
            path_base = path.rstrip(self._suffix)
            paths.remove(path)
            grouped_pdf_paths.append([path])
            rx_related = re.compile(
                f"{re.escape(path_base)}\\.[0-9]+{re.escape(self._suffix)}"
            )
            for other_path in list(paths):
                if rx_related.search(other_path):
                    paths.remove(other_path)
                    grouped_pdf_paths[-1].append(other_path)
        return [
            sorted_paths(group, dup_sep=self._dup_sep)
            for group in grouped_pdf_paths
            if len(group) > 1
        ]

    def merge(self, paths: list[str]) -> str:
        """Format paths with shared base in a compact way.

        Args:
            paths: Paths to be grouped.

        """
        # SR_TMP <
        merged = paths[0]
        assert all(path.startswith(merged.rstrip(self._suffix)) for path in paths)
        # SR_TMP >
        return merged

    def format_compact(self, paths: list[str], syntax: str = "braces") -> str:
        """Format paths with shared base in a compact way.

        Args:
            paths: Paths to be grouped.

            syntax (optional): How to format the numbers that distinguish the
                paths:

                    - "braces": [1, 2, 3] -> "{1,2,3}"

        Example:
            >>> org = PathsOrganizer(suffix="x", dup_sep=".")
            >>> org.format_compact(["bar.x", "bar.1.x", "bar.2.x"])
            "bar.{,1,2}.x"

        """
        idcs: list[int] = []
        rx = re.compile(
            f"(?P<base>^.*?)({re.escape(self._dup_sep)}(?P<idx>[0-9]+))?"
            f"{re.escape(self._suffix)}$"
        )
        unnumbered = False
        base: Optional[str] = None
        for path in paths:
            match = rx.match(path)
            if not match:
                raise Exception(f"invalid path: {path}")
            base_i = match.group("base")
            if base is None:
                base = base_i
            elif base_i != base:
                raise Exception(f"different path bases: {base_i} != {base}")
            try:
                idx = int(match.group("idx"))
            except TypeError:
                unnumbered = True
            else:
                idcs.append(idx)
        value_error_syntax = ValueError(f"invalid syntax '{syntax}'")
        if len(idcs) == 1:
            s_idcs = str(next(iter(idcs)))
        else:
            if syntax == "braces":
                idcs_fmtd = format_numbers_range(
                    sorted(idcs), join_range="..", join_others=","
                )
                s_idcs = "{" + idcs_fmtd + "}"
            else:
                raise value_error_syntax
        if unnumbered:
            if syntax == "braces":
                s_idcs = f"{{,.{s_idcs}}}"
            else:
                raise value_error_syntax
        return f"{base}{s_idcs}{self._suffix}"
