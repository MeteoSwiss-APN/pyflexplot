"""String utilities."""
# Standard library
import re
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


def join_multilines(lines: Sequence[str], *, indent: int = 0) -> str:
    """Join multi-line blocks, indenting each line in the blocks."""
    s_lines = []
    for line in lines:
        s_sub_lines = []
        for sub_line in line.split("\n"):
            s_sub_line = " " * indent + sub_line
            s_sub_lines.append(s_sub_line)
        s_line = "\n".join(s_sub_lines)
        s_lines.append(s_line)
    return "\n".join(s_lines)


def split_outside_parens(
    s: str,
    sep: Optional[str] = None,
    maxsplit: int = -1,
    *,
    parens: Union[str, Tuple[str, str]] = "()",
) -> List[str]:
    """Split a string at delimiters outside parentheses.

    Args:
        s: String to split.

        sep (optional): Separator at which the string is split. May be a
            regular expression. None means whitespace.

        maxsplit (optional): Maximum number of splits. -1 means no limit.

        parens (optional): Type of parentheses; either a string of two
            characters representing the opening and closing parenthesis,
            resprctively; or a tuple of two non-empty strings representing
            collections of opening and closing parentheses, respectively.

    """
    if sep is None:
        sep = " +"
    msg = "parens neither a two-element string nor a tuple of two non-empty strings"
    if len(parens) != 2:
        raise ValueError(f"{msg}: '{parens}'")
    opening, closing = parens  # type: ignore  # "unpacking a string is disallowed"
    if not opening or not closing:
        raise ValueError(f"{msg}: '{parens}'")
    if any(chr in sep for chrs in [opening, closing] for chr in chrs):
        raise ValueError(f"separator contains parens ('{opening}{closing}'): '{sep}'")
    if not re.findall(sep, s):
        return [s]

    # Find nesting levels
    levels: List[int] = []
    level = 0
    for ch in s:
        if ch in opening:
            level += 1
        elif ch in closing:
            level -= 1
        levels.append(level)
    if level != 0:
        raise ValueError(f"non-matching parens '{opening}'/'{closing}': '{s}'")

    # Split at nesting level zero only
    raw_splits = re.split(sep, s)
    splits: List[str] = [raw_splits.pop(0)]
    for match in re.finditer(sep, s):
        max_reached = 0 <= maxsplit < len(splits)
        if levels[match.start()] == 0 and not max_reached:
            splits.append("")
        else:
            splits[-1] += match.group()
        splits[-1] += raw_splits.pop(0)
    splits.extend(raw_splits)

    return splits


def sorted_paths(
    paths: Collection[str],
    dup_sep: Optional[str] = "-",
    *,
    key: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
) -> List[str]:
    """Sort paths, with numbered duplicates in numerical order.

    Duplicate paths are indicated with a trailing number ahead of the suffix,
    separated from the main part of the path by, e.g., a dash, which may be
    omitted for the first duplicate.

    Args:
        paths: Paths to be sorted.

        dup_sep (optional): Character(s) separating the main part of duplicate
            paths (minus suffix) from the number of the duplicate. If multiple
            characters are passed, each is used as a separator individually. If
            None is passed, duplicate paths are not treated specially and the
            result is identical to ``sorted(paths)``.

        key (optional): Function applied to each element before sorting.

        reverse (optional): Reverse sorting order.

    Example:
        >>> paths = ["foo.png", "foo-1.png", "foo-2.png", ..., "foo-10.png"]
        >>> sorted(paths)
        ["foo-1.png", "foo-10.png", "foo-2.png", ..., "foo-0.png", "foo.png"]
        >>> sort_paths(paths)
        ["foo.png", "foo-1.png", "foo-2.png", ..., "foo-10.png"]

    """
    if dup_sep is None:
        return sorted(paths, key=key, reverse=reverse)
    if "-" in dup_sep:
        # Move "-" to the end for compatibility with regex brackets
        dup_sep = dup_sep.replace("-", "") + "-"
    grouped_paths: Dict[str, Dict[int, str]] = {}
    rx = re.compile(
        r"\b(?P<base>.*?)([" + str(dup_sep) + r"](?P<num>[0-9]+))?(?P<suffix>\.\w+)\b"
    )
    for path in paths:
        match = rx.match(path)
        if match:
            base_path = match.group("base") + match.group("suffix")
            num = int(match.group("num")) if match.group("num") else -1
            if base_path not in grouped_paths:
                grouped_paths[base_path] = {}
            grouped_paths[base_path][num] = path
        else:
            raise ValueError(f"invalid path (not matching '{rx.pattern}'): {path}")
    sorted_paths_: List[str] = []
    for base_path in sorted(grouped_paths.keys(), key=key, reverse=reverse):
        paths_by_num = grouped_paths[base_path]
        for num in sorted(paths_by_num.keys(), reverse=reverse):
            path = paths_by_num[num]
            sorted_paths_.append(path)
    return sorted_paths_
