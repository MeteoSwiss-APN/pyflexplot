# -*- coding: utf-8 -*-
"""
String utilities.
"""
# Standard library
import re
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence


def sfmt(obj: Any, q: str = "'", *, none: str = "None") -> str:
    """Format a object to a string, while quoting strings and so forth."""
    if obj is None:
        return none
    elif isinstance(obj, str):
        return f"{q}{obj}{q}"
    else:
        return str(obj)


def to_varname(s, filter_invalid=None):
    """Reformat a string to a valid Python variable name.

    Valid characters are all letters, underscores, and numbers (except as the
    first characters). All other characters must either be converted to one of
    the former, or removed altogether. By default, they are all converted to
    underscores.

    Args:
        s (str): String to be reformatted.

        filter_invalid (callable, optional): A function applied to each invalid
            character to replace it by a valid one. Defaults to None.

            Example: Replace all dashes and spaces by underscores and remove
            all other invalid characters, pass the following:

            ``filter_special=lambda c: "_" if c in "- " else ""``

    """

    # Check input is valid string
    if not s:
        raise ValueError("s is empty", s)
    s = str(s)

    if filter_invalid is None:

        # pylint: disable=W0613  # unused-argument (s)
        # pylint: disable=E0102  # function-redefined
        def filter_invalid(s):
            return "_"

    # Filter all characters, ignoring potential leading numbers
    varname = _filter_s(s, filter_invalid)

    # Handle leading number (if necessary)
    if varname[0] in "0123456789":
        varname = filter_invalid(varname[0]) + varname[1:]

    check_is_valid_varname(varname)
    return varname


def _filter_s(s, filter_invalid):
    rx_valid = re.compile("[a-zA-Z0-9_]")
    varname = ""
    for c in s:
        if not rx_valid.match(c):
            try:
                c = filter_invalid(c)
            except TypeError as e:
                raise ValueError("invalid filter", e, filter_invalid, c)
            else:
                if not isinstance(c, str):
                    raise ValueError("filter must return str", c, filter_invalid, c)
        varname += c
    return varname


def check_is_valid_varname(s):
    """Raise ``ValueError`` if ``s`` is not a valid variable name."""

    if re.match(r"^[0-9]", s):
        raise ValueError("starts with number")

    if not re.match(r"^[a-zA-Z0-9_]*$", s):
        raise ValueError("contains invalid characters")


def is_valid_varname(s):
    """Check if ``s`` is a valid variable name."""
    try:
        check_is_valid_varname(s)
    except ValueError:
        return False
    else:
        return True


def capitalize(s, preserve=True):
    """Capitalize a word, optionally preserving uppercase letters.

    Args:
        s (str): String.

        preserve (bool, optional): Whether to preserve capitalized letters.
            Defaults to True.

    """
    s = str(s)
    if not preserve:
        s = s.lower()
    return f"{s[0].upper()}{s[1:]}"


def titlecase(s, preserve=True):
    """Convert a string to titlecase.

    Args:
        s (str): String.

        preserve (bool, optional): Whether to preserve capitalized letters.
            Defaults to True.

    """
    # TranslatedWords not to be capitalized
    # src: https://stackoverflow.com/a/35018815
    lower = [
        "the",
        "a",
        "an",
        "as",
        "at",
        "but",
        "by",
        "for",
        "in",
        "of",
        "off",
        "on",
        "per",
        "to",
        "up",
        "via",
        "and",
        "nor",
        "or",
        "so",
        "yet",
    ]
    # Note: This is a rather simplistic implementation.
    # A more sophisticated implementation could be guided by, for instance:
    # https://titlecaseconverter.com/words-to-capitalize/?style=AP,APA,CMOS,MLA,NYT,WP

    s = str(s)
    if not preserve:
        s = s.lower()
    words_input = s.split(" ")
    words_title = [capitalize(words_input[0])] + [
        w if w in lower else capitalize(w) for w in words_input[1:-1]
    ]
    if len(words_input) >= 2:
        words_title += [capitalize(words_input[-1])]

    return " ".join(words_title)


def ordinal(i):
    """Format an integer as an ordinal number."""
    if abs(i) % 10 == 1:
        sfx = {11: "th"}.get(abs(i) % 100, "st")
    elif abs(i) % 10 == 2:
        sfx = {12: "th"}.get(abs(i) % 100, "nd")
    elif abs(i) % 10 == 3:
        sfx = {13: "th"}.get(abs(i) % 100, "rd")
    else:
        sfx = "th"
    return f"{i}{sfx}"


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
    opening: str = "(",
    closing: str = ")",
) -> List[str]:
    """Split a string at delimiters outside parentheses.

    Args:
        s: String to split.

        sep (optional): Separator at which the string is split. May be a
            regular expression. None means whitespace.

        maxsplit (optional): Maximum number of splits. -1 means no limit.

        opening (optional): Opening character(s).

        closing (optional): Closing characters.

    """
    if sep is None:
        sep = " +"
    elif any(ch in sep for chs in [opening, closing] for ch in chs):
        raise ValueError(
            f"separator must not contain parens '{opening}'/'{closing}': '{sep}'"
        )
    elif sep not in s:
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
        max_reached = maxsplit >= 0 and len(splits) > maxsplit
        if levels[match.start()] == 0 and not max_reached:
            splits.append("")
        else:
            splits[-1] += match.group()
        splits[-1] += raw_splits.pop(0)
    splits.extend(raw_splits)

    return splits
