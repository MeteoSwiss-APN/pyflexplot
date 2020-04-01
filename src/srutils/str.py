# -*- coding: utf-8 -*-
"""
String utilities.
"""
# Standard library
import re
from typing import Any


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
                raise ValueError(f"invalid filter", e, filter_invalid, c)
            else:
                if not isinstance(c, str):
                    raise ValueError("filter must return str", c, filter_invalid, c)
        varname += c
    return varname


def check_is_valid_varname(s):
    """Raise ``ValueError`` if ``s`` is not a valid variable name."""

    if re.match(r"^[0-9]", s):
        raise ValueError(f"starts with number")

    if not re.match(r"^[a-zA-Z0-9_]*$", s):
        raise ValueError(f"contains invalid characters")


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
