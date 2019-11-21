# -*- coding: utf-8 -*-
"""
String utilities.
"""
import re


def to_varname(s):
    """Reformat a string to a valid Python variable name."""

    def error(msg):
        raise ValueError(f"cannot convert '{s}' to varname: {msg}")

    # Check input is valid string
    try:
        v = str(s)
    except Exception as e:
        error("not valid string")
    if not v:
        error("is empty")

    # Turn all spaces and special characters into underscores
    v = re.sub(r"[^a-zA-Z0-9]", "_", v)

    # Turn leading number into underscore
    v = re.sub(r"^[0-9]", "_", v)

    # Check validity
    try:
        check_is_valid_varname(v)
    except ValueError as e:
        error(str(e))

    return v


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
