# -*- coding: utf-8 -*-
"""
String utilities.
"""
# Standard library
import re


def str_or_none(s, q="'", *, none="None"):
    if s is None:
        return none
    return f"{q}{s}{q}"


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

    def error(msg):
        raise ValueError(f"cannot convert '{s}' to varname: {msg}")

    # Check input is valid string
    try:
        s = str(s)
    except Exception as e:
        error("not valid string")
    if not s:
        error("is empty")

    if filter_invalid is None:
        # Default filter function
        filter_invalid = lambda c: "_"

    def filter(ch):
        """Apply ``filter_invalid`` to character ``ch``."""
        err = f"filtering '{ch}' with function {filter_invalid} failed"
        try:
            chf = filter_invalid(ch)
        except Exception as e:
            error(f"{err}: {type(e).__name__}('{e}')")
        if chf != str(chf):
            error(f"{err}: type {type(chf).__name__} of '{chf}' not str or equivalent")
        return str(chf)

    # Filter all characters, ignoring potential leading numbers
    rx_valid = re.compile("[a-zA-Z0-9_]")
    varname = "".join([c if rx_valid.match(c) else filter(c) for c in s])

    # Handle leading number (if necessary)
    if varname[0] in "0123456789":
        varname = filter(varname[0]) + varname[1:]

    # Check validity
    try:
        check_is_valid_varname(varname)
    except ValueError as e:
        error(str(e))

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
