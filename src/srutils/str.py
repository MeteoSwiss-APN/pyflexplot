# -*- coding: utf-8 -*-
"""
String utilities.
"""
import re

#======================================================================


def to_varname(s, lower=True):
    """Reformat a string to a valid Python variable name."""

    err = f"cannot convert '{s}' to varname"

    # Check input is valid string
    try:
        v = str(s)
    except Exception as e:
        raise ValueError(f"{err}: not valid string")
    if not v:
        raise ValueError(f"{err}: is empty")

    # Turn spaces and dashes into underscores
    v = re.sub(r'[ -]', '_', v)

    # Drop all other special characters
    v = re.sub(r'[^a-zA-Z0-9_]', '', v)

    # Check validity
    if re.match(r'^[0-9]', v):
        raise ValueError(f"{err}: '{v}' starts with number")
    if not re.match(r'^[a-zA-Z0-9_]*$', v):
        raise ValueError(f"{err}: '{v}' contains invalid characters")

    if lower:
        v = v.lower()

    return v


def capitalize(s, preserve=True):
    """Capitalize a word, optionally preserving uppercase letters.

    Args:
        s (str): String.

        preserve (bool, optional): Whether to preserve capitalized
            letters. Defaults to True.

    """
    s = str(s)
    if not preserve:
        s = s.lower()
    return f'{s[0].upper()}{s[1:]}'


def titlecase(s, preserve=True):
    """Convert a string to titlecase.

    Args:
        s (str): String.

        preserve (bool, optional): Whether to preserve capitalized
            letters. Defaults to True.

    """
    # Words not to be capitalized
    # src: https://stackoverflow.com/a/35018815
    lower = [
        'the', 'a', 'an', 'as', 'at', 'but', 'by', 'for', 'in', 'of', 'off',
        'on', 'per', 'to', 'up', 'via', 'and', 'nor', 'or', 'so', 'yet'
    ]
    # Note: This is a rather simplistic implementation.
    # A more sophisticated implementation could be guided by, for instance:
    # https://titlecaseconverter.com/words-to-capitalize/?style=AP,APA,CMOS,MLA,NYT,WP

    s = str(s)
    if not preserve:
        s = s.lower()
    words_input = s.split(' ')
    words_title = (
        [capitalize(words_input[0])] +
        [w if w in lower else capitalize(w) for w in words_input[1:-1]])
    if len(words_input) >= 2:
        words_title += [capitalize(words_input[-1])]

    return ' '.join(words_title)
