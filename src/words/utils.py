# -*- coding: utf-8 -*-
"""
Utils.
"""
import re


def to_varname(s):
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

    return v
