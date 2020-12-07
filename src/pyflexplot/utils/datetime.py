"""Datetime utilities."""
# Standard library
import time
from datetime import datetime
from datetime import timezone
from itertools import starmap
from typing import Union


def init_datetime(raw: Union[int, str]) -> datetime:
    """Initialize a UTC datetime object by formatting an integer or string."""
    fmt = derive_datetime_fmt(raw)
    return datetime(*time.strptime(str(raw), fmt)[:6], tzinfo=timezone.utc)


def derive_datetime_fmt(raw: Union[int, str]) -> str:
    """Derive the datetime format string from the length of a raw value."""
    try:
        n = len(str(int(raw)))
    except ValueError as e:
        raise ValueError(
            f"invalid raw datetime '{raw}' of type {type(raw).__name__}"
        ) from e
    fmts = {
        4: "%Y",
        6: "%Y%m",
        8: "%Y%m%d",
        10: "%Y%m%d%H",
        12: "%Y%m%d%H%M",
        14: "%Y%m%d%H%M%S",
    }
    try:
        fmt = fmts[n]
    except KeyError as e:
        raise ValueError(
            f"raw datetime '{raw}' has unexpected length {n}; choices: "
            + ", ".join(starmap("{} ('{}'))".format, fmts.items()))
        ) from e
    else:
        return fmt
