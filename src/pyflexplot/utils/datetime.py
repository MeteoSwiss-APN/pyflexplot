# -*- coding: utf-8 -*-
"""
Datetime utilities.
"""
# Standard library
import time
from datetime import datetime
from datetime import timezone
from typing import Union


def init_datetime(raw: Union[int, str]) -> datetime:
    """Initialize a UTC datetime object by formatting an integer or string."""
    raw = str(raw)
    try:
        fmt = {
            4: "%Y",
            6: "%Y%m",
            8: "%Y%m%d",
            10: "%Y%m%d%H",
            12: "%Y%m%d%H%M",
            14: "%Y%m%d%H%M%S",
        }[len(raw)]
    except KeyError:
        raise NotImplementedError(f"datetime string with length {len(raw)}: '{raw}'")
    return datetime(*time.strptime(raw, fmt)[:6], tzinfo=timezone.utc)
