"""Datetime utilities."""
from __future__ import annotations

# Standard library
import datetime as dt
import time
from itertools import starmap
from typing import Callable
from typing import overload
from typing import TypeVar
from typing import Union

# Third-party
from typing_extensions import Literal


def init_datetime(raw: Union[int, str]) -> dt.datetime:
    """Initialize a UTC datetime object by formatting an integer or string."""
    fmt = derive_datetime_fmt(raw)
    return dt.datetime(*time.strptime(str(raw), fmt)[:6], tzinfo=dt.timezone.utc)


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


ConvertT = TypeVar("ConvertT", bound=Union[int, str])


@overload
def datetime_range(
    start: Union[dt.datetime, int, str],
    end: Union[dt.datetime, int, str],
    step: Union[dt.timedelta, int, str],
    *,
    convert: Literal[None] = None,
    fmt: str = ...,
) -> list[dt.datetime]:
    ...


@overload
def datetime_range(
    start: Union[dt.datetime, int, str],
    end: Union[dt.datetime, int, str],
    step: Union[dt.timedelta, int, str],
    *,
    convert: Callable[[str], ConvertT],
    fmt: str = ...,
) -> list[ConvertT]:
    ...


def datetime_range(start, end, step, *, convert=None, fmt="%Y%m%d%H%M%S"):
    """Create a range of ``datetime`` objects.

    Args:
        start: First time step.

        end: Last time step.

        step: Time step duration in seconds.

        convert (optional): Return objects of this type instead of ``datetime``;
            formatted with ``fmt`` before conversion.

        fmt (optional): Format string applied to returned datetimes; only
            relevant if ``convert`` is not None.

    """
    if not isinstance(start, dt.datetime):
        start = init_datetime(str(start))
    if not isinstance(end, dt.datetime):
        end = init_datetime(str(end))
    if start > end:
        raise ValueError(f"start after end: {start} > {end}")
    if not isinstance(step, dt.timedelta):
        step = dt.timedelta(seconds=int(step))
    times: list[dt.datetime] = [start]
    while times[-1] + step <= end:
        times.append(times[-1] + step)
    if convert:
        strs = [step.strftime(fmt) for step in times]
        return list(map(convert, strs))
    return times
