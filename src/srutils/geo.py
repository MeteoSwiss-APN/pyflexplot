"""Geometric/geographic utilities."""
from __future__ import annotations

# Standard library
from typing import cast
from typing import Tuple
from typing import Union

FracDegsT = float
FullDegsT = int
FullMinsT = int
FullSecsT = int
DegsT = Union[
    FracDegsT,
    Tuple[FullDegsT],
    Tuple[FullDegsT, FullMinsT],
    Tuple[FullDegsT, FullMinsT, FullSecsT],
]


class Degrees:
    """Degrees, useful for instance to convert between notations."""

    def __init__(self, deg: DegsT):
        """Create an instance of ``Degrees``.

        Args:
            deg: Degrees in one of the following formats: fraction notation;
                full degrees; full degrees/minutes; full degrees/minutes/seconds.

        """
        self._frac: float

        # Check for fraction
        if isinstance(deg, (int, float)):
            self._frac = float(deg)
            return None

        # Check for most common non-stable sequence
        if isinstance(deg, set):
            raise ValueError("deg cannot be a set", deg)

        # Check for `(degs,)`
        try:
            (degs,) = cast(Tuple[FullDegsT], deg)
        except TypeError:
            pass
        else:
            self._frac = float(deg[0])
            return None

        # Check for `(degs, mins)`
        try:
            degs, mins = cast(Tuple[FullDegsT, FullMinsT], deg)
        except TypeError:
            pass
        else:
            self._frac = float(degs) + mins / 60.0
            return None

        # Check for `(degs, mins, secs)`
        try:
            degs, mins, secs = cast(Tuple[FullDegsT, FullMinsT, FullSecsT], deg)
        except TypeError:
            pass
        else:
            self._frac = float(degs) + mins / 60.0 + secs / 3600.0

        raise ValueError(
            f"invalid deg='{deg}'; must be `float` or `(degs, [mins, [secs,]])`"
        )

    def frac(self) -> float:
        """Return degrees as a fraction."""
        return self._frac

    def dms(self) -> tuple[FullDegsT, FullMinsT, FullSecsT]:
        """Return full degrees, minutes, and seconds."""
        degs = self._frac
        mins = degs % 1 * 60
        secs = mins % 1 * 60
        return int(degs), int(mins), int(secs)

    def degs(self) -> FullDegsT:
        """Return full degrees."""
        return self.dms()[0]

    def mins(self) -> FullMinsT:
        """Return full minutes."""
        return self.dms()[1]

    def secs(self) -> FullSecsT:
        """Return full seconds."""
        return self.dms()[2]
