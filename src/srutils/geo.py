# -*- coding: utf-8 -*-
"""
Geometric/geographic utilities.
"""


class Degrees:
    """Degrees, useful for instance to convert between notations."""

    def __init__(self, deg):
        """Create an instance of ``Degrees``.

        Args:
            deg: Degrees in one of the following formats:
                float: Fraction notation.
                tuple[int*1]: Full degrees.
                tuple[int*2]: Full degrees/minutes.
                tuple[int*3]: Full degrees/minutes/seconds.

        """

        # Check for fraction
        if isinstance(deg, (int, float)):
            self._frac = float(deg)
            return None

        # Check for most common non-stable sequence
        if isinstance(deg, set):
            raise ValueError("deg cannot be a set", deg)

        # Check for `(degs,)`
        try:
            (degs,) = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0])
            return None

        # Check for `(degs, mins)`
        try:
            degs, mins = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0]) + deg[1] / 60.0
            return None

        # Check for `(degs, mins, secs)`
        try:
            degs, mins, secs = deg
        except TypeError:
            pass
        else:
            self._frac = float(degs) + mins / 60.0 + secs / 3600.0

        raise ValueError(
            f"invalid deg='{deg}'; " f"must be `float` or `(degs, [mins, [secs, ]])`"
        )

    def frac(self):
        """Return degrees as a fraction."""
        return self._frac

    def dms(self):
        """Return full degrees, minutes, and seconds (int)."""
        degs = self._frac
        mins = degs % 1 * 60
        secs = mins % 1 * 60
        return int(degs), int(mins), int(secs)

    def degs(self):
        """Return full degrees (float)."""
        return self.dms()[0]

    def mins(self):
        """Return full minutes (int)."""
        return self.dms()[1]

    def secs(self):
        """Return full seconds (int)."""
        return self.dms()[2]
