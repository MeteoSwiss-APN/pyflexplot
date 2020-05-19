# -*- coding: utf-8 -*-
"""
Some utilities.

TODO Split module up into `exceptions`, `summarize`, and `format`!
"""
# Standard library
import re
import warnings
from dataclasses import dataclass
from typing import Collection
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""


def check_float_ok(f: float, ff0t: str) -> bool:
    if f != np.inf and f >= 1.0:
        return bool(re.match(f"^{int(f)}" + r"\.[0-9]+$", ff0t))
    return (f == 0.0) or (float(ff0t) != 0.0)


def format_float(
    f: float,
    fmt_e0: Optional[str] = None,
    fmt_f0: Optional[str] = None,
    fmt_e1: Optional[str] = None,
    fmt_f1: Optional[str] = None,
) -> str:
    """Auto-format a float to floating-point or exponential notation.

    Very small and very large numbers are formatted in exponential notation.
    Numbers close enough to zero that they can be written in floating point
    format without exceeding the width of the exponential format are written in
    the former.

    Args:
        f: Number to format.

        fmt_e0 (optional): Exponential-notation format string used to create
            the string that is compared to that produced with ``fmt_f0`` to
            decide the appropriate notation. Defaults to '{f:e}'.

        fmt_f0 (optional): Floating-point notation format string used to create
            the string that is compared to that produced with ``fmt_e0`` to
            decide the appropriate notation. Defaults to '{f:f}'.

        fmt_e1 (optional): Exponential-notation format string used to create
            the return string if exponential notation has been found to be
            appropriate. Defaults to ``fmt_e0``.

        fmt_f1 (optional): Floating-point notation format string used to create
            the return string if floating point notation has been found to be
            appropriate. Defaults to '{f:f}' with the resulting string trimmed
            to the length of that produced with ``fmt_e0``.

    Returns:
        Formatted number.

    Algorithm:
        - Format ``f`` in both exponential notation with ``fmt_e0`` and in
          floating point notation with ``fmt_f0``, resulting in the string
          ``fe0`` and ``ff0``, respectively.

        - Trim ``ff0`` to the length of ``fe0``, resulting in ``ff0t``.

        - If ``ff0t`` is a valid floating point number, then floating point
          notation is the notation of choice. Criteria:

            - For numbers > 1.0, the following elements are preserved:

                - the leading minus sign, if the number is negative;
                - all integer digits;
                - the period; and
                - at least one fractional digit.

            - For numbers < 1.0, the following elements are preserved:

                - the leading minus sign, if the number is negative;
                - one zero integer digit;
                - the period; and
                - at least one non-zero fractional digit (provided the number
                  is non-zero).

        - Otherwise, exponential notation is the notation of choice.

        - Finally, ``f`` is formatted with ``fmt_f1`` or ``fmt_e1``, depending
          on whether floating point or exponential notation, respectively, has
          been determined as the notation of choice.

    """
    f = float(f)

    rx_e = re.compile(r"^{f:[0-9,]*\.?[0-9]*[eE]}$")
    rx_f = re.compile(r"^{f:[0-9,]*\.?[0-9]*f}$")
    for fmt in [fmt_e0, fmt_f0, fmt_e1, fmt_f1]:
        if fmt is not None:
            if not rx_e.match(fmt) and not rx_f.match(fmt):
                raise ValueError(f"invalid format string: '{fmt}'", fmt)

    if fmt_e0 is None:
        fmt_e0 = "{f:e}"
    if fmt_f0 is None:
        fmt_f0 = "{f:f}"

    fe0 = fmt_e0.format(f=f)
    ff0 = fmt_f0.format(f=f)

    n = len(fe0)
    ff0t = ff0[:n]

    if check_float_ok(f, ff0t):
        if fmt_f1 is not None:
            return fmt_f1.format(f=f)
        return ff0t
    elif fmt_e1 is not None:
        return fmt_e1.format(f=f)
    else:
        return fe0


# pylint: disable=R0913  # too-many-arguments
def format_range(
    numbers: Collection[float],
    fmt: str = "g",
    delta: float = 1,
    join_range: str = "-",
    join_others: str = ",",
    range_min: int = 3,
) -> str:
    """Format numbers individually or, if they are consecutive, as a range.

    Args:
        numbers: Collection of numbers to be formatted.

        fmt (optional): How to format individual numbers, e.g., "03d" for zero-
            padded three-digit integers.

        delta (optional): Differences between consecutive numbers.

        join_range (optional): Character(s) used to join the first and last
            number in a range.

        join_others (optional): Character(s) used to join non-consecutive
            numbers or ranges.

        range_min (optional): Minimum number of consecutive numbers to be
            formatted as a range.

    """
    template = f"{{num:{fmt}}}"
    consecutive_numbers_groups: List[List[float]] = []
    previous_number: Optional[float] = None
    for number in sorted(numbers):
        if previous_number is None or number - previous_number != delta:
            consecutive_numbers_groups.append([])
        consecutive_numbers_groups[-1].append(number)
        previous_number = number
    formatted_numbers_and_ranges: List[str] = []
    for consecutive_numbers in consecutive_numbers_groups:
        if len(consecutive_numbers) < range_min:
            numbers_to_format = consecutive_numbers
            join = join_others
        else:
            numbers_to_format = [consecutive_numbers[0], consecutive_numbers[-1]]
            join = join_range
        formatted_numbers: List[str] = []
        for number in numbers_to_format:
            try:
                formatted_numbers.append(template.format(num=number))
            except ValueError:
                raise Exception("cannot format number with template", number, template)
        formatted_numbers_and_ranges.append(join.join(formatted_numbers))
    return join_others.join(formatted_numbers_and_ranges)


def format_level_ranges(
    levels: Sequence[float],
    style: Optional[str] = None,
    widths: Optional[Tuple[int, int, int]] = None,
    extend: Optional[str] = None,
    align: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Format a list of level ranges in a certain style.

    Args:
        levels: Levels between the ranges.

        style (optional): Formatting style (options and examples below).
            Defaults to 'base'.

        widths (optional): Tuple with the minimum character widths of,
            respectively, the left ('lower-than'), center (operator), and right
            ('greater-than') parts of the ranges. Defaults to style-specific
            values.

        extend (optional): Whether the range is closed ('none'), open at the
            lower ('min') or the upper ('max') end, or both ('both'). Same as
            the ``extend`` keyword of, e.g., ``matplotlib.pyplot.contourf``.
            Defaults to 'none'.

        align (optional): Horizontal alignment of the left and right components
            (the center component with the operator is always center-aligned).
            Options: 'left' (both components left-aligned), 'right' (both
            components right-aligned), 'center', (left/right component
            right/left-aligned), and 'edges' (left/right component left/right-
            aligned). Defaults to 'center'.

        **kwargs: Additional style-specific keyword arguments used to
            initialize the respective formatter class. See individual formatter
            classes for details.

    Returns:
        Formatted level range strings, each of which represents the range
            between two successive ``levels``. Depending on ``extend``, the
            number of strings is equal to ('min' or 'max'), one smaller
            ('none'), or one greater ('both') than the number of ``levels``.

    Styles:
        +-------+----------------+--------------------+---------------+
        | style | ex. below      | ex. between        | ex. above     |
        +-------+----------------+--------------------+---------------+
        | base  | '< 10.0'       | '10.0-20.0'        | '>= 20.0'     |
        | int   | '< 10'         | '10-19'            | '>= 20'       |
        | math  | '(-inf, 10.0)' | '[10.0, 20.0)'     | '[20.0, inf)' |
        | up    | '< 10.0'       | '>= 10.0'          | '>= 20.0'     |
        | down  | '< 10.0'       | '< 20.0'           | '>= 20.0'     |
        | and   | '< 10.0'       | '>= 10.0 & < 20.0' | '>= 20.0'     |
        | var   | '10.0 > v'     | '10.0 <= v < 20.0' | 'v >= 20'     |
        +-------+----------------+--------------------+---------------+

    """
    if style is None:
        style = "base"
    if extend is None:
        extend = "none"
    if align is None:
        align = "center"
    else:
        align_choices = ["left", "right", "center", "edges"]
        if align not in align_choices:
            raise ValueError(
                f"invalid value '{align}' of argument align; "
                f"must be one of {','.join(align_choices)}"
            )
    formatters = {
        "base": LevelRangeFormatter,
        "int": LevelRangeFormatterInt,
        "math": LevelRangeFormatterMath,
        "up": LevelRangeFormatterUp,
        "down": LevelRangeFormatterDown,
        "and": LevelRangeFormatterAnd,
        "var": LevelRangeFormatterVar,
    }
    try:
        cls = formatters[style]
    except AttributeError:
        raise ValueError(f"unknown style '{style}'; options: {sorted(formatters)}")
    else:
        formatter = cls(widths=widths, extend=extend, align=align, **kwargs)
    return formatter.format_multiple(levels)


@dataclass
class Component:
    """Auxiliary class to pass results between formatter methods."""

    s: str
    ntex: int

    @classmethod
    def create(cls, arg: Union[str, Tuple[str, int]]) -> "Component":
        if isinstance(arg, str):
            s, ntex = arg, 0
        else:
            s = arg[0]
            ntex = arg[1] or len(s)
        return cls(s, ntex)


@dataclass
class Components:
    """Auxiliary class to pass results between formatter methods."""

    left: Component
    center: Component
    right: Component

    @classmethod
    def create(
        cls,
        left: Union[str, Tuple[str, int]],
        center: Union[str, Tuple[str, int]],
        right: Union[str, Tuple[str, int]],
    ) -> "Components":
        return Components(
            left=Component.create(left),
            center=Component.create(center),
            right=Component.create(right),
        )


class LevelRangeFormatter:
    """Format level ranges, e.g., for legends of color contour plots."""

    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
    ) -> None:
        """Create an instance of ``LevelRangeFormatter``.

        Args: See ``format_level_ranges``.

        """
        if widths is None:
            widths = (5, 3, 5)
        if extend is None:
            extend = "none"
        if align is None:
            align = "center"
        if rstrip_zeros is None:
            rstrip_zeros = False
        self.widths: Tuple[int, int, int] = widths
        self.extend: str = extend
        self.align: str = align
        self.rstrip_zeros: bool = rstrip_zeros

        # Declare attributes
        self._max_val: Optional[float] = None

        self._check_widths(self.widths)

    def _check_widths(self, widths: Tuple[int, int, int]) -> None:
        try:
            # pylint: disable=W0612  # unused-variable
            wl, wc, wr = [int(w) for w in widths]
        except (ValueError, TypeError):
            raise ValueError(f"widths is not a tree-int tuple: {widths}")

    def format_multiple(self, levels: Sequence[float]) -> List[str]:
        labels: List[str] = []
        self._max_val = max(levels)
        if self.extend in ("min", "both"):
            labels.append(self.format(None, levels[0]))
        for lvl0, lvl1 in zip(levels[:-1], levels[1:]):
            labels.append(self.format(lvl0, lvl1))
        if self.extend in ("max", "both"):
            labels.append(self.format(levels[-1], None))
        return labels

    def format(self, lvl0: Optional[float], lvl1: Optional[float]) -> str:
        if self._max_val is None:
            self._max_val = max([lvl0, lvl1])

        cs = self._format_components(lvl0, lvl1)

        s_l = cs.left.s
        s_c = cs.center.s
        s_r = cs.right.s

        dc = "^"
        dl, dr = dict(left="<<", right=">>", center="><", edges="<>")[self.align]

        if self.rstrip_zeros:

            def rstrip_zeros(s):
                if "e" in s or "E" in s:
                    return s
                rx_str = r"(?<!\.)0\b(?!\.)"
                while re.search(rx_str, s):
                    s = re.sub(rx_str, "", s)
                return s

            s_l = rstrip_zeros(s_l)
            s_r = rstrip_zeros(s_r)

        wl, wc, wr = self.widths

        s_l = f"{{:{dl}{wl + cs.left.ntex}}}".format(s_l)
        s_c = f"{{:{dc}{wc + cs.center.ntex}}}".format(s_c)
        s_r = f"{{:{dr}{wr + cs.right.ntex}}}".format(s_r)

        return f"{s_l}{s_c}{s_r}"

    def _format_components(
        self, lvl0: Optional[float], lvl1: Optional[float],
    ) -> Components:
        open_left = lvl0 in (None, np.inf)
        open_right = lvl1 in (None, np.inf)
        if open_left and open_right:
            raise ValueError("range open at both ends")
        elif open_left:
            assert lvl1 is not None  # mypy
            return self._format_open_left(lvl1)
        elif open_right:
            assert lvl0 is not None  # mypy
            return self._format_open_right(lvl0)
        else:
            assert lvl0 is not None and lvl1 is not None  # mypy
            return self._format_closed(lvl0, lvl1)

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:
        lvl0_fmtd = self._format_level(lvl0)
        lvl1_fmtd = self._format_level(lvl1)
        op_fmtd = r"$\tt -$"
        ntex_c = len(op_fmtd) - 1
        s_l = lvl0_fmtd
        s_c = op_fmtd
        s_r = lvl1_fmtd
        return Components.create(s_l, (s_c, ntex_c), s_r)

    def _format_open_left(self, lvl: float) -> Components:
        return self._format_open_core(lvl, r"$\tt <$")

    def _format_open_right(self, lvl: float) -> Components:
        return self._format_open_core(lvl, r"$\tt \geq$")

    def _format_open_core(self, lvl: float, op: str, *, len_op: int = 1) -> Components:
        lvl_fmtd = self._format_level(lvl)
        ntex_c = len(op) - len_op
        s_c = op
        s_r = lvl_fmtd
        return Components.create("", (s_c, ntex_c), s_r)

    def _format_level(self, lvl: float) -> str:
        return format_float(lvl, "{f:.0E}")


class LevelRangeFormatterInt(LevelRangeFormatter):
    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
    ) -> None:
        if widths is None:
            widths = (2, 3, 2)
        if rstrip_zeros is True:
            warnings.warn(f"{type(self).__name__}: force rstrip_zeros=False")
            rstrip_zeros = False
        super().__init__(
            widths=widths, extend=extend, align=align, rstrip_zeros=rstrip_zeros,
        )

    def _format_components(
        self, lvl0: Optional[float], lvl1: Optional[float],
    ) -> Components:
        # if lvl1 is not None:
        if lvl1 is not None:
            # SR_DBG <
            if lvl0 is not None:
                # SR_DBG >
                lvl1 = lvl1 - 1
                if lvl0 == lvl1:
                    return Components.create("", "", self._format_level(lvl1))
        return super()._format_components(lvl0, lvl1)

    def _format_level(self, lvl: float) -> str:
        if int(lvl) != float(lvl):
            warnings.warn(f"{type(self).__name__}._format_level: not an int: {lvl}")
        return f"{{:>{len(str(self._max_val))}}}".format(lvl)

    # def _format_open_left(self, lvl: float) -> Components:
    #     return self._format_open_core(lvl, r"$\tt\leq$")


class LevelRangeFormatterMath(LevelRangeFormatter):
    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
    ) -> None:
        if widths is None:
            widths = (6, 2, 6)
        super().__init__(
            widths=widths, extend=extend, align=align, rstrip_zeros=rstrip_zeros,
        )

    def _format_components(
        self, lvl0: Optional[float], lvl1: Optional[float],
    ) -> Components:
        return Components.create(
            "-inf" if lvl0 is None else f"[{self._format_level(lvl0)}",
            ",",
            "inf" if lvl1 is None else f"{self._format_level(lvl1)})",
        )


class LevelRangeFormatterUp(LevelRangeFormatter):
    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
    ) -> None:
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(
            widths=widths, extend=extend, align=align, rstrip_zeros=rstrip_zeros,
        )

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:
        op_fmtd = r"$\tt \geq$"
        ntex_c = len(op_fmtd) - 1
        s_c = op_fmtd
        s_r = self._format_level(lvl0)
        return Components.create("", (s_c, ntex_c), s_r)


class LevelRangeFormatterDown(LevelRangeFormatter):
    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
    ) -> None:
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(
            widths=widths, extend=extend, align=align, rstrip_zeros=rstrip_zeros,
        )

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:
        op_fmtd = r"$\tt <$"
        ntex_c = len(op_fmtd) - 1
        s_c = op_fmtd
        s_r = self._format_level(lvl1)
        return Components.create("", (s_c, ntex_c), s_r)


class LevelRangeFormatterAnd(LevelRangeFormatter):
    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
    ) -> None:
        if widths is None:
            widths = (8, 3, 8)
        super().__init__(
            widths=widths, extend=extend, align=align, rstrip_zeros=rstrip_zeros,
        )

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:

        op0_fmtd = r"$\tt \geq$"
        lvl0_fmtd = f"{op0_fmtd} {self._format_level(lvl0)}"
        ntex_l = len(op0_fmtd) - 1

        op_fmtd = r"$\tt &$"
        ntex_c = len(op_fmtd) - 1

        op1_fmtd = r"$<$ "
        lvl1_fmtd = op1_fmtd + self._format_level(lvl1)
        ntex_r = len(op1_fmtd) - 1

        s_l = lvl0_fmtd
        s_c = op_fmtd
        s_r = lvl1_fmtd
        return Components.create((s_l, ntex_l), (s_c, ntex_c), (s_r, ntex_r))

    def _format_open_left(self, lvl: float) -> Components:
        op_fmtd = r"\tt $<$"
        lvl_fmtd = f"{op_fmtd} {self._format_level(lvl)}"
        ntex_r = len(op_fmtd) - 1
        s_r = lvl_fmtd
        return Components.create("", "", (s_r, ntex_r))

    def _format_open_right(self, lvl: float) -> Components:
        op0_fmtd = r"$\tt \geq$"
        lvl_fmtd = f"{op0_fmtd} {self._format_level(lvl)}"
        ntex_l = len(op0_fmtd) - 1
        s_l = lvl_fmtd
        return Components.create((s_l, ntex_l), "", "")


class LevelRangeFormatterVar(LevelRangeFormatter):
    def __init__(
        self,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: Optional[str] = None,
        align: Optional[str] = None,
        rstrip_zeros: Optional[bool] = None,
        var: Optional[str] = None,
    ):
        if widths is None:
            widths = (5, 9, 5)
        super().__init__(
            widths=widths, extend=extend, align=align, rstrip_zeros=rstrip_zeros,
        )
        if var is None:
            var = "v"
        self.var: str = var

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:
        op0 = r"$\tt \leq$"
        op1 = r"$\tt <$"
        op_fmtd = f"{op0} {self.var} {op1}"
        ntex_c = len(op0) + len(op1) - 2
        s_l = self._format_level(lvl0)
        s_c = op_fmtd
        s_r = self._format_level(lvl1)
        return Components.create(s_l, (s_c, ntex_c), s_r)

    def _format_open_right(self, lvl: float) -> Components:
        op0 = r"$\tt \leq$"
        op1 = ""
        op_fmtd = f"{op0} {self.var} {op1}"
        ntex_c = len(op0) + len(op1) - 2
        s_l = self._format_level(lvl)
        s_c = op_fmtd
        return Components.create(s_l, (s_c, ntex_c), "")

    def _format_open_left(self, lvl: float) -> Components:
        op0 = " "
        op1 = "$\tt <$"
        op_fmtd = f"{op0} {self.var} {op1}"
        ntex_c = len(op0) + len(op1) - 2
        s_c = op_fmtd
        s_r = self._format_level(lvl)
        return Components.create("", (s_c, ntex_c), s_r)
