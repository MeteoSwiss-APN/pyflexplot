# -*- coding: utf-8 -*-
"""
Utils for the command line tool.
"""
import logging as log
import numpy as np
import re

from collections import namedtuple

from srutils.various import isiterable


# Exceptions


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""

    pass


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""

    pass


class ParentClass:
    """Enable easy access to subclasses via their 'name' attribute."""

    @classmethod
    def subclass(cls, name):
        """Find a subclass by name."""
        subclasses_by_name = cls.subclasses_by_name()
        try:
            return subclasses_by_name[name]
        except KeyError:
            raise ValueError(
                f"class '{cls.__name__}' has no subclass '{name}'; options: "
                f"{sorted(subclasses_by_name)}"
            )

    @classmethod
    def subclasses_by_name(cls, unique=True):
        """Recursively collect named subclasses by 'name' attribute."""
        result = {}
        for sub_cls in cls.__subclasses__():
            try:
                name = sub_cls.name
            except AttributeError:
                continue
            if name is None:
                continue
            if unique and result.get(name) is sub_cls:
                raise Exception(f"duplicate class name '{name}'")
            result[name] = sub_cls
            result.update(sub_cls.subclasses_by_name())
        return result


class SummarizableClass:
    """Summarize important class attributes into a dict."""

    def __init__(self, *args, **kwargs):
        raise Exception(f"{type(self).__name__} must be subclassed")

    @property
    def summarizable_attrs(self):
        raise Exception(
            f"`summarizable_attrs` must be an attribute of subclasses of "
            f"{type(self).__name__}"
        )

    def summarize(self, *, add=None, skip=None):
        """Collect all attributes in ``summarizable_attrs`` in a dict.

        Subclasses must define the property ``summarizable_attrs``, comprising
        a list of attribute names to be collected.

        If attribute values possess a ``summarize`` method themselves, the
        output of that is collected, otherwise the direct values.

        Args:
            add (list, optional): Additional attributes to be collected.
                Defaults to None.

            skip (list, optional): Attributes to skip during collection.
                Defaults to None.

        Returns:
            dict: Dictionary containing the collected attributes and their
                values.

        """
        data = {}
        if skip is None or "type" not in skip:
            data["type"] = type(self).__name__
        attrs = list(self.summarizable_attrs)
        if add is not None:
            attrs += [a for a in add if a not in attrs]
        if skip is not None:
            attrs = [a for a in attrs if a not in skip]
        for attr in attrs:
            data[attr] = self._summarize_obj(getattr(self, attr))
        return data

    def _summarize_obj(self, obj):

        # Summarizable object?
        try:
            data = obj.summarize()
        except AttributeError:
            pass
        else:
            return self._summarize_obj(data)

        # Dict-like object?
        try:
            items = obj.items()
        except AttributeError:
            pass
        else:
            for key, val in items:
                obj[key] = self._summarize_obj(val)
            return obj

        # List-like object?
        if isiterable(obj, str_ok=False):
            type_ = type(obj)
            data = []
            for item in obj:
                data.append(self._summarize_obj(item))
            return type_(data)

        # Giving up!
        return obj


def count_to_log_level(count: int) -> int:
    """Map the occurence of the command line option verbose to the log level"""
    if count == 0:
        return log.ERROR
    elif count == 1:
        return log.WARNING
    elif count == 2:
        return log.INFO
    else:
        return log.DEBUG


def fmt_float(f, fmt_e0=None, fmt_f0=None, fmt_e1=None, fmt_f1=None):
    """Auto-format a float to floating-point or exponential notation.

    Very small and very large numbers are formatted in exponential notation.
    Numbers close enough to zero that they can be written in floating point
    format without exceeding the width of the exponential format are written in
    the former.

    Args:
        f (float): Number to format.

        fmt_e0 (str, optional): Exponential-notation format string used to
            create the string that is compared to that produced with ``fmt_f0``
            to decide the appropriate notation. Defaults to '{f:e}'.

        fmt_f0 (str, optional): Floating-point notation format string used to
            create the string that is compared to that produced with ``fmt_e0``
            to decide the appropriate notation. Defaults to '{f:f}'.

        fmt_e1 (str, optional): Exponential-notation format string used to
            create the return string if exponential notation has been found to
            be appropriate. Defaults to ``fmt_e0``.

        fmt_f1 (str, optional): Floating-point notation format string used to
            create the return string if floating point notation has been found
            to be appropriate. Defaults to '{f:f}' with the resulting string
            trimmed to the length of that produced with ``fmt_e0``.

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
    try:
        float(f)
    except (ValueError, TypeError):
        raise ValueError(f"f='{f}' of type {type(f).__name__} not float-compatible")

    rx_e = re.compile(r"^{f:[0-9,]*\.?[0-9]*[eE]}$")
    rx_f = re.compile(r"^{f:[0-9,]*\.?[0-9]*f}$")
    for name in ["fmt_e0", "fmt_f0", "fmt_e1", "fmt_f1"]:
        fmt = locals()[name]
        if fmt is not None:
            if not rx_e.match(fmt) and not rx_f.match(fmt):
                raise ValueError(f"invalid format string: {name}='{fmt}'")

    if fmt_e0 is None:
        fmt_e0 = "{f:e}"
    if fmt_f0 is None:
        fmt_f0 = "{f:f}"

    fe0 = fmt_e0.format(f=f)
    ff0 = fmt_f0.format(f=f)
    n = len(fe0)
    ff0t = ff0[:n]

    if f != np.inf and f >= 1.0:
        rxs = r"^" + str(int(f)) + r"\.[0-9]+$"
        float_ok = bool(re.match(rxs, ff0t))
    else:
        float_ok = (f == 0.0) or (float(ff0t) != 0.0)

    if float_ok:
        if fmt_f1 is not None:
            return fmt_f1.format(f=f)
        return ff0t
    if fmt_e1 is not None:
        return fmt_e1.format(f=f)
    return fe0


def format_level_ranges(
    levels, style=None, widths=None, extend=None, align=None, **kwargs
):
    """Format a list of level ranges in a certain style.

    Args:
        levels (list[float]): Levels between the ranges.

        style (str, optional): Formatting style (options and examples below).
            Defaults to 'base'.

        widths (tuple[int, int, int], optional): Tuple with the minimum
            character widths of, respectively, the left ('lower-than'), center
            (operator), and right ('greater-than') parts of the ranges.
            Defaults to style-specific values.

        extend (str, optional): Whether the range is closed ('none'), open at
            the lower ('min') or the upper ('max') end, or both ('both'). Same
            as the ``extend`` keyword of, e.g., ``matplotlib.pyplot.contourf``.
            Defaults to 'none'.

        align (str, optional): Horizontal alignment of the left and right
            components (the center component with the operator is always
            center-aligned). Options: 'left' (both components left-aligned),
            'right' (both components right-aligned), 'center', (left/right
            component right/left-aligned), and 'edges' (left/right component
            left/right-aligned). Defaults to 'center'.

        **kwargs: Additional style-specific keyword arguments used to
            initialize the respective formatter class. See individual formatter
            classes for details.

    Returns:
        list[str]: List of formatted level range strings, each of which
            represents the range between two successive ``levels``. Depending
            on ``extend``, the number of strings is equal to ('min' or 'max'),
            one smaller ('none'), or one greater ('both') than the number of
            ``levels``.

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
        "int": LevelRangeFormatter_Int,
        "math": LevelRangeFormatter_Math,
        "up": LevelRangeFormatter_Up,
        "down": LevelRangeFormatter_Down,
        "and": LevelRangeFormatter_And,
        "var": LevelRangeFormatter_Var,
    }
    try:
        cls = formatters[style]
    except AttributeError:
        raise ValueError(f"unknown style '{style}'; options: {sorted(formatters)}")
    else:
        formatter = cls(
            style=style, widths=widths, extend=extend, align=align, **kwargs
        )
    return formatter.fmt_multiple(levels)


class LevelRangeFormatter:
    """Format level ranges, e.g., for legends of color contour plots."""

    def __init__(
        self, style, widths=None, extend="none", align="center", rstrip_zeros=False
    ):
        """Create an instance of ``LevelRangeFormatter``.

        Args: See ``format_level_ranges``.

        """
        if widths is None:
            widths = (5, 3, 5)
        else:
            self._check_widths(widths)
        self.widths = widths
        self.extend = extend
        self.align = align
        self.rstrip_zeros = rstrip_zeros

    def _check_widths(self, widths):
        try:
            wl, wc, wr = [int(w) for w in widths]
        except (ValueError, TypeError):
            raise ValueError(f"widths is not a tree-int tuple: {widths}")

    def fmt_multiple(self, levels):
        ss = []
        if self.extend in ("min", "both"):
            ss.append(self.format(None, levels[0]))
        for lvl0, lvl1 in zip(levels[:-1], levels[1:]):
            ss.append(self.format(lvl0, lvl1))
        if self.extend in ("max", "both"):
            ss.append(self.format(levels[-1], None))
        return ss

    def format(self, lvl0, lvl1):
        cs = self._format_components(lvl0, lvl1)

        s_l = cs.l.s
        s_c = cs.c.s
        s_r = cs.r.s

        dc = "^"
        dl, dr = dict(left="<<", right=">>", center="><", edges="<>")[self.align]

        if self.rstrip_zeros:

            def rstrip_zeros(s):
                rx_str = r"(?<!\.)0\b(?!\.)"
                while re.search(rx_str, s):
                    s = re.sub(rx_str, "", s)
                return s

            s_l = rstrip_zeros(s_l)
            s_r = rstrip_zeros(s_r)

        wl, wc, wr = self.widths

        s_l = f"{{:{dl}{wl + cs.l.ntex}}}".format(s_l)
        s_c = f"{{:{dc}{wc + cs.c.ntex}}}".format(s_c)
        s_r = f"{{:{dr}{wr + cs.r.ntex}}}".format(s_r)

        return f"{s_l}{s_c}{s_r}"

    def _format_components(self, lvl0, lvl1):
        open_left = lvl0 in (None, np.inf)
        open_right = lvl1 in (None, np.inf)
        if open_left and open_right:
            raise ValueError(f"range open at both ends")
        elif open_left:
            return self._format_open_left(lvl1)
        elif open_right:
            return self._format_open_right(lvl0)
        else:
            return self._format_closed(lvl0, lvl1)

    def _format_closed(self, lvl0, lvl1):
        lvl0_fmtd = self._format_level(lvl0)
        lvl1_fmtd = self._format_level(lvl1)
        op_fmtd = r"$\tt -$"
        ntex_c = len(op_fmtd) - 1
        s_l = lvl0_fmtd
        s_c = op_fmtd
        s_r = lvl1_fmtd
        return self._Components(s_l, (s_c, ntex_c), s_r)

    def _format_open_left(self, lvl):
        return self._format_open_core(lvl, r"$\tt <$")

    def _format_open_right(self, lvl):
        return self._format_open_core(lvl, r"$\tt \geq$")

    def _format_open_core(self, lvl, op, *, len_op=1):
        lvl_fmtd = self._format_level(lvl)
        ntex_c = len(op) - len_op
        s_c = op
        s_r = lvl_fmtd
        return self._Components("", (s_c, ntex_c), s_r)

    def _format_level(self, lvl):
        return fmt_float(lvl, "{f:.0E}")

    @classmethod
    def _Component(cls, s, n=None):
        """Auxiliary type for passing results between methods."""
        if n is None:
            n = len(s)
        Component = namedtuple("component", "s ntex")
        return Component(s, n)

    @classmethod
    def _Components(cls, left, center, right):
        """Auxiliary type for passing results between methods."""
        Components = namedtuple("components", "l c r")

        def create_component(name, arg):
            if isinstance(arg, str):
                return cls._Component(arg, 0)
            try:
                return cls._Component(*arg)
            except Exception as e:
                raise ValueError(
                    f"cannot create {name} Component "
                    f"from {type(arg).__name__} {arg}"
                )

        return Components(
            create_component("left", left),
            create_component("center", center),
            create_component("right", right),
        )


class LevelRangeFormatter_Int(LevelRangeFormatter):
    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (2, 3, 2)
        if kwargs.get("rstrip_zeros"):
            log.warning(f"{type(self).__name__}: force rstrip_zeros=False")
        kwargs["rstrip_zeros"] = False
        super().__init__(*args, widths=widths, **kwargs)

    def _format_components(self, lvl0, lvl1):
        if lvl1 is not None:
            lvl1 = lvl1 - 1
            if lvl0 == lvl1:
                return self._Components("", "", self._format_level(lvl1))
        return super()._format_components(lvl0, lvl1)

    def _format_level(self, lvl):
        if int(lvl) != float(lvl):
            log.warning(f"{type(self).__name__}._format_level: not an int: {lvl}")
        return str(lvl)


class LevelRangeFormatter_Math(LevelRangeFormatter):
    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (6, 2, 6)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_components(self, lvl0, lvl1):
        return self._Components(
            "-inf" if lvl0 is None else f"[{self._format_level(lvl0)}",
            ",",
            "inf" if lvl1 is None else f"{self._format_level(lvl1)})",
        )


class LevelRangeFormatter_Up(LevelRangeFormatter):
    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0, lvl1):
        lvl0_fmtd = self._format_level(lvl0)
        op_fmtd = r"$\tt \geq$"
        ntex_c = len(op_fmtd) - 1
        s_c = op_fmtd
        s_r = self._format_level(lvl0)
        return self._Components("", (s_c, ntex_c), s_r)


class LevelRangeFormatter_Down(LevelRangeFormatter):
    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0, lvl1):
        op_fmtd = r"$\tt <$"
        ntex_c = len(op_fmtd) - 1
        s_c = op_fmtd
        s_r = self._format_level(lvl1)
        return self._Components("", (s_c, ntex_c), s_r)


class LevelRangeFormatter_And(LevelRangeFormatter):
    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (8, 3, 8)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0, lvl1):

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
        return self._Components((s_l, ntex_l), (s_c, ntex_c), (s_r, ntex_r))

    def _format_open_left(self, lvl):
        op1_fmtd = r"\tt $<$"
        lvl_fmtd = f"{op1_fmtd} {self._format_level(lvl)}"
        ntex_r = len(op0_fmtd) - 1
        s_r = lvl1_fmtd
        return self._Components("", "", (s_r, ntex_r))

    def _format_open_right(self, lvl):
        op0_fmtd = r"$\tt \geq$"
        lvl_fmtd = f"{op0_fmtd} {self._format_level(lvl)}"
        ntex_l = len(op0_fmtd) - 1
        s_l = lvl_fmtd
        return self._Components((s_l, ntex_l), "", "")


class LevelRangeFormatter_Var(LevelRangeFormatter):
    def __init__(self, *args, widths=None, var="v", **kwargs):
        if widths is None:
            widths = (5, 9, 5)
        super().__init__(*args, widths=widths, **kwargs)
        self.var = var

    def _format_closed(self, lvl0, lvl1):
        op0 = r"$\tt \leq$"
        op1 = r"$\tt <$"
        op_fmtd = f"{op0} {self.var} {op1}"
        ntex_c = len(op0) + len(op1) - 2
        s_l = self._format_level(lvl0)
        s_c = op_fmtd
        s_r = self._format_level(lvl1)
        return self._Components(s_l, (s_c, ntex_c), s_r)

    def _format_open_right(self, lvl):
        op0 = r"$\tt \leq$"
        op1 = ""
        op_fmtd = f"{op0} {self.var} {op1}"
        ntex_c = len(op0) + len(op1) - 2
        s_l = self._format_level(lvl)
        s_c = op_fmtd
        return self._Components(s_l, (s_c, ntex_c), "")

    def _format_open_left(self, lvl):
        lvl_fmtd = self._format_level(lvl)
        op0 = " "
        op1 = "$\tt <$"
        op_fmtd = f"{op0} {self.var} {op1}"
        ntex_c = len(op0) + len(op1) - 2
        s_c = op_fmtd
        s_r = self._format_level(lvl)
        return self._Commponents("", (s_c, ntex_c), s_r)
