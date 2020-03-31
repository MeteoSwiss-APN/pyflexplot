# -*- coding: utf-8 -*-
"""
Some utilities.
"""
# Standard library
import re
import warnings
from dataclasses import dataclass
from dataclasses import is_dataclass
from functools import partial
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
from pydantic import BaseModel

# First-party
from srutils.iter import isiterable


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""


class NotSummarizableError(Exception):
    """Object could not be summarized."""


class AttributeConflictError(Exception):
    """Conflicting object attributes."""


def is_attrs_class(cls: Any) -> bool:
    """Determine whether a class has been defined with ``@attr.attrs``."""
    return isinstance(cls, type) and hasattr(cls, "__attrs_attrs__")


def default_summarize(
    self: Any,
    addl: Optional[Collection[str]] = None,
    skip: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """Default summarize method; see docstring of ``summarizable``.

    Args:
        self: The class instance to be summarized.

        addl: Additional attributes to be summarized. Added to those specified
            in ``self.summarizable_attrs``.

        skip: Attributes not to be summarized despite being specified in
            ``self.summarizable_attrs``.

    Return:
        Summary dict.

    """
    return Summarizer().run(self, addl=addl, skip=skip)


def default_post_summarize(self: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Default post_summarize method; see docstring of ``summarizable``.

    Args:
        self: The class instance to be summarized.

        summary: Summary dict to be modified.

    Return:
        Modified summary dict.

    """
    return summary


def summarizable(
    cls: Optional[Callable] = None,
    *,
    attrs: Optional[Collection[str]] = None,
    attrs_skip: Optional[Collection[str]] = None,
    summarize: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
    post_summarize: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
    auto_collect: bool = True,
    overwrite: bool = False,
) -> Callable:
    """Decorator to make a class summarizable.

    Args:
        cls: Class to be decorated.

        attrs: Class attributes to summarize. Added to ``cls`` as class
            attribute ``summarizable_attrs``, which also contains all auto-
            collected attributes of special classes like dataclasse (unless
            ``auto_collect`` is False), but none specified in ``attrs_skip``.

        attrs_skip: Class attributes not to summarize. Affects attributes in
            ``attrs`` and, more importantly, auto-collected ones such as
            dataclass attributes.

        summarize: Custom function to summarize the class. Returns a dict
            containing the summarized attributes, which is then used to update
            the existing summary dict that has been created based on ``attrs``.
            Replaces ``default_summarize``. Added to ``cls`` as method
            ``summarize``.

        post_summarize: Custom function to post-process the summary dict.
            Replaces ``default_post_summarize``. Added to ``cls`` as method
            ``post_summarize``.

        auto_collect: Auto-collect attributes of certain types of classes, such
            as data classes and dict-convertible classes, in addition to those
            specified in attrs (if given at all).

        overwrite: Overwrite existing class attributes and/or methods. Must be
            True to make classes summarizable that inherit from summarizable
            parent classes.

    """
    if cls is None:
        return partial(
            summarizable,
            attrs=attrs,
            attrs_skip=attrs_skip,
            summarize=summarize,
            post_summarize=post_summarize,
            auto_collect=auto_collect,
            overwrite=overwrite,
        )

    try:
        attrs = [] if attrs is None else [a for a in attrs]
    except TypeError:
        raise ValueError("`attrs` is not iterable", type(attrs), attrs)
    try:
        attrs_skip = [] if attrs_skip is None else [a for a in attrs_skip]
    except TypeError:
        raise ValueError("`attrs_skip` is not iterable", type(attrs_skip), attrs_skip)
    if summarize is None:
        summarize = default_summarize
    if post_summarize is None:
        post_summarize = default_post_summarize

    if auto_collect:
        if is_attrs_class(cls):
            # Collect attributes defined with ``attr.attrib``
            attrs = [a.name for a in cls.__attrs_attrs__] + attrs  # type: ignore
        elif is_dataclass(cls):
            # Collect dataclass fields
            attrs = [f for f in cls.__dataclass_fields__] + attrs  # type: ignore
        elif issubclass(cls, BaseModel):  # type: ignore
            # Collect model fields
            attrs = [f for f in cls.__fields__] + attrs  # type: ignore
    attrs = [a for a in attrs if a not in attrs_skip]

    # Extend class
    for name, attr in [
        ("summarizable_attrs", attrs),
        ("summarize", summarize),
        ("post_summarize", post_summarize),
    ]:
        if not overwrite and hasattr(cls, name):
            raise AttributeConflictError(name, cls)
        setattr(cls, name, attr)
    return cls


class Summarizer:
    """Collect specified attributes of an object in a dict.

    Subclasses must define the property ``summarizable_attrs``, comprising
    a list of attribute names to be collected.

    If attribute values possess a ``summarize`` method themselves, the output
    of that is collected. Otherwise, it is attemted to convert the values to
    common types like dicts or lists. If all attempts fail, the raw value is
    added to the summary dict.

    """

    def run(
        self,
        obj: Any,
        *,
        addl: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
    ) -> Dict[str, Any]:
        """Summarize specified attributes of ``obj`` in a dict.

        The attributes to be summarized must be specified by name in the
        attribute ``obj.summarizable_attrs``.

        Args:
            obj: Object to summarize.

            addl (optional): Additional attributes to be collected.

            skip (optional): Attributes to skip during collection.

        Returns:
            Dictionary containing the collected attributes and their values.

        """
        data: Dict[str, Any] = {}

        if skip is None or "type" not in skip:
            data["type"] = type(obj).__name__

        attrs = list(obj.summarizable_attrs)

        if addl is not None:
            attrs += [a for a in addl if a not in attrs]

        if skip is not None:
            attrs = [a for a in attrs if a not in skip]

        for attr in attrs:
            data[attr] = self._summarize(getattr(obj, attr))

        return obj.post_summarize(data)

    def _summarize(self, obj: Any) -> Union[Dict[str, Any], Any]:
        """Try to summarize the object in various ways."""
        methods = [
            self._try_summarizable,
            self._try_dict_like,
            self._try_list_like,
            self._try_named,
        ]
        for method in methods:
            try:
                return method(obj)
            except NotSummarizableError:
                continue
        return obj

    def _try_summarizable(self, obj: Any) -> Dict[str, Any]:
        """Try to summarize ``obj`` as a summarizable object."""
        try:
            data = obj.summarize()
        except AttributeError:
            raise NotSummarizableError("summarizable", obj)
        else:
            return self._summarize(data)

    def _try_dict_like(self, obj: Any) -> Dict[Any, Any]:
        """Try to summarize ``obj`` as a dict-like object."""
        try:
            items = obj.items()
        except AttributeError:
            try:
                obj = dict(obj)
            except (TypeError, ValueError):
                raise NotSummarizableError("dict-like", obj)
            else:
                items = obj.items()
        return {self._summarize(key): self._summarize(val) for key, val in items}

    def _try_list_like(self, obj: Any) -> Sequence[Any]:
        """Try to summarize ``obj`` as a list-like object."""
        if not isiterable(obj, str_ok=False):
            raise NotSummarizableError("list-like", obj)
        type_ = type(obj)
        data = []
        for item in obj:
            data.append(self._summarize(item))
        if isinstance(obj, np.ndarray):
            return data
        return type_(data)

    def _try_named(self, obj: Any) -> str:
        """Try to summarize ``obj`` as a named object (e.g., function/method)."""
        try:
            name = obj.__name__
        except AttributeError:
            pass
        else:
            try:
                obj_self = obj.__self__
            except AttributeError:
                pass
            else:
                name = f"{obj_self.__class__.__name__}.{name}"
            return f"{type(obj).__name__}:{name}"
        raise NotSummarizableError("named", obj)


def check_float_ok(f: float, ff0t: str) -> bool:
    if f != np.inf and f >= 1.0:
        return bool(re.match(f"^{int(f)}" + r"\.[0-9]+$", ff0t))
    return (f == 0.0) or (float(ff0t) != 0.0)


def fmt_float(
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


@dataclass
class Component:
    """Auxiliary class to pass results between formatter methods."""

    s: str
    ntex: int

    @classmethod
    def create(cls, name: str, arg: Union[str, Tuple[str, int]]) -> "Component":
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
            left=Component.create("left", left),
            center=Component.create("center", center),
            right=Component.create("right", right),
        )


class LevelRangeFormatter:
    """Format level ranges, e.g., for legends of color contour plots."""

    def __init__(
        self,
        style: str,
        *,
        widths: Optional[Tuple[int, int, int]] = None,
        extend: str = "none",
        align: str = "center",
        rstrip_zeros: bool = False,
    ) -> None:
        """Create an instance of ``LevelRangeFormatter``.

        Args: See ``format_level_ranges``.

        """
        self.widths = widths or (5, 3, 5)
        self.extend = extend
        self.align = align
        self.rstrip_zeros = rstrip_zeros

        self._check_widths(self.widths)

    def _check_widths(self, widths: Tuple[int, int, int]) -> None:
        try:
            wl, wc, wr = [int(w) for w in widths]
        except (ValueError, TypeError):
            raise ValueError(f"widths is not a tree-int tuple: {widths}")

    def fmt_multiple(self, levels: Sequence[float]) -> List[str]:
        ss: List[str] = []
        if self.extend in ("min", "both"):
            ss.append(self.format(None, levels[0]))
        for lvl0, lvl1 in zip(levels[:-1], levels[1:]):
            ss.append(self.format(lvl0, lvl1))
        if self.extend in ("max", "both"):
            ss.append(self.format(levels[-1], None))
        return ss

    def format(self, lvl0: Optional[float], lvl1: Optional[float]) -> str:
        cs = self._format_components(lvl0, lvl1)

        s_l = cs.left.s
        s_c = cs.center.s
        s_r = cs.right.s

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
            raise ValueError(f"range open at both ends")
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
        return fmt_float(lvl, "{f:.0E}")


class LevelRangeFormatter_Int(LevelRangeFormatter):
    def __init__(
        self, *args, widths: Optional[Tuple[int, int, int]] = None, **kwargs,
    ) -> None:
        if widths is None:
            widths = (2, 3, 2)
        if kwargs.get("rstrip_zeros"):
            warnings.warn(f"{type(self).__name__}: force rstrip_zeros=False")
        kwargs["rstrip_zeros"] = False
        super().__init__(*args, widths=widths, **kwargs)

    def _format_components(
        self, lvl0: Optional[float], lvl1: Optional[float],
    ) -> Components:
        if lvl1 is not None:
            lvl1 = lvl1 - 1
            if lvl0 == lvl1:
                return Components.create("", "", self._format_level(lvl1))
        return super()._format_components(lvl0, lvl1)

    def _format_level(self, lvl: float) -> str:
        if int(lvl) != float(lvl):
            warnings.warn(f"{type(self).__name__}._format_level: not an int: {lvl}")
        return str(lvl)


class LevelRangeFormatter_Math(LevelRangeFormatter):
    def __init__(
        self, *args, widths: Optional[Tuple[int, int, int]] = None, **kwargs,
    ) -> None:
        if widths is None:
            widths = (6, 2, 6)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_components(
        self, lvl0: Optional[float], lvl1: Optional[float],
    ) -> Components:
        return Components.create(
            "-inf" if lvl0 is None else f"[{self._format_level(lvl0)}",
            ",",
            "inf" if lvl1 is None else f"{self._format_level(lvl1)})",
        )


class LevelRangeFormatter_Up(LevelRangeFormatter):
    def __init__(
        self, *args, widths: Optional[Tuple[int, int, int]] = None, **kwargs,
    ) -> None:
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:
        op_fmtd = r"$\tt \geq$"
        ntex_c = len(op_fmtd) - 1
        s_c = op_fmtd
        s_r = self._format_level(lvl0)
        return Components.create("", (s_c, ntex_c), s_r)


class LevelRangeFormatter_Down(LevelRangeFormatter):
    def __init__(
        self, *args, widths: Optional[Tuple[int, int, int]] = None, **kwargs,
    ):
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0: float, lvl1: float) -> Components:
        op_fmtd = r"$\tt <$"
        ntex_c = len(op_fmtd) - 1
        s_c = op_fmtd
        s_r = self._format_level(lvl1)
        return Components.create("", (s_c, ntex_c), s_r)


class LevelRangeFormatter_And(LevelRangeFormatter):
    def __init__(
        self, *args, widths: Optional[Tuple[int, int, int]] = None, **kwargs,
    ) -> None:
        if widths is None:
            widths = (8, 3, 8)
        super().__init__(*args, widths=widths, **kwargs)

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


class LevelRangeFormatter_Var(LevelRangeFormatter):
    def __init__(
        self,
        *args,
        widths: Optional[Tuple[int, int, int]] = None,
        var: str = "v",
        **kwargs,
    ):
        if widths is None:
            widths = (5, 9, 5)
        super().__init__(*args, widths=widths, **kwargs)
        self.var = var

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
