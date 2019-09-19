# -*- coding: utf-8 -*-
"""
Utils for the command line tool.
"""
import functools
import itertools
import logging as log
import numpy as np
import re

from pprint import pformat

from .utils_dev import ipython  #SR_DEV

#======================================================================
# Exceptions
#======================================================================


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""
    pass


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""
    pass


#======================================================================


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
                f"class '{cls.__name__}' has no subclass '{name}'; "
                f"options: {sorted(subclasses_by_name)}")

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
            f"{type(self).__name__}")

    def summarize(self, *, add=None, skip=None):
        """Collect all attributes in ``summarizable_attrs`` in a dict.

        Subclasses must define the property ``summarizable_attrs``,
        comprising a list of attribute names to be collected.

        If attribute values possess a ``summarize`` method themselves,
        the output of that is collected, otherwise the direct values.

        Args:
            add (list, optional): Additional attributes to be collected.
                Defaults to None.

            skip (list, optional): Attributes to skip during collection.
                Defaults to None.

        Returns:
            dict: Dictionary containing the collected attributes and
                their values.

        """
        data = {}
        if skip is None or 'type' not in skip:
            data['type'] = type(self).__name__
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


#======================================================================


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
            return

        # Check for most common non-stable sequence
        if isinstance(deg, set):
            raise ValueError(f"deg cannot be a set")

        # Check for `(degs,)`
        try:
            degs, = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0])
            return

        # Check for `(degs, mins)`
        try:
            degs, mins = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0]) + deg[1]/60.0
            return

        # Check for `(degs, mins, secs)`
        try:
            degs, mins, secs = deg
        except TypeError:
            pass
        else:
            self._frac = float(deg[0]) + deg[1]/60.0 + deg[2]/3600.0

        raise ValueError(
            f"invalid deg='{deg}'; "
            f"must be `float` or `(degs, [mins, [secs, ]])`")

    def frac(self):
        """Return degrees as a fraction."""
        return self._frac

    def dms(self):
        """Return full degrees, minutes, and seconds (int)."""
        degs = self._frac
        mins = degs%1*60
        secs = mins%1*60
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


#======================================================================


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


def merge_dicts(*dicts, unique_keys=True):
    """Merge multiple dictionaries with or without shared keys.

    Args:
        dicts (list[dict]) Dictionaries to be merged.

        unique_keys (bool, optional): Whether keys must be unique.
            If True, duplicate keys raise a ``KeyConflictError``
            exception. If False, dicts take precedence in reverse order
            of ``dicts``, i.e., keys occurring in multiple dicts will
            have the value of the last dict containing that key.

    Raises:
        KeyConflictError: If ``unique_keys=True`` when a key occurs
            in multiple dicts.

    Returns:
        dict: Merged dictionary.

    """
    merged = {}
    for dict_ in dicts:
        if not unique_keys:
            merged.update(dict_)
        else:
            for key, val in dict_.items():
                if key in merged:
                    raise KeyConflictError(key)
                merged[key] = val
    return merged


def dict_mult_vals_product(dict_):
    """Combine value lists in a dict in all possible ways.

    Example:
        in: {'foo_lst': [1, 2, 3], 'bar': 4, 'baz_lst': [5, 6]}

        out: [
                {'foo': 1, 'bar: 4, 'baz': 5},
                {'foo': 2, 'bar: 4, 'baz': 5},
                {'foo': 3, 'bar: 4, 'baz': 5},
                {'foo': 1, 'bar: 4, 'baz': 6},
                {'foo': 2, 'bar: 4, 'baz': 6},
                {'foo': 3, 'bar: 4, 'baz': 6},
            ]

    TODOs:
        Improve wording of short description of docstring.

    """
    keys, vals = [], []
    for key, val in sorted(dict_.items()):
        keys.append(key.replace('_lst', ''))
        vals.append(val if key.endswith('_lst') else [val])
    return [dict(zip(keys, vals_i)) for vals_i in itertools.product(*vals)]


#======================================================================


def check_array_indices(shape, inds):
    """Check that slicing indices are consistent with array shape."""

    def inds2str(inds):
        """Convert indices to string, turning slices into '::' etc."""
        strs = []
        for ind in inds:
            if isinstance(ind, slice):
                strs.append(
                    f"{'' if ind.start is None else ind.start}:"
                    f"{'' if ind.stop is None else ind.stop}:"
                    f"{'' if ind.step is None else ind.step}")
            else:
                strs.append(str(ind))
        return f"[{', '.join(strs)}]"

    if len(inds) > len(shape):
        inds_str = inds2str(inds)
        raise IndexError(f"too many indices for shape {shape}: {inds_str}")

    for j, (n, i) in enumerate(zip(shape, inds)):
        if isinstance(i, slice):
            continue
        elif i >= n:
            e = f'{i} >= {n}'
        elif i < -n:
            e = f'{i} < -{n}'
        else:
            continue
        inds_str = inds2str(inds)
        raise IndexError(
            f"index {j} of {inds_str} out of bounds for shape {shape}: {e}")


def pformat_dictlike(obj):
    """Pretty-format a dict-like object."""

    s = f'{obj.__class__.__name__}({pformat(dict(obj))})'

    # Insert line breaks between braces and content
    s = s.replace("({'", "({\n '").replace('})', ',\n)}')

    # Increase indent
    #s = s.replace('\n ', '\n  ')

    return s


def nested_dict_set(dct, keys, val):
    """Set a value in a nested dict, creating subdicts if necessary.

    Args:
        dct (dict): Dictionary.

        keys (list[<key>]): List of keys.

        val (object): Value.

    """
    parent = dct
    n = len(keys)
    for i, key in enumerate(keys):
        if i < n - 1:
            if key not in parent:
                parent[key] = {}
            parent = parent[key]
        else:
            parent[key] = val


def isiterable(obj, str_ok=True):
    """Check whether an object is iterable.

    Args:
        obj (object): Object to check.

        str_ok (bool, optional): Whether strings are considered
            iterable. Defaults to True.

    """
    if isinstance(obj, str):
        return str_ok
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def group_kwargs(name, name_out=None, separator=None):
    """Collect all keyword arguments whose name starts with a prefix.

    All keyword arguments '<name>__foo', '<name>__bar', etc. are
    collected and put in a dictionary as 'foo', 'bar', etc., which
    is passed on as a keyword argument '<name>'.

    Args:
        name (str): Name of the group. Constitutes the prefix of the
            arguments to be collected, separated from the latter by
            ``separator``.

        name_out (str, optional): Name of the dictionary which the
            collected arguments are grouped into. Defaults to ``name``.

        separator (str, optional): Separator between the prefixed
            ``name`` and the argument. Defaults to '__'.

    Usage example:

        @collect_kwargs('test', 'kwargs_test')
        def test(arg1, kwargs_test):
            print(kwargs_test)

        test(arg1='foo', test__arg2='bar', test__arg3='baz')

        > {'arg2': 'bar', 'arg3': 'baz'}

    """
    if name_out is None:
        name_out = name
    if separator is None:
        separator = '__'
    prefix = f'{name}{separator}'
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if name_out in kwargs:
                raise ValueError(
                    f"keyword argument '{name_out}' already present")
            group = {}
            for key in [k for k in kwargs]:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    group[new_key] = kwargs.pop(key)
            kwargs[name_out] = group
            return f(*args, **kwargs)
        return wrapper
    return decorator


#======================================================================


def format_level_ranges(
        levels, style=None, widths=None, extend=None, **kwargs):
    """Format a list of level ranges in a certain style.

    Args:
        levels (list[float]): Levels between the ranges.

        style (str, optional): Formatting style (options and examples
            below). Defaults to 'base'.

        widths (tuple[int, int, int], optional): Tuple with the
            minimum character widths of, respectively, the left
            ('lower-than'), center (operator), and right
            ('greater-than') parts of the ranges. Defaults to style-
            specific values.

        extend (str, optional): Whether the range is closed ('none'),
            open at the lower ('min') or the upper ('max') end, or both
            ('both').  Same as the ``extend`` keyword of, e.g.,
            ``matplotlib.pyplot.contourf``. Defaults to 'none'.

        **kwargs: Additional style-specific keyword arguments used to
            initialize the respective formatter class.
            individual formatter classes for details.

    Returns:
        list[str]: List of formatted level range strings, each of which
            represents the range between two successive ``levels``.
            Depending on ``extend``, the number of strings is equal to
            ('min' or 'max'), one smaller ('none'), or one greater
            ('both') than the number of ``levels``.

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
        style = 'base'
    if extend is None:
        extend = 'none'
    formatters = {
        'base': LevelRangeFormatter,
        'int': LevelRangeFormatter_Int,
        'math': LevelRangeFormatter_Math,
        'up': LevelRangeFormatter_Up,
        'down': LevelRangeFormatter_Down,
        'and': LevelRangeFormatter_And,
        'var': LevelRangeFormatter_Var,
    }
    try:
        cls = formatters[style]
    except AttributeError:
        raise ValueError(
            f"unknown style '{style}'; options: {sorted(formatters)}")
    else:
        formatter = cls(style=style, widths=widths, extend=extend, **kwargs)
    return formatter.format_multiple(levels)


class LevelRangeFormatter:
    """Format level ranges, e.g., for legends of color contour plots."""

    def __init__(self, style, widths=None, extend='none', rstrip_zeros=True):
        self.style = style  #SR_TMP
        if widths is None:
            widths = (5, 3, 5)
        else:
            self._check_widths(widths)
        self.widths = widths
        self.extend = extend
        self.rstrip_zeros = rstrip_zeros

    def _check_widths(self, widths):
        try:
            wl, wc, wr = [int(w) for w in widths]
        except (ValueError, TypeError):
            raise ValueError(f"widths is not a tree-int tuple: {widths}")

    def format_multiple(self, levels):
        labels = []
        if self.extend in ('min', 'both'):
            labels.append(self.format(None, levels[0]))
        for lvl0, lvl1 in zip(levels[:-1], levels[1:]):
            labels.append(self.format(lvl0, lvl1))
        if self.extend in ('max', 'both'):
            labels.append(self.format(levels[-1], None))
        return labels

    def format(self, lvl0, lvl1):
        label = self._format_core(lvl0, lvl1)
        if self.rstrip_zeros:
            rx_str = r'(?<!\.)0\b(?!\.)'
            while re.search(rx_str, label):
                label = re.sub(rx_str, ' ', label)
        return label

    def _format_core(self, lvl0, lvl1):
        if (lvl0, lvl1) == (None, None):
            raise ValueError(f"both levels are None")
        elif lvl0 is None:
            return self._format_open_left(lvl1)
        elif lvl1 is None:
            return self._format_open_right(lvl0)
        else:
            return self._format_closed(lvl0, lvl1)

    def _format_closed(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        lvl0_fmtd = self._format_level(lvl0)
        lvl1_fmtd = self._format_level(lvl1)
        op_fmtd = r'$\tt -$'
        wc_fmtd = wc + len(op_fmtd) - 1
        label_l = f"{{:>{wl}}}".format(lvl0_fmtd)
        label_c = f"{{:^{wc_fmtd}}}".format(op_fmtd)
        label_r = f"{{:<{wr}}}".format(lvl1_fmtd)
        return f"{label_l}{label_c}{label_r}"

    def _format_open_left(self, lvl):
        return self._format_open_core(lvl, r'$\tt <$')

    def _format_open_right(self, lvl):
        return self._format_open_core(lvl, r'$\tt \geq$')

    def _format_open_core(self, lvl, op, *, len_op=1):
        lvl_fmtd = self._format_level(lvl)
        wl, wc, wr = self.widths
        wc_tex = wc + len(op) - len_op
        label_l = ' '*wl
        label_c = f"{{:^{wc_tex}}}".format(op)
        label_r = f"{{:<{wr}}}".format(lvl_fmtd)
        return f"{label_l}{label_c}{label_r}"

    def _format_level(self, lvl):
        if lvl is None:
            raise ValueError(f"lvl is None")
        fmtd = f'{lvl:.0E}'
        n = len(fmtd)
        ll = np.log10(lvl)
        if ll >= -n + 2 and ll <= n - 1:
            fmtd = f'{lvl:f}' [:n]
        return fmtd


class LevelRangeFormatter_Int(LevelRangeFormatter):

    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (2, 3, 2)
        if kwargs.get('rstrip_zeros'):
            log.warning(f"{type(self).__name__}: force rstrip_zeros=False")
        kwargs['rstrip_zeros'] = False
        super().__init__(*args, widths=widths, **kwargs)

    def _format_core(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        if lvl1 is not None:
            lvl1 = lvl1 - 1
            if lvl0 == lvl1:
                label_r = f"{{:<{wr}}}".format(self._format_level(lvl1))
                return f"{' '*(wl + wc)}{label_r}"
        return super()._format_core(lvl0, lvl1)

    def _format_level(self, lvl):
        if int(lvl) != float(lvl):
            log.warning(
                f"{type(self).__name__}._format_level: not an int: {lvl}")
        return int(lvl)


class LevelRangeFormatter_Math(LevelRangeFormatter):

    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (6, 2, 6)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_core(self, lvl0, lvl1):
        label_l = self._format_l(lvl0)
        label_c = self._format_c()
        label_r = self._format_r(lvl1)
        return f"{label_l}{label_c}{label_r}"

    def _format_l(self, lvl):
        wl, wc, wr = self.widths
        if lvl is not None:
            return f"[{{:>{wl}}}".format(self._format_level(lvl))
        return f"({{:>{wl}}}".format('-inf')

    def _format_r(self, lvl):
        wl, wc, wr = self.widths
        if lvl is not None:
            return f"{{:<{wr}}})".format(self._format_level(lvl))
        return f"{{:<{wr}}})".format('inf')

    def _format_c(self):
        wl, wc, wr = self.widths
        return f"{{:<{wc}}}".format(',')


class LevelRangeFormatter_Up(LevelRangeFormatter):

    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        lvl0_fmtd = self._format_level(lvl0)
        op_fmtd = r'$\tt \geq$'
        wc_tex = wc + len(op_fmtd) - 1
        label_l = ' '*wl
        label_c = f"{{:<{wc_tex}}}".format(op_fmtd)
        label_r = f"{{:<{wr}}}".format(self._format_level(lvl0))
        return f"{label_l}{label_c}{label_r}"


class LevelRangeFormatter_Down(LevelRangeFormatter):

    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (0, 2, 5)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        op_fmtd = r'$\tt <$'
        wc_tex = wc + len(op_fmtd) - 1
        label_l = ' '*wl
        label_c = f"{{:<{wc_tex}}}".format(op_fmtd)
        label_r = f"{{:<{wr}}}".format(self._format_level(lvl1))
        return f"{label_l}{label_c}{label_r}"


class LevelRangeFormatter_And(LevelRangeFormatter):

    def __init__(self, *args, widths=None, **kwargs):
        if widths is None:
            widths = (8, 3, 8)
        super().__init__(*args, widths=widths, **kwargs)

    def _format_closed(self, lvl0, lvl1):
        wl, wc, wr = self.widths

        op0_fmtd = r'$\tt \geq$'
        lvl0_fmtd = f"{op0_fmtd} {self._format_level(lvl0)}"
        wl_tex = wl + len(op0_fmtd) - 1

        op1_fmtd = r'$<$ '
        lvl1_fmtd = op1_fmtd + self._format_level(lvl1)
        wr_tex = wr + len(op1_fmtd) - 1

        op_fmtd = r'$\tt &$'
        wc_tex = wc + len(op_fmtd) - 1

        label_l = f"{{:<{wl_tex}}}".format(lvl0_fmtd)
        label_c = f"{{:^{wc_tex}}}".format(op_fmtc)
        label_r = f"{{:>{wr_tex}}}".format(lvl1_fmtd)
        return f"{label_l}{label_c}{label_r}"

    def _format_open_left(self, lvl):
        wl, wc, wr = self.widths
        op1_fmtd = r'\tt $<$'
        lvl_fmtd = f"{op1_fmtd} {self._format_level(lvl)}"
        wr_tex = wr + len(op0_fmtd) - 1
        label_l = ' '*wl
        label_c = ' '*wc
        label_r = f"{{:>{wr_tex}}}".format(lvl1_fmtd)
        return f"{label_l}{label_c}{label_r}"

    def _format_open_right(self, lvl):
        wl, wc, wr = self.widths
        op0_fmtd = r'$\tt \geq$'
        lvl_fmtd = f"{op0_fmtd} {self._format_level(lvl)}"
        wl_tex = wl + len(op0_fmtd) - 1
        label_l = f"{{:<{wl_tex}}}".format(lvl_fmtd)
        label_c = ' '*wc
        label_r = ' '*wr
        return f"{label_l}{label_c}{label_r}"


class LevelRangeFormatter_Var(LevelRangeFormatter):

    def __init__(self, *args, widths=None, var='v', **kwargs):
        if widths is None:
            widths = (5, 9, 5)
        super().__init__(*args, widths=widths, **kwargs)
        self.var = var

    def _format_closed(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        op0 = r'$\tt \leq$'
        op1 = '$\tt <$'
        op_fmtd = f"{op0} {self.var} {op1}"
        wc_tex = wc + len(op0) + len(op1) - 2
        label_l = f"{{:>{wl}}} ".format(self._format_level(lvl0))
        label_c = f"{{:^{wc_tex}}}".format(op_fmtd)
        label_r = f"{{:<{wr}}}".format(self._format_level(lvl1))
        return f"{label_l}{label_c}{label_r}"

    def _format_open_right(self, lvl):
        wl, wc, wr = self.widths
        op0 = r'$\tt \leq$'
        op1 = ' '
        op_fmtd = f"{op0} {self.var} {op1}"
        wc_tex = wc + len(op0) + len(op1) - 2
        label_l = f"{{:>{wl}}} ".format(self._format_level(lvl))
        label_c = f"{{:^{wc_tex}}}".format(op_fmtd)
        label_r = ' '*wr
        return f"{label_l}{label_c}{label_r}"

    def _format_open_left(self, lvl):
        lvl_fmtd = self._format_level(lvl)
        wl, wc, wr = self.widths
        op0 = ' '
        op1 = '$\tt <$'
        op_fmtd = f"{op0} {self.var} {op1}"
        wc_tex = wc + len(op0) + len(op1) - 2
        label_l = ' '*wl
        label_c = f"{{:^{wc_tex}}}".format(op_fmtd)
        label_r = f"{{:>{wr}}}".format(self._format_level(lvl))
        return f"{label_l}{label_c}{label_r}"
