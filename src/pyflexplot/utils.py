# -*- coding: utf-8 -*-
"""
Utils for the command line tool.
"""
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


#======================================================================


def format_level_ranges(style, widths, extend, levels, **kwargs):
    cls = {
        'simple': LevelRangeFormatter_Simple,
        'simple-int': LevelRangeFormatter_SimpleInt,
        'math': LevelRangeFormatter_Math,
        'up': LevelRangeFormatter_Up,
        'down': LevelRangeFormatter_Down,
        'and': LevelRangeFormatter_And,
        'var': LevelRangeFormatter_Var,
    }.get(style, LevelRangeFormatter)  #SR_TMP
    return cls(style, widths, extend, **kwargs).format_multiple(levels)


class LevelRangeFormatter:
    """Format level ranges, e.g., for legends of color contour plots."""

    def __init__(self, style, widths, extend, rstrip_zeros=True, var='v'):
        self.style = style  #SR_TMP
        self._check_widths(widths)
        self.widths = widths
        self.extend = extend
        self.rstrip_zeros = rstrip_zeros
        self.var = var  #SR_TMP specific to style='var'


    def _check_widths(self, widths):
        try:
            wl, wc, wr = [int(w) for w in widths]
        except (ValueError, TypeError):
            raise ValueError(f"widths is not a tree-int tuple: {widths}")

    def format_multiple(self, levels):

        labels = []

        # Under range
        if self.extend in ('min', 'both'):
            labels.append(self.format(None, levels[0]))

        # In range
        for lvl0, lvl1 in zip(levels[:-1], levels[1:]):
            labels.append(self.format(lvl0, lvl1))

        # Over range
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
        wl, wc, wr = self.widths
        if lvl0 is None:
            lvl0 = -np.inf
        if lvl1 is None:
            lvl1 = np.inf
        return f"{{:{wl}.1e}}{{:^{wc + 2}}}{{:{wr}.1e}}".format(lvl0, r'$-$', lvl1)

    #SR_TMP<<<
    def _format_level(self, lvl):
        if lvl is None:
            raise ValueError(f"lvl is None")
        fmtd = f'{lvl:.0E}'
        n = len(fmtd)
        ll = np.log10(lvl)
        if ll >= -n + 2 and ll <= n - 1:
            fmtd = f'{lvl:f}' [:n]
        return fmtd


class LevelRangeFormatter_Simple(LevelRangeFormatter):

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
        return f"{{:>{wl}}}{{:^{wc + 2}}}{{:<{wr}}}".format(
            lvl0_fmtd, r'$-$', lvl1_fmtd)

    def _format_open_left(self, lvl):
        op = r'$<$'
        len_op = 2
        return self._format_open_core(op, len_op, lvl)

    def _format_open_right(self, lvl):
        op = r'$\geq$'
        len_op = 2
        return self._format_open_core(op, len_op, lvl)

    def _format_open_core(self, op, len_op, lvl):
        lvl_fmtd = self._format_level(lvl)
        wl, wc, wr = self.widths
        wc_tex = wc + len(op) - len_op
        if self.style == 'var':
            label = f"{{:^{wl}}}".format(self.var)
        else:
            label = ' '*wl
        label += f"{{:^{wc_tex}}}{{:>{wr}}}".format(op, lvl_fmtd)
        return label


class LevelRangeFormatter_SimpleInt(LevelRangeFormatter_Simple):

    def __init__(self, style, widths, extend):
        super().__init__(style, widths, extend, rstrip_zeros=False)

    def _format_closed(self, lvl0, lvl1):

        # Check input types
        if int(lvl0) != float(lvl0):
            raise ValueError(
                f"lvl0 is not an integer: {int(lvl0)} != {float(lvl0)}")
        if int(lvl1) != float(lvl1):
            raise ValueError(
                f"lvl1 is not an integer: {int(lvl1)} != {float(lvl1)}")

        lvl0_fmtd = self._format_level(lvl0)
        lvl1 = lvl1 - 1
        if lvl1 == lvl0:
            return f"{lvl0_fmtd}"
        lvl1_fmtd = self._format_level(lvl1)

        return f"{lvl0_fmtd:>6} $-$ {lvl1_fmtd:<6}"

    def _format_level(self, lvl):
        return int(lvl)


class LevelRangeFormatter_Math(LevelRangeFormatter):

    def _format_closed(self, lvl0, lvl1):
        label_l = self._format_l(lvl0)
        label_c = self._format_c()
        label_r = self._format_r(lvl1)
        return f"{label_l}{label_c}{label_r}"

    def _format_open_left(self, lvl):
        label_l = self._format_l(None)
        label_c = self._format_c()
        label_r = self._format_r(lvl)
        return f"{label_l}{label_c}{label_r}"

    def _format_open_right(self, lvl):
        label_l = self._format_l(lvl)
        label_c = self._format_c()
        label_r = self._format_r(None)
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

    def _format_core(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        if lvl0 is None:
            lvl0 = np.inf
        lvl0_fmtd = self._format_level(lvl0)
        return  f"$\geq$ {lvl0_fmtd:<}"
        label_l = ' '*wl
        label_c = f"{{:^{wc + 4}}}".format(r'$\geq$')
        label_r = f"{{:<{wr}}}".format(self._format_level(lvl0))
        return f"{label_c}{label_r}"


class LevelRangeFormatter_Down(LevelRangeFormatter):

    def _format_core(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        if lvl1 is None:
            lvl1 = np.inf
        label_l = ' '*wl
        label_c = f"{{:^{wc + 1}}}".format(r'$<$')
        label_r = f"{{:<{wr}}}".format(self._format_level(lvl1))
        return f"{label_l}{label_c}{label_r}"


class LevelRangeFormatter_And(LevelRangeFormatter):

    def _format_core(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        if lvl0 is None:
            lvl0 = -np.inf
        if lvl1 is None:
            lvl1 = np.inf
        label_l = f"$\geq$ {{:>{wl - 3}}}".format(self._format_level(lvl0))
        label_c = f"{{:^{wc}}}".format(r'$&$')
        label_r = f"$<$ {{:>{wl - 3}}}".format(self._format_level(lvl1))
        return f"{label_l}{label_c}{label_r}"

class LevelRangeFormatter_Var(LevelRangeFormatter):

    def _format_core(self, lvl0, lvl1):
        wl, wc, wr = self.widths
        if lvl0 is None:
            lvl0 = -np.inf
        if lvl1 is None:
            lvl1 = np.inf
        lvl0_fmtd = self._format_level(lvl0)
        lvl1_fmtd = self._format_level(lvl1)
        return (
            f"{lvl0_fmtd:>} " + r'$\leq$' +
            f" {self.var} < " + f"{lvl1_fmtd:<}")
