# -*- coding: utf-8 -*-
"""
Input specifications.
"""
import itertools
import logging as log
import numpy as np

from copy import copy, deepcopy

from srutils.dict import dict_mult_vals_product
from srutils.dict import nested_dict_set
from srutils.dict import pformat_dictlike
from srutils.various import isiterable

from .utils import ParentClass
from .utils import SummarizableClass
from .words import WORDS


def int_or_list(arg):
    try:
        iter(arg)
    except TypeError:
        return int(arg)
    else:
        return [int(a) for a in arg]


# Variable Specifications


class VarSpecs(SummarizableClass, ParentClass):
    """FLEXPART input variable specifications."""

    summarizable_attrs = []  # SR_TODO

    # Keys with respective type
    specs_type_default = {
        "species_id": (int_or_list, None),
        "integrate": (bool, None),
        "time": (int, None),
        "nageclass": (int, None),
        "numpoint": (int, None),
    }

    # SR_TMP <<<
    def issubcls(self, name):
        return name in self.name.split(":")

    # @classmethod
    # def subcls(cls, name):
    #     raise DeprecationWarning(f"{cls.__name__}.subcls")

    def __init__(self, name, var_specs_dct, *, rlat=None, rlon=None, words, lang):
        """Create an instance of ``VarSpecs``.

        Args:
            name (str): Name.

            var_specs_dct (dict): Variable specification dict comprised of only
                single-object elements (see ``VarSpecs.create`` for more
                information on single- vs. multi-object elements).

            rlat (tuple, optional): Rotated latitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            rlon (tuple, optional): Rotated longitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            words (Words, optional): Word translations. Defaults to ``WORDS``.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).

        """
        assert name in self.name  # SR_TMP
        self.name = name

        def prepare_dim(dim):
            if dim is None:
                dim = (None,)
            elif isinstance(dim, slice):
                dim = (dim.start, dim.stop, dim.step)
            try:
                slice(*dim)
            except ValueError:
                raise
            return dim

        self.rlat = prepare_dim(rlat)
        self.rlon = prepare_dim(rlon)

        self._words = words or WORDS
        self._lang = lang or "en"
        self._words.set_default_lang(self._lang)

        self._set_attrs(var_specs_dct)

    def _set_attrs(self, var_specs_dct):
        var_specs_dct_todo = {k: v for k, v in var_specs_dct.items()}
        for key, (type_, default) in self.specs_type_default.items():
            try:
                val = var_specs_dct_todo.pop(key)
            except KeyError:
                val = default
            if val is not None:
                try:
                    val = type_(val)
                except TypeError:
                    raise ValueError(
                        f"argument '{key}': type '{type(val).__name__}' incompatible "
                        f"with '{type_.__name__}'"
                    )
            setattr(self, key, val)
        if var_specs_dct_todo:
            raise ValueError(
                f"{len(var_specs_dct_todo)} unexpected arguments: "
                f"{sorted(var_specs_dct_todo)}"
            )

    @classmethod
    def create(cls, name, var_specs_dct, **kwargs):
        """Create one or more instances of ``VarSpecs``.

        In general, the values of the specification dict ``var_specs_dct`` are
        single objects (usually a number or strings); these are single-object
        elements of the dict. ``VarSpecs`` objects are created from
        specification dicts comprised of only such single-object elements.

        However, there may also be multi-object elements, the key of which ends
        in "_lst", and the value of which is a list of multiple objects. (This
        is effectively a shorthand to specify multiple specification dicts in
        one.)

        To expand such a specification dict with multi-object elements into
        multiple dicts comprised only of single-object elements, all unique
        combinations of the objects in these lists are determined.

        From each single-object-value-only specification dict a ``VarSpecs``
        object is created.

        Args:
            name (str): Name.

            var_specs_dct (dict): Variable specification dict (see above).

            **kwargs: Additional keyword arguments used to create the
                individual ``VarSpecs`` objects.

        Returns:
            list[VarSpecs]: One or move ``VarSpecs`` objects.

        Examples:
            var_specs_dct == {"foo": 1, "bar": 2}
            # -> [VarSpecs(foo=1, bar=1)]

            var_specs_dct == {"foo": [1, 2], "bar": 3}
            # -> [VarSpecs(foo=1, bar=3), VarSpecs(foo=2, bar=3)]

            var_specs_dct == {"foo": [1, 2], "bar": [3, 4], "baz": "hi"}
            # -> [VarSpecs(foo=1, bar=3, baz="hi"),
                  VarSpecs(foo=1, bar=4, baz="hi"),
                  VarSpecs(foo=2, bar=3, baz="hi"),
                  VarSpecs(foo=2, bar=4, baz="hi")]

        """
        cls = cls.subcls(name)  # SR_TMP
        var_specs_dct_lst = dict_mult_vals_product(var_specs_dct)
        var_specs_lst = [cls(name, d, **kwargs) for d in var_specs_dct_lst]
        # breakpoint(header=f"{cls.__name__}.create")
        return var_specs_lst

    def merge_with(self, others):

        # Words and language
        for other in others:
            if other._lang != self._lang:
                raise Exception(
                    f"merge of {other} with {self} failed: languages differ: "
                    f"{other._lang} != {self._lang}"
                )
            if other._words != self._words:
                raise Exception(
                    f"merge of {other} with {self} failed: words differ: "
                    f"{other._words} != {self._words}"
                )

        # Attributes
        attrs = {}
        for key, val0 in sorted(self):

            vals = [val0]
            for other in others:
                val = getattr(other, key)
                if val not in vals:
                    vals.append(val)

            if len(vals) == 1:
                attrs[key] = next(iter(vals))
            elif key == "deposition" and set(vals) == set(["dry", "wet"]):
                attrs[key] = "tot"
            else:
                attrs[key] = vals

        return type(self)(words=self._words, lang=self._lang, **attrs)

    def __hash__(self):
        h = 0
        for key, val in sorted(iter(self)):
            if isinstance(val, slice):
                val = (val.start, val.stop, val.step)
            elif isiterable(val, str_ok=False):
                val = tuple(val)
            h += hash(val)
            h *= 10
        return h

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return pformat_dictlike(self)

    def __str__(self):
        return pformat_dictlike(self)

    def __getitem__(self, key):
        if key.startswith("_"):
            raise ValueError(f"invalid key '{key}'")
        try:
            return self.__dict__[key]
        except KeyError as e:
            raise e from None

    def __setitem__(self, key, val):
        if key.startswith("_") or key not in self.__dict__:
            raise ValueError(f"invalid key '{key}'")
        self.__dict__[key] = val

    def __iter__(self):
        for key, val in sorted(self.__dict__.items()):
            if not key.startswith("_"):
                yield key, val

    def var_name(self):
        """Derive variable name from specifications."""
        raise NotImplementedError(f"{type(self).__name__}.var_name")

    def dim_inds_by_name(self, *, rlat=None, rlon=None):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds["nageclass"] = self.nageclass
        inds["numpoint"] = self.numpoint

        if not self.integrate or self.time == slice(None):
            inds["time"] = self.time
        else:
            inds["time"] = slice(None, self.time + 1)

        inds["rlat"] = slice(*self.rlat)
        inds["rlon"] = slice(*self.rlon)

        return inds

    def long_name(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.long_name: override it!")

    def short_name(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.short_name: override it!")

    def var_name(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.var_name: override it!")


class VarSpecs_Concentration(VarSpecs):
    name = "concentration"

    specs_type_default = {
        **VarSpecs.specs_type_default,
        "level": (int_or_list, None),
    }

    def long_name(self):
        ctx = "abbr" if self.integrate else "*"
        return self._words["activity_concentration"].s

    def short_name(self):
        s = ""
        if self.integrate:
            return (
                f"{self._words['integrated', None, 'abbr']} "
                f"{self._words['concentration', None, 'abbr']}"
            )
        return self._words["concentration"].s

    def var_name(self):
        """Derive variable name from specifications."""

        def fmt(sid):
            return f"spec{sid:03d}"

        try:
            iter(self.species_id)
        except TypeError:
            return fmt(self.species_id)
        else:
            return [fmt(sid) for sid in self.species_id]

    def dim_inds_by_name(self, *args, **kwargs):
        """Derive indices along NetCDF dimensions."""
        inds = super().dim_inds_by_name(*args, **kwargs)
        inds["level"] = self.level
        return inds


class VarSpecs_Deposition(VarSpecs):
    name = "deposition"

    specs_type_default = {
        **VarSpecs.specs_type_default,
        "deposition": (str, None),
    }

    def deposition_type(self):
        type_ = self["deposition"]
        word = "total" if type_ == "tot" else type_
        return self._words[word, None, "f"].s

    def long_name(self):
        return f"{self.deposition_type()} {self._words['surface_deposition']}"

    def short_name(self):
        return self._words["deposition"].s

    def var_name(self):
        """Derive variable name from specifications."""
        prefix = {"wet": "WD", "dry": "DD"}[self.deposition]
        return f"{prefix}_spec{self.species_id:03d}"


class VarSpecs_AffectedArea(VarSpecs_Deposition):
    name = "deposition:affected_area"

    def long_name(self):
        dep_name = super().long_name()
        return f"{self._words['affected_area']} " f"({dep_name})"


class Varspecs_AffectedAreaMono(VarSpecs_AffectedArea):
    name = "deposition:affected_area:affected_area_mono"


class VarSpecs_EnsMean_Concentration(VarSpecs_Concentration):
    name = "concentration:ens_mean_concentration"

    def long_name(self):
        return (
            f"{self._words['activity_concentration']}\n"
            f"{self._words['ensemble_mean']}"
        )


class VarSpecs_EnsMean_Deposition(VarSpecs_Deposition):
    name = "deposition:ens_mean_deposition"

    def long_name(self):
        return (
            f"{self._words['ensemble_mean']} {self.deposition_type()} "
            f"{self._words['surface_deposition']}"
        )


class VarSpecs_EnsMean_AffectedArea(VarSpecs_AffectedArea):
    name = "deposition:affected_area:ens_mean_affected_area"

    def long_name(self):
        return (
            f"{self._words['ensemble_mean']} {self._words['affected_area']} "
            f"({self.deposition_type()})"
        )


class VarSpecs_EnsThrAgrmt:
    def long_name(self):
        # SR_TMP <<<
        lang = self._words.default_lang
        of = dict(en="of", de="der")[lang]
        return (
            f"{self._words['ensemble']}{dict(en=' ', de='-')[lang]}"
            f"{self._words['threshold_agreement']} {of} {super().long_name()}"
        )

    def short_name(self):
        return (
            f"{self._words['number_of', None, 'abbr'].c} "
            f"{self._words['member', None, 'pl']}"
        )


class VarSpecs_EnsThrAgrmt_Concentration(VarSpecs_EnsThrAgrmt, VarSpecs_Concentration):
    name = "concentration:ens_thr_agrmt_concentration"


class VarSpecs_EnsThrAgrmt_Deposition(VarSpecs_EnsThrAgrmt, VarSpecs_Deposition):
    name = "deposition:ens_thr_agrmt_deposition"


class VarSpecs_EnsThrAgrmt_AffectedArea(VarSpecs_EnsThrAgrmt, VarSpecs_AffectedArea):
    name = "deposition:affected_area:ens_thr_agrmt_affected_area"


# Field Specifications


class FieldSpecs(SummarizableClass):
    """FLEXPART field specifications."""

    summarizable_attrs = []  # SR_TODO

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = ["species_id"]

    def __init__(
        self, name, var_specs_lst, *, op=np.nansum, var_attrs_replace=None, lang="en",
    ):
        """Create an instance of ``FieldSpecs``.

        Args:
            name (str): Name.

            var_specs_lst (list[VarSpecs]): Specifications of one or more input
                variables used to subsequently create a plot field.

            op (function or list[function], optional): Opterator(s) used to
                combine input fields read based on ``var_specs_lst``. Must
                accept argument ``axis=0`` to only recude along over the
                fields.

                If a single operator is passed, it is used to sequentially
                combine one field after the other, in the same order as the
                corresponding specifications (``var_specs_lst``).

                If a list of operators has been passed, then it's length must
                be one smaller than that of ``var_specs_lst``, such that each
                operator is used between two subsequent fields (again in the
                same order as the corresponding specifications).

                Defaults to np.nansum.

            var_attrs_replace (dict[str: dict], optional): Variable attributes
                to be replaced. Necessary if multiple specifications dicts are
                passed for all those attributes that differ between the
                resulting attributes collections. Defaults to '{}'.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).
        """
        self.name = name

        # SR_TMP <
        assert not isinstance(var_specs_lst, VarSpecs)
        assert isiterable(var_specs_lst, str_ok=False)
        assert isinstance(next(iter(var_specs_lst)), VarSpecs)
        # SR_TMP >

        # Create variable specifications objects
        self.var_specs_lst = self._prepare_var_specs_lst(var_specs_lst)

        # Store operator(s)
        self.check_op(op)
        if callable(op):
            self.op = op
            self.op_lst = None
        else:
            self.op = None
            self.op_lst = op

        # SR_TMP < SR_TODO remove var_attrs_replace if this is not triggered!
        if var_attrs_replace is not None:
            raise Exception(
                f"{type(self).__name__}: var_attrs_replace is not None: "
                f"{var_attrs_replace}"
            )
        # SR_TMP >

        # Store variable attributes
        if var_attrs_replace is None:
            var_attrs_replace = {}
        self.var_attrs_replace = var_attrs_replace

    def set_addtl_attrs(self, **attrs):
        """Set additional attributes."""
        for attr, val in sorted(attrs.items()):
            if hasattr(self, attr):
                raise ValueError(
                    f"attribute '{type(self).__name__}.{attr}' already exists"
                )
            setattr(self, attr, val)

    def _prepare_var_specs_lst(self, var_specs_lst):

        try:
            iter(var_specs_lst)
        except TypeError:
            raise ValueError(
                f"var_specs: type '{type(var_specs_lst).__name__}' not iterable"
            ) from None

        # Handle dimensions with optionally multiple values
        # Example: Sum over multiple species
        for key in self.dims_opt_mult_vals:
            for var_specs in copy(var_specs_lst):
                assert isinstance(var_specs, VarSpecs)  # SR_TMP
                if not isiterable(var_specs, str_ok=False):
                    vals = copy(var_specs[key])
                    var_specs[key] = vals.pop(0)
                    var_specs_lst_new = [deepcopy(var_specs) for _ in vals]
                    for var_specs_new, val in zip(var_specs_lst_new, vals):
                        var_specs_new[key] = val
                        var_specs_lst.append(var_specs_new)

        return var_specs_lst

    def check_op(self, op):
        """Check operator(s)."""
        try:
            n_ops = len(op)
        except TypeError:
            if not callable(op):
                raise ValueError(f"op: {type(op).__name__} not callable")
            return

        n_var_specs = len(self.var_specs_lst)
        if n_ops != n_var_specs - 1:
            raise ValueError(
                f"wrong number of operators passed in {type(ops).__name__}: {n_ops} != "
                f"{n_var_specs}"
            )
        for op_i in op:
            if not callable(op_i):
                raise ValueError(f"op: {type(op_i).__name__} not callable")

    def __repr__(self):
        s = f"{type(self).__name__}(\n"

        # Variables specifications
        s += f"  var_specs: {len(self.var_specs_lst)}x\n"
        for var_specs in self.var_specs_lst:
            for line in str(var_specs).split("\n"):
                s += f"    {line}\n"

        # Operator(s)
        if self.op is not None:
            s += f"  op: {self.op.__name__}\n"
        else:
            s += f"  ops: {len(self.op_lst)}x\n"
            for op in self.op_lst:
                s += "f    {op.__name__}\n"

        # Variable attributes replacements
        s += f"  var_attrs_replace: {len(self.var_attrs_replace)}x\n"
        for key, val in sorted(self.var_attrs_replace.items()):
            s += f"    '{key}': {val}\n"

        s += f")"
        return s

    def __hash__(self):
        return sum([sum([hash(vs) for vs in self.var_specs_lst])])

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def var_specs_merged(self):
        """Return merged variable specifications."""
        return self.var_specs_lst[0].merge_with(self.var_specs_lst[1:])

    def var_specs_shared(self, key):
        """Return a varible specification, if it is shared by all."""
        vals = [getattr(vs, key) for vs in self.var_specs_lst]
        all_equal = all(v == vals[0] for v in vals[1:])
        if not all_equal:
            raise ValueError(
                f"'{key}' differs among {len(self.var_specs_lst)} var stats: {vals}"
            )
        return next(iter(vals))


class FieldSpecs_Concentration(FieldSpecs):
    name = "concentration"

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = FieldSpecs.dims_opt_mult_vals + ["level"]

    def __init__(self, var_specs, *args, **kwargs):
        """Create an instance of ``FieldSpecs_Concentration``."""
        super().__init__([var_specs], *args, **kwargs)


class FieldSpecs_Deposition(FieldSpecs):
    name = "deposition"

    def __init__(self, var_specs, *args, lang=None, **kwargs):
        """Create an instance of ``FieldSpecs_Deposition``."""
        lang = lang or "en"

        var_specs_lst = [var_specs]

        # Deposition mode
        if var_specs["deposition"] in ["wet", "dry"]:
            pass

        elif var_specs["deposition"] == "tot":
            long_name = var_specs.long_name()
            nested_dict_set(
                kwargs,
                ["var_attrs_replace", "variable", "long_name", "value"],
                long_name,
            )
            var_specs_new = deepcopy(var_specs)
            var_specs["deposition"] = "wet"
            var_specs_new["deposition"] = "dry"
            var_specs_lst.append(var_specs_new)

        else:
            raise NotImplementedError(f"deposition type '{var_specs['deposition']}'")

        super().__init__(var_specs_lst, *args, **kwargs)


class FieldSpecs_AffectedArea(FieldSpecs_Deposition):
    name = "affected_area"


class FieldSpecs_AffectedAreaMono(FieldSpecs_AffectedArea):
    name = "affected_area_mono"


class FieldSpecs_Ens(FieldSpecs):
    name = "ens"


class FieldSpecs_EnsMean_Concentration(FieldSpecs_Ens, FieldSpecs_Concentration):
    name = "ens_mean_concentration"


class FieldSpecs_EnsMean_Deposition(FieldSpecs_Ens, FieldSpecs_Deposition):
    name = "ens_mean_deposition"


class FieldSpecs_EnsMean_AffectedArea(FieldSpecs_Ens, FieldSpecs_AffectedArea):
    name = "ens_mean_affected_area"


class FieldSpecs_EnsThrAgrmt_Concentration(FieldSpecs_Ens, FieldSpecs_Concentration):
    name = "ens_thr_agrmt_concentration"


class FieldSpecs_EnsThrAgrmt_Deposition(FieldSpecs_Ens, FieldSpecs_Deposition):
    name = "ens_thr_agrmt_deposition"
