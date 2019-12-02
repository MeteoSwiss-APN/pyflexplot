# -*- coding: utf-8 -*-
"""
Input specifications.
"""
import itertools
import logging as log
import numpy as np

from copy import copy, deepcopy

from srutils.dict import nested_dict_set
from srutils.dict import pformat_dictlike

from .utils import ParentClass
from .utils import SummarizableClass
from .words import words


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
    _keys_w_type = {
        "species_id": int_or_list,
        "integrate": bool,
        # Dimensions
        "time": int,
        "nageclass": int,
        "numpoint": int,
    }

    @classmethod
    def specs(cls, types=False):
        for k, v in sorted(cls._keys_w_type.items()):
            if types:
                yield k, v
            else:
                yield k

    def __init__(self, *, rlat=None, rlon=None, **kwargs):
        """Create an instance of ``VarSpecs``.

        Args:
            rlat (tuple, optional): Rotated latitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            rlon (tuple, optional): Rotated longitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            **kwargs: Arguments as in ``VarSpecs.specs(types=True)``. The keys
                correspond to the argument's names, and the values specify a
                type which the respective argument value must be compatible
                with.

        """

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

        for key, type_ in self.specs(types=True):
            try:
                val = kwargs.pop(key)
            except KeyError:
                raise ValueError(f"missing argument '{key}'")
            try:
                setattr(self, key, type_(val))
            except TypeError:
                raise ValueError(
                    f"argument '{key}': type '{type(val).__name__}' incompatible with "
                    f"'{type_.__name__}'"
                )
        if kwargs:
            raise ValueError(f"{len(kwargs)} unexpected arguments: {sorted(kwargs)}")

    @classmethod
    def multiple(cls, *args, **kwargs):
        """Create multiple instances of ``VarSpecs``.

        Each of the arguments of ``__init__`` can be passed by the original
        name with one value (e.g., ``time=1``) or pluralized with multiple
        values (e.g., ``time=[1, 2]``).

        One ``VarSpecs`` instance is created for each combination of all input
        arguments.

        """
        return cls._multiple_as_type(cls, *args, **kwargs)

    @classmethod
    def multiple_as_dict(cls, *args, **kwargs):
        return cls._multiple_as_type(dict, *args, **kwargs)

    @classmethod
    def _multiple_as_type(cls, type_, rlat=None, rlon=None, **kwargs):
        keys_singular = sorted(cls.specs())
        vals_plural = []
        for key_singular in keys_singular:
            key_plural = f"{key_singular}_lst"

            if key_plural in kwargs:
                # Passed as plural
                if key_singular in kwargs:
                    # Error: passed as both plural and sigular
                    raise ValueError(
                        f"argument conflict: '{key_singular}', '{key_plural}'"
                    )
                vals_plural.append([v for v in kwargs.pop(key_plural)])

            elif key_singular in kwargs:
                # Passed as sigular
                vals_plural.append([kwargs.pop(key_singular)])

            else:
                # Not passed at all
                raise ValueError(
                    f"missing argument: '{key_singular}' or '{key_plural}'"
                )

        if kwargs:
            # Passed too many arguments
            raise ValueError(f"{len(kwargs)} unexpected arguments: {sorted(kwargs)}")

        # Create one specs per parameter combination
        specs_lst = []
        for vals in itertools.product(*vals_plural):
            kwargs_i = {k: v for k, v in zip(keys_singular, vals)}
            specs = type_(rlat=rlat, rlon=rlon, **kwargs_i)
            specs_lst.append(specs)

        return specs_lst

    def merge_with(self, others):
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

        return self.__class__(**attrs)

    def __hash__(self):
        h = 0
        for key, val in sorted(iter(self)):
            if isinstance(val, slice):
                h += hash((val.start, val.stop, val.step))
            else:
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

    def __iter__(self):
        for key, val in sorted(self.__dict__.items()):
            if not key.startswith("_"):
                yield key, val

    def var_name(self):
        """Derive variable name from specifications."""
        raise NotImplementedError(f"{self.__class__.__name__}.var_name")

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


class VarSpecs_Concentration(VarSpecs):
    name = "concentration"

    _keys_w_type = {
        **VarSpecs._keys_w_type,
        "level": int_or_list,
    }

    @classmethod
    def long_name(cls, lang, var_specs):
        ctx = "abbr" if var_specs.integrate else "*"
        return words["activity_concentration", lang, ctx].s

    @classmethod
    def short_name(cls, lang, var_specs):
        s = ""
        if var_specs.integrate:
            return (
                f"{words['integrated', lang, 'abbr'].s} "
                f"{words['concentration', lang, 'abbr'].s}"
            )
        return words["concentration", lang].s

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

    _keys_w_type = {
        **VarSpecs._keys_w_type,
        "deposition": str,
    }

    @classmethod
    def deposition_type(cls, lang, var_specs):
        type_ = dict(var_specs)["deposition"]
        word = "total" if type_ == "tot" else type_
        return words[word, lang, "f"].s

    @classmethod
    def long_name(cls, lang, var_specs, abbr=False):
        dep_type = cls.deposition_type(lang, var_specs)
        ctx = "abbr" if abbr else "*"
        return f"{dep_type} " f"{words['surface_deposition', lang, ctx].s}"

    @classmethod
    def short_name(cls, lang, var_specs):
        return words["deposition", lang].s

    def var_name(self):
        """Derive variable name from specifications."""
        prefix = {"wet": "WD", "dry": "DD"}[self.deposition]
        return f"{prefix}_spec{self.species_id:03d}"


class VarSpecs_AffectedArea(VarSpecs_Deposition):
    name = "affected_area"

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_name = VarSpecs_Deposition.long_name(lang, var_specs, abbr=True)
        return f"{words['affected_area', lang].s} " f"({dep_name})"


class Varspecs_AffectedAreaMono(VarSpecs_AffectedArea):
    name = "affected_area_mono"


class VarSpecs_EnsMean_Concentration(VarSpecs_Concentration):
    name = "ens_mean_concentration"

    @classmethod
    def long_name(cls, lang, var_specs):
        return (
            f"{words['activity_concentration', lang].s}\n"
            f"{words['ensemble_mean', lang].s}"
        )


class VarSpecs_EnsMean_Deposition(VarSpecs_Deposition):
    name = "ens_mean_deposition"

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_type = cls.deposition_type(lang, var_specs)
        return (
            f"{words['ensemble_mean', lang].s} {dep_type} "
            f"{words['surface_deposition'].s}"
        )


class VarSpecs_EnsMean_AffectedArea(VarSpecs_AffectedArea):
    name = "ens_mean_affected_area"

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_type = cls.deposition_type(lang, var_specs)
        return (
            f"{words['ensemble_mean', lang].s} {words['affected_area', lang].s} "
            f"({dep_type})"
        )


class VarSpecs_EnsThrAgrmt:
    @classmethod
    def long_name(cls, lang, var_specs):
        long_name_super = super().long_name(lang, var_specs)
        long_name_base = (
            # SR_TMP <
            f"{words['ensemble'].s}{dict(en=' ', de='-')[lang]}"
            f"{words['threshold_agreement'].s} {dict(en='of', de='der')[lang]} "
            # SR_TMP >
        )
        return long_name_base + long_name_super

    @classmethod
    def short_name(cls, lang, var_specs):
        return "Members"


class VarSpecs_EnsThrAgrmt_Concentration(VarSpecs_EnsThrAgrmt, VarSpecs_Concentration):
    name = "ens_thr_agrmt_concentration"


class VarSpecs_EnsThrAgrmt_Deposition(VarSpecs_EnsThrAgrmt, VarSpecs_Deposition):
    name = "ens_thr_agrmt_deposition"


class VarSpecs_EnsThrAgrmt_AffectedArea(VarSpecs_EnsThrAgrmt, VarSpecs_AffectedArea):
    name = "ens_thr_agrmt_affected_area"


# Field Specifications


class FieldSpecs(SummarizableClass, ParentClass):
    """FLEXPART field specifications."""

    cls_var_specs = VarSpecs

    summarizable_attrs = []  # SR_TODO

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = ["species_id"]

    def __init__(
        self,
        var_specs_lst,
        *,
        op=np.nansum,
        var_attrs_replace=None,
        lang="en",
        **addtl_attrs,
    ):
        """Create an instance of ``FieldSpecs``.

        Args:
            var_specs_lst (list[dict]): Specifications dicts of input
                variables, each of which is is used to create an instance of
                ``VarSpecs`` as specified by the class attribute
                ``cls_var_specs``. Each ultimately yields a 2D slice of an
                input variable.

            op (function or list[function], optional): Opterator(s) to combine
                the input fields obtained based on the input variable
                specifications. If multipe operators are passed, their number
                must one smaller than that of the specifications, and they are
                applied consecutively from left to right without regard to
                operator precedence. Must accept argument ``axis=0`` to only
                recude along over the fields. Defaults to np.nansum.

            var_attrs_replace (dict[str: dict], optional): Variable attributes
                to be replaced. Necessary if multiple specifications dicts are
                passed for all those attributes that differ between the
                resulting attributes collections. Defaults to '{}'.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).
        """

        self._prepare_var_specs_lst(var_specs_lst)

        # Create variable specifications objects
        self.var_specs_lst = self.create_var_specs(var_specs_lst)

        # Set additional attributes
        self.set_addtl_attrs(**addtl_attrs)

        # Store operator(s)
        self.check_op(op)
        if callable(op):
            self.op = op
            self.op_lst = None
        else:
            self.op = None
            self.op_lst = op

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

        # Handle dimensions with optionally multiple values
        # Example: Sum over multiple species
        for key in self.dims_opt_mult_vals:
            for var_specs in copy(var_specs_lst):
                try:
                    iter(var_specs[key])
                except TypeError:
                    pass
                else:
                    vals = copy(var_specs[key])
                    var_specs[key] = vals.pop(0)
                    var_specs_lst_new = [deepcopy(var_specs) for _ in vals]
                    for var_specs_new, val in zip(var_specs_lst_new, vals):
                        var_specs_new[key] = val
                        var_specs_lst.append(var_specs_new)

    def create_var_specs(self, var_specs_dct_lst):
        """Create variable specifications objects from dicts."""
        try:
            iter(var_specs_dct_lst)
        except TypeError:
            raise ValueError(
                f"var_specs: type '{type(var_specs_dct_lst).__name__}' not iterable"
            ) from None

        var_specs_lst = []
        for i, kwargs_specs in enumerate(var_specs_dct_lst):
            try:
                var_specs = self.cls_var_specs(**kwargs_specs)
            except Exception as e:
                raise ValueError(
                    f"var_specs[{i}]: cannot create instance of "
                    f"{self.cls_var_specs.__name__} from {kwargs_specs}: "
                    f"{e.__class__.__name__}({e})"
                ) from None
            else:
                var_specs_lst.append(var_specs)

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
        s = f"{self.__class__.__name__}(\n"

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

    @classmethod
    def multiple(cls, vars_specs, *args, **kwargs):
        var_specs_lst = cls.cls_var_specs.multiple_as_dict(**vars_specs)
        field_specs_lst = []
        for var_specs in var_specs_lst:
            try:
                field_specs = cls(var_specs, *args, **kwargs)
            except Exception as e:
                raise Exception(
                    f"cannot initialize {cls.__name__} "
                    f"({type(e).__name__}: {e})"
                    f"\nvar_specs: {var_specs}"
                )
            else:
                field_specs_lst.append(field_specs)
        return field_specs_lst

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
    cls_var_specs = VarSpecs.subclass("concentration")

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = FieldSpecs.dims_opt_mult_vals + ["level"]

    def __init__(self, var_specs, *args, **kwargs):
        """Create an instance of ``FieldSpecs_Concentration``.

        Args:
            var_specs (dict): Specifications dict of input variable used to
                create an instance of ``VarSpecs_Concentration`` as specified
                by the class attribute ``cls_var_specs``.

            **kwargs: Keyword arguments passed to ``FieldSpecs``.

        """
        if not isinstance(var_specs, dict):
            raise ValueError(
                f"var_specs must be 'dict', not '{type(var_specs).__name__}'"
            )
        super().__init__([var_specs], *args, **kwargs)


class FieldSpecs_Deposition(FieldSpecs):
    name = "deposition"
    cls_var_specs = VarSpecs.subclass("deposition")

    def __init__(self, var_specs, *args, lang="en", **kwargs):
        """Create an instance of ``FieldSpecs_Deposition``.

        Args:
            var_specs (dict): Specifications dict of input variable used to
                create instance(s) of ``VarSpecs_Deposition`` as specified by
                the class attribute ``cls_var_specs``.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).

            **kwargs: Keyword arguments passed to ``FieldSpecs``.

        """
        var_specs_lst = [dict(var_specs)]

        # Deposition mode
        for var_specs in copy(var_specs_lst):

            if var_specs["deposition"] in ["wet", "dry"]:
                pass

            elif var_specs["deposition"] == "tot":
                nested_dict_set(
                    kwargs,
                    ["var_attrs_replace", "variable", "long_name", "value"],
                    self.cls_var_specs.long_name(lang, var_specs),
                )
                var_specs_new = deepcopy(var_specs)
                var_specs["deposition"] = "wet"
                var_specs_new["deposition"] = "dry"
                var_specs_lst.append(var_specs_new)

            else:
                raise NotImplementedError(
                    f"deposition type '{var_specs['deposition']}'"
                )

        super().__init__(var_specs_lst, *args, **kwargs)


class FieldSpecs_AffectedArea(FieldSpecs_Deposition):
    name = "affected_area"
    cls_var_specs = VarSpecs.subclass("affected_area")


class FieldSpecs_AffectedAreaMono(FieldSpecs_AffectedArea):
    name = "affected_area_mono"


class FieldSpecs_Ens(FieldSpecs):
    name = "ens"


class FieldSpecs_EnsMean_Concentration(FieldSpecs_Ens, FieldSpecs_Concentration):
    name = "ens_mean_concentration"
    cls_var_specs = VarSpecs.subclass("ens_mean_concentration")


class FieldSpecs_EnsMean_Deposition(FieldSpecs_Ens, FieldSpecs_Deposition):
    name = "ens_mean_deposition"
    cls_var_specs = VarSpecs.subclass("ens_mean_deposition")


class FieldSpecs_EnsMean_AffectedArea(FieldSpecs_Ens, FieldSpecs_AffectedArea):
    name = "ens_mean_affected_area"
    cls_var_specs = VarSpecs.subclass("ens_mean_affected_area")


class FieldSpecs_EnsThrAgrmt_Concentration(FieldSpecs_Ens, FieldSpecs_Concentration):
    name = "ens_thr_agrmt_concentration"
    cls_var_specs = VarSpecs.subclass("ens_thr_agrmt_concentration")


class FieldSpecs_EnsThrAgrmt_Deposition(FieldSpecs_Ens, FieldSpecs_Deposition):
    name = "ens_thr_agrmt_deposition"
    cls_var_specs = VarSpecs.subclass("ens_thr_agrmt_deposition")
