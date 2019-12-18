# -*- coding: utf-8 -*-
"""
Input specifications.
"""
import logging as log
import numpy as np

from copy import copy, deepcopy

from srutils.dict import nested_dict_set
from srutils.various import isiterable

from .utils import SummarizableClass
from .var_specs import VarSpecs


class FieldSpecs(SummarizableClass):
    """FLEXPART field specifications."""

    summarizable_attrs = []  # SR_TODO fill!

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = ["species_id"]

    def __init__(
        self,
        name,
        var_specs_lst,
        attrs=None,
        *,
        op=np.nansum,
        var_attrs_replace=None,
        lang="en",
    ):
        """Create an instance of ``FieldSpecs``.

        Args:
            name (str): Name.

            var_specs_lst (list[VarSpecs]): Specifications of one or more input
                variables used to subsequently create a plot field.

            attrs (dict[str], optional): Additional arbitrary attributes.
                Defaults to None.

        Kwargs:
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

        self.set_attrs(attrs)

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

    def set_attrs(self, attrs):
        """Set instance attributes."""
        if attrs is None:
            return
        for attr, val in attrs.items():
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
        for key, val in self.var_attrs_replace.items():
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
