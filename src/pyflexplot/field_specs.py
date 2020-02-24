# -*- coding: utf-8 -*-
"""
Field specifications.
"""
# Third-party
import numpy as np

# Local
from .utils import summarizable
from .var_specs import MultiVarSpecs


@summarizable
class FieldSpecs:
    """FLEXPART field specifications."""

    # Dimensions with optionally multiple values
    # SR_TMP <<<
    @property
    def dims_opt_mult_vals(self):
        lst = ["species_id"]
        if self.name == "concentration":
            lst.append("level")
        return lst

    def __init__(
        self, name, multi_var_specs, attrs=None, *, op=np.nansum,
    ):
        """Create an instance of ``FieldSpecs``.

        Args:
            name (str): Name.

            multi_var_specs (MultiVarSpecs): Specifications of one or more
                input variables used to subsequently create a plot field.

            attrs (dict[str], optional): Additional arbitrary attributes.
                Defaults to None.

        Kwargs:
            op (function or list[function], optional): Opterator(s) used to
                combine input fields read based on ``multi_var_specs``. Must
                accept argument ``axis=0`` to only recude along over the
                fields.

                If a single operator is passed, it is used to sequentially
                combine one field after the other, in the same order as the
                corresponding specifications (``multi_var_specs``).

                If a list of operators has been passed, then it's length must
                be one smaller than that of ``multi_var_specs``, such that each
                operator is used between two subsequent fields (again in the
                same order as the corresponding specifications).

                Defaults to np.nansum.

        """
        self.name = name
        # SR_TMP <
        if not isinstance(self, FieldSpecs):
            assert self.name == type(self).name
        # SR_TMP >

        assert isinstance(multi_var_specs, MultiVarSpecs)  # SR_TMP
        self.multi_var_specs = multi_var_specs

        self.set_attrs(attrs)

        # Store operator(s)
        self.check_op(op)
        if callable(op):
            self.op = op
            self.op_lst = None
        else:
            self.op = None
            self.op_lst = op

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

    def check_op(self, op):
        """Check operator(s)."""
        try:
            n_ops = len(op)
        except TypeError:
            if not callable(op):
                raise ValueError(f"op: {type(op).__name__} not callable")
            return

        n_var_specs = len(self.multi_var_specs)
        if n_ops != n_var_specs - 1:
            raise ValueError(
                f"wrong number of operators passed in {type(self).__name__}: "
                f"{n_ops} != {n_var_specs}"
            )
        for op_i in op:
            if not callable(op_i):
                raise ValueError(f"op: {type(op_i).__name__} not callable")

    def __repr__(self):
        s = f"{type(self).__name__}(\n"

        # Variables specifications
        s += f"  var_specs: {len(self.multi_var_specs)}x\n"
        for var_specs in self.multi_var_specs:
            for line in str(var_specs).split("\n"):
                s += f"    {line}\n"

        # Operator(s)
        if self.op is not None:
            s += f"  op: {self.op.__name__}\n"
        else:
            s += f"  ops: {len(self.op_lst)}x\n"
            for op in self.op_lst:
                s += "f    {op.__name__}\n"

        s += f")"
        return s

    def __hash__(self):
        return sum([sum([hash(vs) for vs in self.multi_var_specs])])

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    # SR_TMP TODO get rid of this
    def var_specs_shared(self, key):
        """Return a varible specification, if it is shared by all."""
        assert key in ["rlat", "rlon"]  # SR_TMP restrict to only current use
        vals = [getattr(vs, key) for vs in self.multi_var_specs]
        all_equal = all(v == vals[0] for v in vals[1:])
        if not all_equal:
            raise ValueError(
                f"'{key}' differs among {len(self.multi_var_specs)} var stats: {vals}"
            )
        return next(iter(vals))
