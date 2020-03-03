# -*- coding: utf-8 -*-
"""
Field specifications.
"""
# Standard library
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

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
        self,
        name: str,
        multi_var_specs: MultiVarSpecs,
        *,
        op: Union[Callable, List[Callable]] = np.nansum,
    ):
        """Create an instance of ``FieldSpecs``.

        Args:
            name: Name.

            multi_var_specs (MultiVarSpecs): Specifications of one or more
                input variables used to subsequently create a plot field.

        Kwargs:
            op: Opterator(s) used to combine input fields read based on
                ``multi_var_specs``. Must accept argument ``axis=0`` to only
                reduce along over the fields.

                If a single operator is passed, it is used to sequentially
                combine one field after the other, in the same order as the
                corresponding specifications (``multi_var_specs``).

                If a list of operators has been passed, then it's length must
                be one smaller than that of ``multi_var_specs``, such that each
                operator is used between two subsequent fields (again in the
                same order as the corresponding specifications).

        """
        self.name = name
        self.multi_var_specs = multi_var_specs

        # Store operator(s)
        self._op: Optional[Callable]
        self._op_lst: Optional[Sequence[Callable]]
        if callable(op):
            self._op = op
            self._op_lst = None
        else:
            self._op = None
            self._op_lst = op

    def __repr__(self):
        s = f"{type(self).__name__}(\n"

        # Variables specifications
        s += f"  var_specs: {len(self.multi_var_specs)}x\n"
        for var_specs in self.multi_var_specs:
            for line in str(var_specs).split("\n"):
                s += f"    {line}\n"

        # Operator(s)
        if self._op is not None:
            s += f"  op: {self._op.__name__}\n"
        else:
            s += f"  ops: {len(self._op_lst)}x\n"
            for op in self._op_lst:
                s += "f    {op.__name__}\n"

        s += f")"
        return s

    def __hash__(self):
        return sum([sum([hash(vs) for vs in self.multi_var_specs])])

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def merge_fields(self, flds):
        """Merge fields by applying a single operator or an operator chain."""
        if self._op is not None:
            fld = self._op(flds, axis=0)
        elif self._op_lst is not None:
            if not len(flds) == len(self._op_lst) + 1:
                raise ValueError(
                    "wrong number of fields", len(flds), len(self._op_lst) + 1,
                )
            fld = flds[0]
            for i, fld_i in enumerate(flds[1:]):
                _op = self._op_lst[i]
                fld = _op([fld, fld_i], axis=0)
        else:
            raise Exception("no operator(s) defined")
        return fld
