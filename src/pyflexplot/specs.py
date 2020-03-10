# -*- coding: utf-8 -*-
"""
Input variable specifications.
"""
# Standard library
from typing import Any
from typing import List
from typing import Sequence
from typing import Union
from typing import overload

# First-party
from srutils.dict import format_dictlike

# Local
from .setup import Setup
from .utils import summarizable


@overload
def int_or_list(arg: Union[int, float]) -> int:
    ...


@overload
def int_or_list(arg: Sequence[Union[int, float]]) -> List[int]:
    ...


def int_or_list(
    arg: Union[Union[int, float], Sequence[Union[int, float]]]
) -> Union[int, List[int]]:
    if isinstance(arg, Sequence):
        return [int(a) for a in arg]
    else:
        return int(arg)


@summarizable
class VarSpecs:
    """FLEXPART input variable specifications."""

    def __init__(self, setup: Setup) -> None:
        """Create an instance of ``VarSpecs``."""
        self._setup = setup

    @classmethod
    def create_many(
        cls, setups: Sequence[Setup], pre_expand_time: bool = False,
    ):
        def create_var_specs_lst_lst(setups):
            var_specs_lst_lst = []
            for setup in setups:
                var_specs_lst = []
                for sub_setup in setup.decompress():
                    var_specs_lst.append(cls(sub_setup))
                var_specs_lst_lst.append(var_specs_lst)
            return var_specs_lst_lst

        def flatten(lst_lst):
            return [obj for lst in lst_lst for obj in lst]

        if not pre_expand_time:
            return flatten(create_var_specs_lst_lst(setups))

        setups_time = [
            sub_setup
            for setup in setups
            for sub_setup in setup.decompress(["time_idcs"])
        ]
        return create_var_specs_lst_lst(setups_time)

    def __eq__(self, other):
        return self._setup == other._setup

    def __repr__(self):
        return format_dictlike(self)

    def dict(self):
        time = (
            self._setup.time_idcs
            if len(self._setup.time_idcs) != 1
            else next(iter(self._setup.time_idcs))
        )  # SR_TMP
        return {
            "deposition": self._setup.deposition_type,
            "integrate": self._setup.integrate,
            "level": self._setup.level_idx,
            "nageclass": self._setup.age_class_idx,
            "noutrel": self._setup.nout_rel_idx,
            "numpoint": self._setup.release_point_idx,
            "species_id": self._setup.species_id,
            "time": time,
        }


class FldSpecs:
    """Hold multiple ``VarSpecs`` objects."""

    def __init__(self, setup: Setup, var_specs_lst: Sequence[VarSpecs]) -> None:
        self.setup = setup
        self.var_specs_lst = var_specs_lst

    @classmethod
    def create(cls, setup_or_setups: Union[Setup, Sequence[Setup]]) -> List["FldSpecs"]:
        """Create instances of ``FldSpecs`` from ``Setup`` object(s)."""
        if not isinstance(setup_or_setups, Setup):
            return [obj for setup in setup_or_setups for obj in cls.create(setup)]
        else:
            setup = setup_or_setups
            var_specs_lst_lst = VarSpecs.create_many([setup], pre_expand_time=True)
            fld_specs_lst = []
            for var_specs_lst in var_specs_lst_lst:
                obj = cls(setup, var_specs_lst)
                fld_specs_lst.append(obj)
            return fld_specs_lst

    def __repr__(self):
        s_setup = "\n  ".join(repr(self.setup).split("\n"))
        s_specs = ",\n    ".join([str(specs) for specs in self.var_specs_lst])
        return (
            f"{type(self).__name__}("
            f"\n  setup={s_setup},"
            f"\n  var_specs_lst=[\n    {s_specs}],"
            f"\n)"
        )

    def __eq__(self, other):
        # raise DeprecationWarning()
        try:
            var_specs_eq = self.var_specs_lst == other.var_specs_lst
            setup_eq = self.setup == other.setup
        except AttributeError:
            raise ValueError(
                f"incomparable types: {type(self).__name__}, {type(other).__name__}"
            )
        # # Note: var_specs_eq may not equal setup_eq for timeless var_specs,
        # # that is, when the latter lack a meaningful time index (-999)
        # assert var_specs_eq == setup_eq
        return var_specs_eq and setup_eq

    def collect(self, param: str) -> List[Any]:
        """Collect all values of a given parameter."""
        if param == "time":
            setup_param = "time_idcs"
            return [
                next(iter(getattr(vs._setup, setup_param))) for vs in self.var_specs_lst
            ]
        setup_param = {
            "species_id": "species_id",
            "integrate": "integrate",
            "nageclass": "age_class_idx",
            "numpoint": "release_point_idx",
            "noutrel": "nout_rel_idx",
            "level": "level_idx",
            "deposition": "deposition_type",
        }.get(param, param)
        return [getattr(vs._setup, setup_param) for vs in self.var_specs_lst]

    def collect_equal(self, param: str) -> Any:
        """Obtain the value of a param, expecting it to be equal for all."""
        values = self.collect(param)
        if not all(value == values[0] for value in values[1:]):
            raise Exception("values differ for param", param, values)
        return next(iter(values))
