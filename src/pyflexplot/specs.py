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

# Local
from .setup import InputSetup
from .setup import InputSetupCollection


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


# SR_TMP <<< Leftover from VarSpecs
def create_var_setups(fld_setups: InputSetupCollection):
    var_setups_lst = []
    for fld_setup in fld_setups:
        # SR_TMP <
        # for fld_sub_setup in fld_setup.decompress_partially(["time"]):
        #     var_setups_lst.append(fld_sub_setup.decompress())
        assert len(fld_setup.infile) == 1  # SR_TMP
        for fld_sub_setup in fld_setup.decompress_partially(
            ["time"], skip=["ens_member_id"]
        ):
            assert len(fld_sub_setup.infile) == 1  # SR_TMP
            var_setups_lst.append(
                fld_sub_setup.decompress_partially(None, skip=["ens_member_id"])
            )
            # SR_TMP >
        # SR_TMP >
    return var_setups_lst


class FldSpecs:
    """Specifications to compute a field."""

    def __init__(self, fld_setup: InputSetup, var_setups: Sequence[InputSetup]) -> None:
        self.fld_setup = fld_setup
        self.var_setups = var_setups

    @classmethod
    def create(
        cls, setup_or_setups: Union[InputSetup, InputSetupCollection],
    ) -> List["FldSpecs"]:
        """Create instances of ``FldSpecs`` from ``InputSetup`` object(s)."""
        if isinstance(setup_or_setups, InputSetupCollection):
            setups = setup_or_setups
        elif isinstance(setup_or_setups, InputSetup):
            setups = InputSetupCollection([setup_or_setups])
        else:
            raise ValueError(
                "setup_or_setups has invalid type",
                type(setup_or_setups),
                setup_or_setups,
            )
        if len(setups) > 1:
            return [obj for setup in setups for obj in cls.create(setup)]
        else:
            fld_setup = next(iter(setups))
            var_setups_lst = create_var_setups(setups)
            fld_specs_lst = []
            for var_setups in var_setups_lst:
                fld_specs = cls(fld_setup, var_setups)
                fld_specs_lst.append(fld_specs)
            return fld_specs_lst

    def __repr__(self):
        s_fld_setup = "\n  ".join(repr(self.fld_setup).split("\n"))
        s_var_setups = ",\n    ".join([str(var_setup) for var_setup in self.var_setups])
        return (
            f"{type(self).__name__}("
            f"\n  fld_setup={s_fld_setup},"
            f"\n  var_setups=[\n    {s_var_setups}],"
            f"\n)"
        )

    def __eq__(self, other):
        # raise DeprecationWarning()
        try:
            var_setups_eq = self.var_setups == other.var_setups
            fld_setup_eq = self.fld_setup == other.fld_setup
        except AttributeError:
            raise ValueError(
                f"incomparable types: {type(self).__name__}, {type(other).__name__}"
            )
        # Note: ``var_setups_eq`` may not equal ``fld_setup_eq`` for timeless
        # ``var_setups``, i.e., if the latter sport a dummy time index (-999)!
        # assert var_setups_eq == setup_eq
        return var_setups_eq and fld_setup_eq

    def collect(self, param: str) -> List[Any]:
        """Collect all values of a given parameter."""
        setup_param = "deposition" if param == "deposition_type" else param
        return [getattr(var_setup, setup_param) for var_setup in self.var_setups]

    def collect_equal(self, param: str) -> Any:
        """Obtain the value of a param, expecting it to be equal for all."""
        values = self.collect(param)
        if not all(value == values[0] for value in values[1:]):
            raise Exception("values differ for param", param, values)
        return next(iter(values))
