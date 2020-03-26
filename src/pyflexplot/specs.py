# -*- coding: utf-8 -*-
"""
Input variable specifications.
"""
# Standard library
from typing import Collection
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


class FldSpecs:
    """Specifications to compute a field."""

    def __init__(self, var_setups: InputSetupCollection) -> None:
        if not var_setups:
            raise ValueError("missing var_setups", var_setups)
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
            sub_setups_lst: List[InputSetupCollection] = []
            for setup in setups:
                sub_setup_lst = setup.decompress_partially(
                    ["time"], skip=["ens_member_id"]
                )
                for sub_setup in sub_setup_lst:
                    sub_setups_lst.append(
                        sub_setup.decompress_partially(None, skip=["ens_member_id"])
                    )
            return [cls(sub_setups) for sub_setups in sub_setups_lst]

    def __repr__(self):
        s_var_setups = ",\n    ".join([str(var_setup) for var_setup in self.var_setups])
        return f"{type(self).__name__}(\n  var_setups=[\n    {s_var_setups}],\n)"

    def __eq__(self, other):
        # raise DeprecationWarning()
        try:
            var_setups_eq = self.var_setups == other.var_setups
        except AttributeError:
            raise ValueError(
                f"incomparable types {type(self).__name__} and {type(other).__name__}"
            )
        return var_setups_eq

    def decompress_partially(self, params: Collection[str]) -> List["FldSpecs"]:
        return [
            type(self)(setups)
            for setups in self.var_setups.decompress_partially(params)
        ]
