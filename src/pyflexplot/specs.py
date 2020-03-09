# -*- coding: utf-8 -*-
"""
Input variable specifications.
"""
# First-party
from srutils.dict import format_dictlike

# Local
from .setup import Setup
from .utils import summarizable


def int_or_list(arg):
    try:
        iter(arg)
    except TypeError:
        return int(arg)
    else:
        return [int(a) for a in arg]


@summarizable
class VarSpecs:
    """FLEXPART input variable specifications."""

    def __init__(self, setup):
        """Create an instance of ``VarSpecs``."""
        self._setup = setup

    # SR_TMP <<<
    @property
    def time(self):
        assert len(self._setup.time_idcs) == 1
        return next(iter(self._setup.time_idcs))

    @classmethod
    def create_many(cls, setups, pre_expand_time=False):
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
        return self.dict() == other.dict()

    def __repr__(self):
        return format_dictlike(self)

    def dict(self):
        assert len(self._setup.time_idcs) == 1  # SR_TMP
        return {
            "deposition": self._setup.deposition_type,
            "integrate": self._setup.integrate,
            "level": self._setup.level_idx,
            "nageclass": self._setup.age_class_idx,
            "noutrel": self._setup.nout_rel_idx,
            "numpoint": self._setup.release_point_idx,
            "species_id": self._setup.species_id,
            "time": next(iter(self._setup.time_idcs)),  # SR_TMP
        }


class FldSpecs:
    """Hold multiple ``VarSpecs`` objects."""

    def __init__(self, setup, var_specs_lst):
        self.setup = setup
        self._var_specs_lst = var_specs_lst

    @classmethod
    def create(cls, setup_or_setups):
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
        s_specs = ",\n    ".join([str(specs) for specs in self])
        return (
            f"{type(self).__name__}("
            f"\n  setup={s_setup},"
            f"\n  var_specs_lst=[\n    {s_specs}],"
            f"\n)"
        )

    def __eq__(self, other):
        if isinstance(other, type(self)) or isinstance(self, type(other)):
            return (
                self.setup == other.setup
                and self._var_specs_lst == other._var_specs_lst
            )
        return False

    def __iter__(self):
        return iter(self._var_specs_lst)

    def __len__(self):
        return len(self._var_specs_lst)

    def collect(self, param):
        """Collect all values of a given parameter."""
        if param == "time":
            setup_param = "time_idcs"
            return [next(iter(getattr(vs._setup, setup_param))) for vs in self]
        setup_param = {
            "species_id": "species_id",
            "integrate": "integrate",
            "nageclass": "age_class_idx",
            "numpoint": "release_point_idx",
            "noutrel": "nout_rel_idx",
            "level": "level_idx",
            "deposition": "deposition_type",
        }.get(param, param)
        return [getattr(vs._setup, setup_param) for vs in self]

    def collect_equal(self, param):
        """Obtain the value of a param, expecting it to be equal for all."""
        values = self.collect(param)
        if not all(value == values[0] for value in values[1:]):
            raise Exception("values differ for param", param, values)
        return next(iter(values))
