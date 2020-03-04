# -*- coding: utf-8 -*-
"""
Input variable specifications.
"""
# First-party
from srutils.dict import format_dictlike

# Local
from .setup import Setup
from .utils import summarizable
from .words import WORDS


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
        self._words = WORDS
        self._words.set_default_lang(self._setup.lang)

    # SR_TMP <<<
    @property
    def rlat(self):
        return (None,)

    # SR_TMP <<<
    @property
    def rlon(self):
        return (None,)

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
        dct = {
            "integrate": self._setup.integrate,
            "nageclass": self._setup.age_class_idx,
            "noutrel": self._setup.nout_rel_idx,
            "numpoint": self._setup.release_point_idx,
            "rlat": (None,),
            "rlon": (None,),
            "species_id": self._setup.species_id,
            "time": next(iter(self._setup.time_idcs)),  # SR_TMP
        }
        if self._setup.variable == "concentration":  # SR_TMP
            dct["level"] = self._setup.level_idx
        if self._setup.variable == "deposition":  # SR_TMP
            dct["deposition"] = self._setup.deposition_type
        return dct

    def dim_inds_by_name(self, time_all=True):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds["nageclass"] = self._setup.age_class_idx
        inds["numpoint"] = self._setup.release_point_idx
        inds["noutrel"] = self._setup.nout_rel_idx

        # SR_TMP <
        if time_all:
            inds["time"] = slice(None)
        else:
            raise NotImplementedError("time_all == False")
        # SR_TMP >

        inds["rlat"] = slice(None)
        inds["rlon"] = slice(None)

        if self._setup.variable == "concentration":
            inds["level"] = self._setup.level_idx

        return inds

    # SR_TMP TODO move to some config/setup class
    def long_name(self, *, variable=None, plot_type=None):
        if (variable, plot_type) == (None, None):
            variable = self._setup.variable
            plot_type = self._setup.plot_type
        w = self._words
        dep = self.deposition_type_word()
        if plot_type in ["affected_area", "affected_area_mono"]:
            super_name = self.long_name(variable="deposition")
            return f"{w['affected_area']} ({super_name})"
        elif plot_type == "ens_thr_agrmt":
            super_name = self.short_name(variable="deposition")
            return f"{w['threshold_agreement']} ({super_name})"
        elif plot_type == "ens_cloud_arrival_time":
            return f"{w['cloud_arrival_time']}"
        if variable == "deposition":
            if plot_type == "ens_min":
                return f"{w['ensemble_minimum']} {dep} {w['surface_deposition']}"
            elif plot_type == "ens_max":
                return f"{w['ensemble_maximum']} {dep} {w['surface_deposition']}"
            elif plot_type == "ens_median":
                return f"{w['ensemble_median']} {dep} {w['surface_deposition']}"
            elif plot_type == "ens_mean":
                return f"{w['ensemble_mean']} {dep} {w['surface_deposition']}"
            else:
                return f"{dep} {w['surface_deposition']}"
        if variable == "concentration":
            if plot_type == "ens_min":
                return f"{w['ensemble_minimum']} {w['concentration']}"
            elif plot_type == "ens_max":
                return f"{w['ensemble_maximum']} {w['concentration']}"
            elif plot_type == "ens_median":
                return f"{w['ensemble_median']} {w['concentration']}"
            elif plot_type == "ens_mean":
                return f"{w['ensemble_mean']} {w['concentration']}"
            else:
                ctx = "abbr" if self._setup.integrate else "*"
                return w["activity_concentration", None, ctx].s
        raise NotImplementedError(
            f"long_name for variable '{variable}' and plot_type '{plot_type}'"
        )

    def short_name(self, *, variable="", plot_type=""):
        if variable + plot_type == "":
            variable = self._setup.variable
            plot_type = self._setup.plot_type
        w = self._words
        if variable == "concentration":
            if plot_type == "ens_cloud_arrival_time":
                return f"{w['arrival'].c} ({w['hour', None, 'pl']}??)"
            else:
                if self._setup.integrate:
                    return (
                        f"{w['integrated', None, 'abbr']} "
                        f"{w['concentration', None, 'abbr']}"
                    )
                return w["concentration"].s
        if variable == "deposition":
            if plot_type == "ens_thr_agrmt":
                return f"{w['number_of', None, 'abbr'].c} {w['member', None, 'pl']}"
            else:
                return w["deposition"].s
        raise NotImplementedError(
            f"short_name for variable '{variable}' and plot_type '{plot_type}'"
        )

    def var_name(self, *args, **kwargs):
        if self._setup.variable == "concentration":
            if isinstance(self._setup.species_id, int):
                return f"spec{self._setup.species_id:03d}"
            return [f"spec{sid:03d}" for sid in self._setup.species_id]
        elif self._setup.variable == "deposition":
            prefix = {"wet": "WD", "dry": "DD"}[self._setup.deposition_type]
            return f"{prefix}_spec{self._setup.species_id:03d}"
        raise NotImplementedError(f"{type(self).__name__}.var_name: override it!")

    def deposition_type_word(self):
        if self._setup.variable == "deposition":
            type_ = self._setup.deposition_type
            word = {"tot": "total"}.get(type_, type_)
            return self._words[word, None, "f"].s
        return "none"


class MultiVarSpecs:
    """Hold multiple ``VarSpecs`` objects."""

    def __init__(self, setup, var_specs_lst):
        self.setup = setup
        self.var_specs_lst = var_specs_lst

    @classmethod
    def create(cls, setup_or_setups):
        """Create instances of ``MultiVarSpecs`` from ``Setup`` object(s)."""
        if isinstance(setup_or_setups, Setup):
            setup = setup_or_setups
            multi_var_specs_lst = []
            var_specs_lst_lst = VarSpecs.create_many([setup], pre_expand_time=True)
            for var_specs_lst in var_specs_lst_lst:
                multi_var_specs_lst.append(cls(setup, var_specs_lst))
            return multi_var_specs_lst
        else:
            setups = setup_or_setups
        return [obj for setup in setups for obj in cls.create(setup)]

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
                self.setup == other.setup and self.var_specs_lst == other.var_specs_lst
            )
        return False

    def __iter__(self):
        return iter(self.var_specs_lst)

    def __len__(self):
        return len(self.var_specs_lst)

    def collect(self, param):
        """Collect all values of a given parameter."""
        if param in ["rlat", "rlon"]:
            return [getattr(vs, param) for vs in self]
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
