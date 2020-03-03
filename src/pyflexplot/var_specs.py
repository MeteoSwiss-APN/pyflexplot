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

        self._name = setup.tmp_cls_name()  # SR_TMP

        self.rlat = (None,)
        self.rlon = (None,)

        self._setup = setup
        self._words = WORDS
        self._words.set_default_lang(self._setup.lang)

        # SR_TMP < TODO eliminate
        self.species_id = setup.species_id
        self.integrate = setup.integrate
        self.nageclass = setup.age_class_idx
        self.numpoint = setup.release_point_idx
        self.noutrel = setup.nout_rel_idx
        if setup.variable == "concentration":  # SR_TMP
            self.level = setup.level_idx
        if setup.variable == "deposition":  # SR_TMP
            self.deposition = setup.deposition_type
        # SR_TMP >

    # SR_TMP <<<
    @property
    def time(self):
        assert len(self._setup.time_idcs) == 1
        return next(iter(self._setup.time_idcs))

    @classmethod
    def create_many(cls, setups, pre_expand_time=False):
        if not pre_expand_time:
            var_specs_lst = []
            for setup in setups:
                for sub_setup in setup.decompress():
                    var_specs_lst.append(cls(sub_setup))
            return var_specs_lst
        else:
            var_specs_lst_lst = []
            for setup in setups:
                for sub_setup in setup.decompress(["time_idcs"]):
                    var_specs_lst_lst.append([])
                    for sub_sub_setup in sub_setup.decompress():
                        var_specs_lst_lst[-1].append(cls(sub_sub_setup))
            return var_specs_lst_lst

    def __eq__(self, other):
        return self.dict() == other.dict()

    def __repr__(self):
        return format_dictlike(self)

    def dict(self):
        dct = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            # SR_TMP <
            if name not in list(self.__dict__) + ["time"]:
                continue
            # SR_TMP >
            dct[name] = getattr(self, name)
        return dct

    def dim_inds_by_name(self, time_all=True):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds["nageclass"] = self.nageclass
        inds["numpoint"] = self.numpoint
        inds["noutrel"] = self.noutrel

        # SR_TMP <
        if time_all:
            inds["time"] = slice(None)
        else:
            raise NotImplementedError("time_all == False")
        # SR_TMP >

        inds["rlat"] = slice(*self.rlat)
        inds["rlon"] = slice(*self.rlon)

        if self._setup.variable == "concentration":
            inds["level"] = self.level

        return inds

    # SR_TMP TODO move to some config/setup class
    def long_name(self, name=None):
        if name is None:
            name = self._name
        if ":" in name:
            # raise DeprecationWarning('":" in name')  # SR_TODO
            name = name.split(":")[-1]
        if name == "ens_max_affected_area":
            return (
                f"{self._words['ensemble_maximum']} {self._words['affected_area']} "
                f"({self.deposition_type()})"
            )
        if name == "ens_min_affected_area":
            return (
                f"{self._words['ensemble_minimum']} {self._words['affected_area']} "
                f"({self.deposition_type()})"
            )
        if name == "ens_median_affected_area":
            return (
                f"{self._words['ensemble_median']} {self._words['affected_area']} "
                f"({self.deposition_type()})"
            )
        if name == "ens_mean_affected_area":
            return (
                f"{self._words['ensemble_mean']} {self._words['affected_area']} "
                f"({self.deposition_type()})"
            )
        if name == "ens_max_deposition":
            return (
                f"{self._words['ensemble_maximum']} "
                f"{self.deposition_type()} {self._words['surface_deposition']}"
            )
        if name == "ens_max_concentration":
            return f"{self._words['ensemble_maximum']} {self._words['concentration']}"
        if name == "ens_min_deposition":
            return (
                f"{self._words['ensemble_minimum']} "
                f"{self.deposition_type()} {self._words['surface_deposition']}"
            )
        if name == "ens_min_concentration":
            return f"{self._words['ensemble_minimum']} {self._words['concentration']}"
        if name == "ens_median_deposition":
            return (
                f"{self._words['ensemble_median']} "
                f"{self.deposition_type()} {self._words['surface_deposition']}"
            )
        if name == "ens_median_concentration":
            return f"{self._words['ensemble_median']} {self._words['concentration']}"
        if name == "ens_mean_deposition":
            return (
                f"{self._words['ensemble_mean']} "
                f"{self.deposition_type()} {self._words['surface_deposition']}"
            )
        if name == "ens_mean_concentration":
            return f"{self._words['ensemble_mean']} {self._words['concentration']}"
        if name.startswith("affected_area"):
            dep_name = self.long_name("deposition")
            return f"{self._words['affected_area']} " f"({dep_name})"
        if name.startswith("ens_thr_agrmt"):
            super_name = self.short_name(self._setup.variable)
            return f"{self._words['threshold_agreement']} ({super_name})"
        if name.startswith("ens_cloud_arrival_time"):
            return f"{self._words['cloud_arrival_time']}"
        if name == "concentration":
            ctx = "abbr" if self.integrate else "*"
            return self._words["activity_concentration", None, ctx].s
        if name == "deposition":
            return f"{self.deposition_type()} {self._words['surface_deposition']}"
        raise NotImplementedError(f"{type(self).__name__}.long_name")

    def short_name(self, name=None):
        if name is None:
            name = self._name
        if ":" in name:
            # raise DeprecationWarning('":" in name')  # SR_TODO
            name = name.split(":")[-1]
        if name == "ens_cloud_arrival_time_concentration":
            return (
                # f"{self._words['arrival_time'].c}\n"
                # f"({self._words['hour', None, 'pl'].c} {self._words['from_now']})"
                f"{self._words['arrival'].c} "
                f"({self._words['hour', None, 'pl']}??)"
            )
        if name.startswith("ens_thr_agrmt"):
            return (
                f"{self._words['number_of', None, 'abbr'].c} "
                f"{self._words['member', None, 'pl']}"
            )
        if name.endswith("deposition"):
            return self._words["deposition"].s
        if name.endswith("concentration"):
            if self.integrate:
                return (
                    f"{self._words['integrated', None, 'abbr']} "
                    f"{self._words['concentration', None, 'abbr']}"
                )
            return self._words["concentration"].s
        raise NotImplementedError(f"{type(self).__name__}.short_name")

    def var_name(self, *args, **kwargs):
        if self._setup.variable == "concentration":
            try:
                iter(self.species_id)
            except TypeError:
                return f"spec{self.species_id:03d}"
            else:
                return [f"spec{sid:03d}" for sid in self.species_id]
        elif self._setup.variable == "deposition":
            prefix = {"wet": "WD", "dry": "DD"}[self.deposition]
            return f"{prefix}_spec{self.species_id:03d}"
        raise NotImplementedError(f"{type(self).__name__}.var_name: override it!")

    def deposition_type(self):
        if self._setup.variable != "deposition":
            raise Exception(f"unexpected var specs type", type(self))
        type_ = self.deposition
        word = {"tot": "total"}.get(type_, type_)
        return self._words[word, None, "f"].s


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

    def collect(self, key):
        """Collect all values for a given key."""
        try:
            return [getattr(vs, key) for vs in self]
        except AttributeError:
            raise ValueError("invalid key", key)

    def collect_equal(self, key):
        """Obtain the value for a key, expecting it to be equal for all."""
        values = self.collect(key)
        if not all(value == values[0] for value in values[1:]):
            raise Exception("values differ for key", key, values)
        return next(iter(values))
