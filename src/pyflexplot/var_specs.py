# -*- coding: utf-8 -*-
"""
Input variable specifications.
"""
# Standard library
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple

# First-party
from srutils.dict import decompress_multival_dict
from srutils.dict import format_dictlike
from srutils.various import isiterable

# Local
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

    name: Optional[str] = None

    # Keys with respective type
    # SR_TMP TODO move to some config/setup class
    @property
    def specs_type_default(self) -> Mapping[str, Tuple[Any, Any]]:
        specs_type = {
            "species_id": (int_or_list, None),
            "integrate": (bool, None),
            "time": (int, None),
            "nageclass": (int, None),
            "numpoint": (int, None),
            "noutrel": (int, None),
        }
        if self._setup.variable == "concentration":
            specs_type.update({"level": (int_or_list, None)})
        elif self._setup.variable == "deposition":
            specs_type.update({"deposition": (str, None)})
        return specs_type

    def __init__(
        self, setup, var_specs_dct, *, rlat=None, rlon=None, words=None,
    ):
        """Create an instance of ``VarSpecs``.

        Args:
            setup (Setup): Plot setup.

            var_specs_dct (dict): Variable specification dict comprised of only
                single-object elements (see ``VarSpecs.???`` for more
                information on single- vs. multi-object elements).

            rlat (tuple, optional): Rotated latitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            rlon (tuple, optional): Rotated longitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            words (Words, optional): Word translations. Defaults to ``WORDS``.

        """
        self._name = setup.tmp_cls_name()  # SR_TMP

        if setup.variable == "deposition":
            deposition_type = var_specs_dct["deposition"]
            if (
                deposition_type
                and deposition_type not in ["wet", "dry"]
                and set(deposition_type) != {"wet", "dry"}
            ):
                raise ValueError(
                    f"invalid deposition type '{deposition_type}'", var_specs_dct,
                )

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

        self._setup = setup
        self._words = words or WORDS
        self._words.set_default_lang(self._setup.lang)

        # Set attributes
        var_specs_dct_todo = {k: v for k, v in var_specs_dct.items()}
        for key, (type_, default) in self.specs_type_default.items():
            try:
                val = var_specs_dct_todo.pop(key)
            except KeyError:
                val = default
            if val is not None:
                try:
                    val = type_(val)
                except TypeError:
                    raise ValueError(
                        f"argument '{key}': type '{type(val).__name__}' incompatible "
                        f"with '{type_.__name__}'"
                    )
            setattr(self, key, val)
        if var_specs_dct_todo:
            raise ValueError(
                f"{len(var_specs_dct_todo)} unexpected arguments: {var_specs_dct_todo}"
            )

    # SR_TMP <<<
    @classmethod
    def from_setup(cls, setup, var_specs_dct=None, **kwargs):

        sub_setups = setup.decompress()
        if len(sub_setups) > 1:
            var_specs_lst = []
            for sub_setup in sub_setups:
                var_specs_lst_lst_i = cls.from_setup(sub_setup, **kwargs)
                assert len(var_specs_lst_lst_i) == 1
                var_specs_lst.extend(var_specs_lst_lst_i[0])
            var_specs_lst_lst = [var_specs_lst]
            return var_specs_lst_lst

        if var_specs_dct is None:
            var_specs_dct = {
                "time": setup.time_idcs,
                "nageclass": setup.age_class_idx,
                "noutrel": setup.nout_rel_idx,
                "numpoint": setup.release_point_idx,
                "species_id": setup.species_id,
                "integrate": setup.integrate,
            }
            if setup.variable == "concentration":
                var_specs_dct["level"] = setup.level_idx
            elif setup.variable == "deposition":
                if setup.deposition_type == "tot":
                    var_specs_dct["deposition"] = ("wet", "dry")
                else:
                    var_specs_dct["deposition"] = setup.deposition_type

        var_specs_dct_lst_outer = decompress_multival_dict(
            var_specs_dct, depth=1, cls_expand=list,
        )
        var_specs_dct_lst_lst = [
            decompress_multival_dict(dct, depth=1, cls_expand=tuple)
            for dct in var_specs_dct_lst_outer
        ]

        def create_objs_rec(obj):
            if isinstance(obj, dict):
                return cls(setup, obj, **kwargs)
            elif isinstance(obj, list):
                return [create_objs_rec(i) for i in obj]
            raise ValueError(f"obj of type {type(obj).__name__}")

        var_specs_lst_lst = create_objs_rec(var_specs_dct_lst_lst)

        return var_specs_lst_lst

    # SR_TMP <<<
    @classmethod
    def from_setups(cls, setups, **kwargs):
        # SR_TMP <
        # Separate setups by time index
        orig_setups, setups = setups, []
        for setup in orig_setups:
            for time_idx in setup.time_idcs:
                setups.append(setup.derive({"time_idcs": [time_idx]}))
        # SR_TMP >
        var_specs_lst_lst = []
        for setup in setups:
            var_specs_lst_lst_tmp = cls.from_setup(setup, **kwargs)
            assert len(var_specs_lst_lst_tmp) == 1
            var_specs_lst_lst.append(next(iter(var_specs_lst_lst_tmp)))
        return var_specs_lst_lst

    def __hash__(self):
        h = 0
        for key, val in iter(self):
            if isinstance(val, slice):
                val = (val.start, val.stop, val.step)
            elif isiterable(val, str_ok=False):
                val = tuple(val)
            h += hash(val)
            h *= 10
        return h

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return format_dictlike(self)

    def __str__(self):
        return format_dictlike(self)

    def __getitem__(self, key):
        if key.startswith("_"):
            raise ValueError(f"invalid key '{key}'")
        try:
            return self.__dict__[key]
        except KeyError as e:
            raise e from None

    def __setitem__(self, key, val):
        if key.startswith("_") or key not in self.__dict__:
            raise ValueError(f"invalid key '{key}'")
        self.__dict__[key] = val

    def __iter__(self):
        for key, val in self.__dict__.items():
            if not key.startswith("_"):
                yield key, val

    def dim_inds_by_name(self, *, rlat=None, rlon=None):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds["nageclass"] = self.nageclass
        inds["numpoint"] = self.numpoint
        inds["noutrel"] = self.noutrel

        if not self.integrate or self.time == slice(None):
            inds["time"] = self.time
        else:
            inds["time"] = slice(None, self.time + 1)

        inds["rlat"] = slice(*self.rlat)
        inds["rlon"] = slice(*self.rlon)

        if self._setup.variable == "concentration":
            inds["level"] = self.level

        return inds

    # SR_TMP TODO move to some config/setup class
    def long_name(self, name=None):
        if name is None:
            name = self._name
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
        type_ = self["deposition"]
        word = {"tot": "total"}.get(type_, type_)
        return self._words[word, None, "f"].s


class MultiVarSpecs:
    """Hold multiple ``VarSpecs`` objects."""

    def __init__(self, setup, var_specs_lst):
        self.setup = setup
        self.var_specs_lst = var_specs_lst

    @classmethod
    def from_setup(cls, setup, **kwargs):
        """Create instances of ``MultiVarSpecs`` from a ``Setup`` object."""
        return [
            cls(setup, var_specs_lst)
            for var_specs_lst in VarSpecs.from_setups([setup], **kwargs)  # SR_TMP
        ]

    @classmethod
    def from_setups(cls, setups, **kwargs):
        return [obj for setup in setups for obj in cls.from_setup(setup, **kwargs)]

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
