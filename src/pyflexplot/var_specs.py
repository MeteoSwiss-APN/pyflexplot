# -*- coding: utf-8 -*-
"""
Input variable specifications.
"""
import logging as log

from srutils.dict import decompress_dict_multivals
from srutils.dict import format_dictlike
from srutils.various import isiterable

from .utils import ParentClass
from .utils import SummarizableClass
from .words import WORDS


#
# TODO
#
# - Add class ``MultiVarSpecs`` or so to represent multiple ``VarSpecs`` objects
#


def int_or_list(arg):
    try:
        iter(arg)
    except TypeError:
        return int(arg)
    else:
        return [int(a) for a in arg]


class VarSpecs(SummarizableClass, ParentClass):
    """FLEXPART input variable specifications."""

    summarizable_attrs = []  # SR_TODO

    name = None

    # Keys with respective type
    specs_type_default = {
        "species_id": (int_or_list, None),
        "integrate": (bool, None),
        "time": (int, None),
        "nageclass": (int, None),
        "numpoint": (int, None),
    }

    # SR_TMP TODO eventually eliminate
    def issubcls(self, name):
        return name in type(self).name.split(":")

    # SR_TMP TODO use to double-check independence from ParentClass
    # + @classmethod
    # + def subcls(cls, name):
    # +     raise DeprecationWarning(f"{cls.__name__}.subcls")

    def __init__(
        self, name, var_specs_dct, *, rlat=None, rlon=None, words=None, lang=None,
    ):
        """Create an instance of ``VarSpecs``.

        Args:
            name (str): Name.

            var_specs_dct (dict): Variable specification dict comprised of only
                single-object elements (see ``VarSpecs.create`` for more
                information on single- vs. multi-object elements).

            rlat (tuple, optional): Rotated latitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            rlon (tuple, optional): Rotated longitude slice parameters, passed
                to built-in ``slice``. Defaults to None.

            words (Words, optional): Word translations. Defaults to ``WORDS``.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).

        """
        assert name in type(self).name  # SR_TMP
        self.name = name

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

        self._words = words or WORDS
        self._lang = lang or "en"
        self._words.set_default_lang(self._lang)

        self._set_attrs(var_specs_dct)

    def _set_attrs(self, var_specs_dct):
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
                f"{len(var_specs_dct_todo)} unexpected arguments: "
                f"{var_specs_dct_todo}"
            )

    @classmethod
    def create(cls, name, var_specs_dct, **kwargs):
        """Create one or more instances of ``VarSpecs``.

        The values of the specification dict elements may be

            - a single value, e.g., ``{"a": 0}``;

            - a tuple of values, e.g., ``{"a": (0, 1)``; or

            - a list of values and/or value tuples, e.g., ``{"a": [0, 1]}``,
                ``{"a": [0, (1, 2)]}``, or ``{"a": [(0, 1), (2, 3)]}``.

        Both tuple and list values constitute a shorthand to specify multiple
        specification dicts at once, and are expanded into multiple dicts
        with only single-value elements by combining all tuple/list elements.

        The difference between tuple and list values is that value tuples
        specify the input for an individual plot (e.g., multiple time steps
        to integrate over), while value lists specify input for separate plots
        (e.g., multiple time steps that are plotted separately).

        After this expansion, all dicts only comprise single-value elements,
        and each is used to create a ``VarSpecs`` object.

        The ``VarSpecs`` objects are returned in a nested list, whereby the
        outer nest corresponds to the list values (separate plots), and the
        inner nest to the tuple values (separate input fields per plot).

        Args:
            name (str): Name.

            var_specs_dct (dict): Variable specification dict (see above).

            **kwargs: Additional keyword arguments used to create the
                individual ``VarSpecs`` objects.

        Returns:
            list[list[VarSpecs]]: Nested list of ``VarSpecs`` object(s).

        Examples:
            >>> f = lambda dct: VarSpecs.create("test", dct)

            >>> f({"a": 1, "b": 2})
            [[VarSpecs(a=1, b=1)]]

            >>> f({"a": (1, 2), "b": 3})
            [[VarSpecs(a=1, b=3), VarSpecs(a=2, b=3)]]

            >>> f({"a": [1, 2], "b": 3})
            [[VarSpecs(a=1, b=3)], [VarSpecs(a=2, b=3)]]

            >>> f({"a": [1, (2, 3)], "b": 4})
            [[VarSpecs(a=1, b=4)], [VarSpecs(a=2, b=4), VarSpecs(a=3, b=4)]]

            >>> f({"a": [1, 2], "b": [3, (4, 5)], "c": "hi"})
            [
                [VarSpecs(a=1, b=3, c="hi")],
                [VarSpecs(a=1, b=4, c="hi"), VarSpecs(a=1, b=5, c="hi")],
                [VarSpecs(a=2, b=3, c="hi")],
                [VarSpecs(a=2, b=4, c="hi"), VarSpecs(a=2, b=5, c="hi")],
            ]

        """
        cls = cls.subcls(name)  # SR_TMP
        var_specs_dct_lst_outer = decompress_dict_multivals(
            var_specs_dct, depth=1, cls_expand=list,
        )
        var_specs_dct_lst_lst = [
            decompress_dict_multivals(dct, depth=1, cls_expand=tuple)
            for dct in var_specs_dct_lst_outer
        ]

        def create_objs_rec(obj):
            if isinstance(obj, dict):
                return cls(name, obj, **kwargs)
            elif isinstance(obj, list):
                return [create_objs_rec(i) for i in obj]
            raise ValueError(f"obj of type {type(obj).__name__}")

        return create_objs_rec(var_specs_dct_lst_lst)

    def merge_with(self, others):

        # Words and language
        for other in others:
            if other._lang != self._lang:
                raise Exception(
                    f"merge of {other} with {self} failed: languages differ: "
                    f"{other._lang} != {self._lang}"
                )
            if other._words != self._words:
                raise Exception(
                    f"merge of {other} with {self} failed: words differ: "
                    f"{other._words} != {self._words}"
                )

        # Attributes
        var_specs_dct = {}
        for key, val0 in iter(self):
            if key in ["name", "rlat", "rlon"]:
                continue

            vals = [val0]
            for other in others:
                val = getattr(other, key)
                if val not in vals:
                    vals.append(val)

            if len(vals) == 1:
                var_specs_dct[key] = next(iter(vals))
            elif key == "deposition" and set(vals) == set(["dry", "wet"]):
                var_specs_dct[key] = tuple(vals)
            else:
                var_specs_dct[key] = vals

        return type(self)(
            self.name,
            var_specs_dct,
            rlat=self.rlat,
            rlon=self.rlon,
            words=self._words,
            lang=self._lang,
        )

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

    def var_name(self):
        """Derive variable name from specifications."""
        raise NotImplementedError(f"{type(self).__name__}.var_name")

    def dim_inds_by_name(self, *, rlat=None, rlon=None):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds["nageclass"] = self.nageclass
        inds["numpoint"] = self.numpoint

        if not self.integrate or self.time == slice(None):
            inds["time"] = self.time
        else:
            inds["time"] = slice(None, self.time + 1)

        inds["rlat"] = slice(*self.rlat)
        inds["rlon"] = slice(*self.rlon)

        return inds

    def long_name(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.long_name: override it!")

    def short_name(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.short_name: override it!")

    def var_name(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.var_name: override it!")


class VarSpecs_Concentration(VarSpecs):
    name = "concentration"

    specs_type_default = {
        **VarSpecs.specs_type_default,
        "level": (int_or_list, None),
    }

    def long_name(self):
        ctx = "abbr" if self.integrate else "*"
        return self._words["activity_concentration"].s

    def short_name(self):
        s = ""
        if self.integrate:
            return (
                f"{self._words['integrated', None, 'abbr']} "
                f"{self._words['concentration', None, 'abbr']}"
            )
        return self._words["concentration"].s

    def var_name(self):
        """Derive variable name from specifications."""

        def fmt(sid):
            return f"spec{sid:03d}"

        try:
            iter(self.species_id)
        except TypeError:
            return fmt(self.species_id)
        else:
            return [fmt(sid) for sid in self.species_id]

    def dim_inds_by_name(self, *args, **kwargs):
        """Derive indices along NetCDF dimensions."""
        inds = super().dim_inds_by_name(*args, **kwargs)
        inds["level"] = self.level
        return inds


class VarSpecs_Deposition(VarSpecs):
    name = "deposition"

    specs_type_default = {
        **VarSpecs.specs_type_default,
        "deposition": (str, None),
    }

    def __init__(self, name, var_specs_dct, *args, **kwargs):
        dep = var_specs_dct["deposition"]
        if dep not in ["wet", "dry"] and set(dep) != {"wet", "dry"}:
            raise ValueError(
                f"invalid deposition type '{dep}'", var_specs_dct,
            )
        super().__init__(name, var_specs_dct, *args, **kwargs)

    def deposition_type(self):
        type_ = self["deposition"]
        word = "total" if type_ == "tot" else type_
        return self._words[word, None, "f"].s

    def long_name(self):
        return f"{self.deposition_type()} {self._words['surface_deposition']}"

    def short_name(self):
        return self._words["deposition"].s

    def var_name(self):
        """Derive variable name from specifications."""
        prefix = {"wet": "WD", "dry": "DD"}[self.deposition]
        return f"{prefix}_spec{self.species_id:03d}"


class VarSpecs_AffectedArea(VarSpecs_Deposition):
    name = "deposition:affected_area"

    def long_name(self):
        dep_name = super().long_name()
        raise Exception(f"{type(self).__name__}.long_name")
        return f"{self._words['affected_area']} " f"({dep_name})"


class Varspecs_AffectedAreaMono(VarSpecs_AffectedArea):
    name = "deposition:affected_area:affected_area_mono"


class VarSpecs_EnsMean_Concentration(VarSpecs_Concentration):
    name = "concentration:ens_mean_concentration"

    def long_name(self):
        return (
            f"{self._words['activity_concentration']}\n"
            f"{self._words['ensemble_mean']}"
        )


class VarSpecs_EnsMean_Deposition(VarSpecs_Deposition):
    name = "deposition:ens_mean_deposition"

    def long_name(self):
        return (
            f"{self._words['ensemble_mean']} {self.deposition_type()} "
            f"{self._words['surface_deposition']}"
        )


class VarSpecs_EnsMean_AffectedArea(VarSpecs_AffectedArea):
    name = "deposition:affected_area:ens_mean_affected_area"

    def long_name(self):
        return (
            f"{self._words['ensemble_mean']} {self._words['affected_area']} "
            f"({self.deposition_type()})"
        )


class VarSpecs_EnsThrAgrmt:
    def long_name(self):
        # SR_TMP <<<
        lang = self._words.default_lang
        of = dict(en="of", de="der")[lang]
        return (
            f"{self._words['ensemble']}{dict(en=' ', de='-')[lang]}"
            f"{self._words['threshold_agreement']} {of} {super().long_name()}"
        )

    def short_name(self):
        return (
            f"{self._words['number_of', None, 'abbr'].c} "
            f"{self._words['member', None, 'pl']}"
        )


class VarSpecs_EnsThrAgrmt_Concentration(VarSpecs_EnsThrAgrmt, VarSpecs_Concentration):
    name = "concentration:ens_thr_agrmt_concentration"


class VarSpecs_EnsThrAgrmt_Deposition(VarSpecs_EnsThrAgrmt, VarSpecs_Deposition):
    name = "deposition:ens_thr_agrmt_deposition"


class VarSpecs_EnsThrAgrmt_AffectedArea(VarSpecs_EnsThrAgrmt, VarSpecs_AffectedArea):
    name = "deposition:affected_area:ens_thr_agrmt_affected_area"


class MultiVarSpecs:
    """Hold multiple ``VarSpecs`` objects."""

    def __init__(self, name, var_specs_lst):
        self.name = name
        self.var_specs_lst = var_specs_lst

    @classmethod
    def create(cls, name, var_specs_dct, *args, **kwargs):
        if var_specs_dct.get("deposition") == "tot":
            var_specs_dct["deposition"] = ("wet", "dry")
        var_specs_lst_lst = VarSpecs.create(name, var_specs_dct, *args, **kwargs)
        return [cls(name, var_specs_lst) for var_specs_lst in var_specs_lst_lst]

    def __iter__(self):
        return iter(self.var_specs_lst)

    def __len__(self):
        return len(self.var_specs_lst)
