# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot setup and setup files.

The setup parameters that are exposed in the setup files are all described in
the docstring of the class method ``Setup.create``.

"""
# Standard library
import dataclasses as dc
import re
from copy import deepcopy
from pprint import pformat
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
from typing_extensions import Literal

# First-party
from srutils.dataclasses import cast_field_value
from srutils.dict import compress_multival_dicts
from srutils.dict import decompress_multival_dict
from srutils.dict import merge_dicts
from srutils.exceptions import InvalidParameterNameError
from srutils.format import nested_repr
from srutils.format import sfmt

# Local
from ..utils.exceptions import UnequalSetupParamValuesError
from .dimensions import Dimensions
from .dimensions import is_dimensions_param
from .model_setup import is_model_setup_param
from .model_setup import ModelSetup
from .plot_panel_setup import is_plot_panel_setup_param
from .plot_panel_setup import PlotPanelSetup
from .plot_panel_setup import PlotPanelSetupGroup
from .plot_panel_setup import PlotPanelSetupGroupFormatter


# SR_TMP <<< TODO cleaner solution
def is_plot_setup_param(param: str) -> bool:
    return param in PlotSetup.get_params()


# SR_TMP <<< TODO cleaner solution
def get_setup_param_value(setup: "PlotSetup", param: str) -> Any:
    if is_plot_setup_param(param):
        return getattr(setup, param)
    elif is_plot_panel_setup_param(param):
        # SR_TMP <
        # return getattr(setup.panels, param)
        return setup.panels.collect_equal(param)
        # SR_TMP >
    elif is_model_setup_param(param):
        return getattr(setup.model, param.replace("model.", ""))
    elif is_dimensions_param(param):
        return setup.panels.collect_equal(param.replace("dimensions.", ""))
    raise ValueError("invalid input setup parameter", param)


# SR_TODO Clean up docstring -- where should format key hints go?
@dc.dataclass
class PlotSetup:
    """Setup of a whole plot.

    See docstring of ``Setup.create`` for details on parameters.

    Note that until all the dimensions have been completed etc., a ``PlotSetup``
    object may ultimately correspond to multiple plots, but at plotting time,
    each ``PlotSetup`` object corresponds to one plot (which may be multiple
    times with different names/formats, though).

    """

    infile: str  # = "none"
    outfile: Union[str, Tuple[str, ...]]  # = "none"
    outfile_time_format: str = "%Y%m%d%H%M"
    scale_fact: float = 1.0
    plot_type: str = "auto"
    multipanel_param: Optional[str] = None
    model: ModelSetup = ModelSetup()
    panels: PlotPanelSetupGroup = PlotPanelSetupGroup([PlotPanelSetup()])

    def __post_init__(self) -> None:

        # Check plot_type
        choices = ["auto", "multipanel"]
        assert self.plot_type in choices, self.plot_type

        # Check multipanel_param
        multipanel_param_choices = ["ens_variable"]
        n_panel_choices = [4]
        n_panels = len(self.panels)
        if self.multipanel_param is None:
            if n_panels != 1:
                raise ValueError(
                    f"number of panels ({n_panels}) must be 1  for single-panel plots"
                    "(multipanel_param is None)"
                )
        elif n_panels not in n_panel_choices:
            raise ValueError(
                f"wrong number of panels ({n_panels}); supported: {n_panel_choices}"
                f" (multipanel_param '{self.multipanel_param}')"
            )
        elif self.multipanel_param not in multipanel_param_choices:
            raise NotImplementedError(
                f"unknown multipanel_param '{self.multipanel_param}'"
                f"; choices: {', '.join(multipanel_param_choices)}"
            )

        # Check model
        if not isinstance(self.model, ModelSetup):
            raise ValueError(
                "'model' has wrong type: expected ModelSetup, got "
                + type(self.model).__name__
            )

        # Check panels
        if not isinstance(self.panels, PlotPanelSetupGroup):
            raise ValueError(
                "'panels' has wrong type: expected PlotPanelSetupGroup, got "
                + type(self.panels).__name__
            )

    @overload
    def decompress(
        self,
        select: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
        *,
        internal: Literal[True] = True,
    ) -> "PlotSetupGroup":
        ...

    @overload
    def decompress(
        self,
        select: Optional[Collection[str]] = None,
        skip: Optional[Collection[str]] = None,
        *,
        internal: Literal[False],
    ) -> List["PlotSetup"]:
        ...

    # pylint: disable=R0912  # too-many-branches (>12)
    # pylint: disable=R0914  # too-many-locals (>15)
    # pylint: disable=R0915  # too-many-statements (>50)
    def decompress(self, select=None, skip=None, *, internal=True):
        """Create a setup object for each combination of list parameter values.

        Note that the parameter 'model.ens_member_id' (or 'ens_member_id') can
        only be expanded if ``internal`` is false. (This is not reflected in
        the function signature because it would be overly complicated.)

        Args:
            select: Names of parameter to select; all others are skipped;
                names in both ``select`` and ``skip`` will be skipped.

            skip (optional): Names of parameters to skip; if their values are
                list of multiple values, those are retained as such; names in
                both ``select`` and ``skip`` will be skipped.

            internal (optional): Decompress setup group internally and return
                one group containing the decompressed setup objectss; otherwise,
                a separate group is returned for each decompressed setup object.

        """
        if internal:
            if (select, skip) == (None, None):
                raise ValueError(
                    "cannot fully decompress with internal=True; skip"
                    " 'model.ens_member_id' or pass internal=False"
                )
            selected = select and (
                "model.ens_member_id" in select or "ens_member_id" in select
            )
            skipped = "model.ens_member_id" in (skip or []) or "ens_member_id" in (
                skip or []
            )
            if (not select or selected) and not skipped:
                raise ValueError(
                    "cannot decompress 'model.ens_member_id' with internal=True;"
                    " skip 'model.ens_member_id' or pass internal=False"
                )

        def group_params(
            params: Optional[Collection[str]],
        ) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
            model: Optional[List[str]] = None
            panel: Optional[List[str]] = None
            other: Optional[List[str]] = None
            for param in params or []:
                if is_model_setup_param(param):
                    if param.startswith("model."):
                        param = param.replace("model.", "")
                    if model is None:
                        model = []
                    model.append(param)
                elif is_plot_panel_setup_param(param) or is_dimensions_param(param):
                    if param.startswith("dimensions."):
                        param = param.replace("dimensions.", "")
                    if panel is None:
                        panel = []
                    panel.append(param)
                else:
                    if other is None:
                        other = []
                    other.append(param)
            return model, panel, other

        # Group select and skip parameters
        select_model, select_panel, select_outer = group_params(select)
        skip_model, skip_panel, skip_outer = group_params(skip)
        unrestricted = select is None and skip is None
        decompress_model = (
            unrestricted or select_model is not None or skip_model is not None
        )
        decompress_panel = (
            unrestricted or select_panel is not None or skip_panel is not None
        )
        decompress_outer = (
            unrestricted or select_outer is not None or skip_outer is not None
        )

        # PlotSetup.model
        model_dct = self.dict().pop("model")
        if not decompress_model:
            model_dcts = [model_dct]
        else:
            model_dcts = decompress_multival_dict(model_dct, select_model, skip_model)

        # PlotSetup.panels
        panel_dcts: List[Dict[str, Any]]
        if not decompress_panel:
            panel_dcts = self.panels.dicts()
        else:
            panel_dcts = []
            for panels in self.panels.decompress(
                select_panel, skip_panel, internal=False
            ):
                panel_dcts.extend(panels.dicts())

        # PlotSetup.*
        outer_dct = self.dict()
        if not decompress_outer:
            outer_dcts = [outer_dct]
        else:
            outer_dct.pop("panels")
            outer_dcts = decompress_multival_dict(outer_dct, select_outer, skip_outer)

        # Merge expanded dicts
        dcts: List[Dict[str, Any]] = []
        for outer_dct_i in outer_dcts:
            for model_dct in model_dcts:
                for panel_dct in panel_dcts:
                    dcts.append(
                        {
                            **deepcopy(outer_dct_i),
                            "model": deepcopy(model_dct),
                            "panels": [deepcopy(panel_dct)],
                        }
                    )

        setups: List[PlotSetup] = []
        for dct in dcts:
            setups.append(type(self).create(dct))

        if internal:
            return PlotSetupGroup(setups)
        return setups

    @overload
    def derive(self, params: Mapping[str, Any]) -> "PlotSetup":
        ...

    @overload
    def derive(self, params: Sequence[Mapping[str, Any]]) -> "PlotSetupGroup":
        ...

    def derive(
        self, params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> Union["PlotSetup", "PlotSetupGroup"]:
        """Derive ``Setup`` object(s) with adapted parameters."""
        if isinstance(params, Sequence):
            if not params:
                setup_lst = [self.copy()]
            else:
                setup_lst = [self.derive(sub_params) for sub_params in params]
            return PlotSetupGroup(setup_lst)
        elif isinstance(params, Mapping):
            dct = merge_dicts(self.dict(), params, overwrite_seqs=True)
            return type(self).create(dct)
        else:
            raise ValueError(
                f"params must be sequence or mapping, not {type(params).__name__}"
            )

    def dict(self, rec: bool = True) -> Dict[str, Any]:
        """Return the parameter names and values as a dict.

        Args:
            rec (optional): Recursively return sub-objects like ``CoreSetup`` as
                dicts.

        """
        return {
            **dc.asdict(self),
            "model": dc.asdict(self.model) if rec else self.model,
            "panels": self.panels.dicts() if rec else self.panels,
        }

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        dct = self.dict(rec=False)
        model = dct.pop("model")
        panels = dct.pop("panels")
        return tuple(
            list(dct.items()) + [("model", model.tuple()), ("panels", panels.tuple())]
        )

    def copy(self):
        return self.create(self.dict())

    def __hash__(self) -> int:
        return hash(self.tuple())

    def __len__(self) -> int:
        return len(self.dict())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dc._MISSING_TYPE):
            return False
        try:
            other_dict = other.dict()
        except AttributeError:
            try:
                other_dict = dict(other)  # type: ignore
            except TypeError:
                try:
                    other_dict = dc.asdict(other)
                except TypeError:
                    return False
        return self.dict() == other_dict

    def __repr__(self) -> str:  # type: ignore
        return nested_repr(self)

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "PlotSetup":
        """Create an instance of ``Setup``.

        Args:
            params: Parameters to create an instance of ``Setup``, including
                parameters to create the ``CoreSetup`` instance ``Setup.panels``
                and the ``Dimensions`` instance ``Setup.panels.dimensions``. See
                below for a description of each parameter.

        The parameter descriptions are (for now) collected here because they are
        all directly exposed in the setup files without reflecting the internal
        composition hierarchy ``PlotSetup -> PlotPanelSetup -> Dimensions``.
        This docstring thus serves as a single point of reference for all
        parameters. The docstrings of the individual classes, were the
        parameters should be described as arguments, refer to this docstring to
        avoid duplication.

        Params:
            base_time: Start of the model simulation on which the dispersion
                simulation is based.

            combine_levels: Sum up over multiple vertical levels. Otherwise,
                each is plotted separately.

            combine_species: Sum up over all specified species. Otherwise, each
                is plotted separately.

            dimensions_default: How to complete unspecified dimensions based
                on the values available in the input file. Choices: 'all',
                'first.

            domain: Plot domain. Defaults to 'data', which derives the domain
                size from the input data. Use the format key '{domain}' to embed
                it in ``outfile``. Choices": "auto", "full", "data",
                "release_site", "ch".

            domain_size_lat: Latitudinal extent of domain in degrees. Defaults
                depend on ``domain``.

            domain_size_lon: Longitudinal extent of domain in degrees. Defaults
                depend on ``domain``.

            ens_member_id: Ensemble member ids. Use the format key
                '{ens_member}' to embed it in ``outfile``. Omit for
                deterministic simulations.

            ens_param_mem_min: Minimum number of ensemble members used to
                compute some ensemble variables. Its precise meaning depends on
                the variable.

            ens_param_pctl: Percentile for ``ens_variable = 'percentile'``.

            ens_param_thr: Threshold used to compute some ensemble variables.
                Its precise meaning depends on the variable.

            ens_param_thr_type: Type of threshold, either 'lower' or 'upper'.

            ens_variable: Ensemble variable computed from plot variable. Use the
                format key '{ens_variable}' to embed it in ``outfile``.

            infile: Input file path(s). May contain format keys.

            integrate: Integrate field over time.

            lang: Language. Use the format key '{lang}' to embed it in
                ``outfile``. Choices: "en", "de".

            level: Index/indices of vertical level (zero-based, bottom-up). To
                sum up multiple levels, combine their indices with '+'. Use the
                format key '{level}' to embed it in ``outfile``.

            multipanel_param: Parameter used to plot multiple panels. Only valid
                for ``plot_type = "multipanel"``. The respective parameter must
                have one value per panel. For example, a four-panel plot with
                one ensemble statistic plot each may be specified with
                ``multipanel_param = "ens_variable"`` and ``ens_variable =
                ["minimum", "maximum", "median", "mean", "std_dev",
                "med_abs_dev"]``.

            nageclass: Index of age class (zero-based). Use the format key
                '{nageclass}' to embed it in ``outfile``.

            noutrel: Index of noutrel (zero-based). Use the format key
                '{noutrel}' to embed it in ``outfile``.

            numpoint: Index of release point (zero-based).

            outfile: Output file path(s). May contain format keys.

            outfile_time_format: Format specification (e.g., '%Y%m%d%H%M') for
                time steps (``time_step``, ``base_time``) embedded in
                ``outfile``.

            plot_type: Plot type. Use the format key '{plot_type}' to embed it
                in ``outfile``.

            plot_variable: Variable to plot. Choices: "concentration",
                "tot_deposition", "dry_deposition", "wet_deposition",
                "affected_area", "cloud_arrival_time", "cloud_departure_time".

            species_id: Species id(s). To sum up multiple species, combine their
                ids with '+'. Use the format key '{species_id}' to embed it in
                ``outfile``.

            time: Time step indices (zero-based). Use the format key '{time}'
                to embed one in ``outfile``.

        """
        params = dict(params)
        panel_params = list(params.pop("panels", []))
        model_params = dict(params.pop("model", {}))
        params = {
            param: cast_field_value(
                cls,
                param,
                value,
                auto_wrap=True,
                bool_mode="intuitive",
                timedelta_unit="hours",
                unpack_str=False,
            )
            for param, value in params.items()
        }
        if panel_params:
            multipanel_param = params.get("multipanel_param")
            params["panels"] = PlotPanelSetupGroup.create(
                panel_params, multipanel_param=multipanel_param
            )
        if model_params:
            params["model"] = ModelSetup.create(model_params)
        return cls(**params)

    @classmethod
    def get_params(cls) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "PlotSetup"]) -> "PlotSetup":
        if isinstance(obj, cls):
            return obj
        return cls.create(obj)  # type: ignore

    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        """Cast a parameter to the appropriate type."""
        param_choices = sorted(
            [param for param in cls.get_params() if param != "panels"]
            + [param for param in PlotPanelSetup.get_params() if param != "dimensions"]
            + list(Dimensions.get_params())
        )
        param_choices_fmtd = ", ".join(map(str, param_choices))
        sub_cls_by_name = {"model": ModelSetup, "dimensions": Dimensions}
        try:
            sub_cls = sub_cls_by_name[param]
        except KeyError:
            pass
        else:
            result: Dict[str, Any] = {}
            for sub_param, sub_value in value.items():
                try:
                    # Ignore type to prevent mypy error "has no attribute"
                    sub_value = sub_cls.cast(sub_param, sub_value)  # type: ignore
                    # Don't assign directly to result[sub_param] to prevent the
                    # line from becoming too long, which causes black to disable
                    # the "type: ignore" by moving it to the wrong line
                    # Versions: mypy==0.790; black==20.8b1 (2021-01-06)
                except InvalidParameterNameError as e:
                    raise InvalidParameterNameError(
                        f"{sub_param} ({type(sub_value).__name__}: {sub_value})"
                        f"; choices: {param_choices_fmtd}"
                    ) from e
                else:
                    result[sub_param] = sub_value
            return result
        try:
            if is_dimensions_param(param):
                return Dimensions.cast(param, value)
            elif is_model_setup_param(param):
                return ModelSetup.cast(param, value)
            elif is_plot_panel_setup_param(param):
                return cast_field_value(
                    PlotPanelSetup,
                    param,
                    value,
                    auto_wrap=True,
                    bool_mode="intuitive",
                    timedelta_unit="hours",
                    unpack_str=False,
                )
            return cast_field_value(
                cls,
                param,
                value,
                auto_wrap=True,
                bool_mode="intuitive",
                timedelta_unit="hours",
                unpack_str=False,
            )
        except InvalidParameterNameError as e:
            raise InvalidParameterNameError(
                f"{param} ({type(value).__name__}: {value})"
                f"; choices: {param_choices_fmtd}"
            ) from e

    # SR_TMP Identical to ModelSetup.cast_many
    @classmethod
    def cast_many(
        cls, params: Union[Collection[Tuple[str, Any]], Mapping[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            params_dct: Dict[str, Any] = {}
            for param, value in params:
                if param in params_dct:
                    raise ValueError("duplicate parameter", param)
                params_dct[param] = value
            return cls.cast_many(params_dct)
        params_cast = {}
        for param, value in params.items():
            params_cast[param] = cls.cast(param, value)
        return params_cast

    @classmethod
    def prepare_params(cls, raw_params: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
        """Prepare raw input params and cast them to their appropriate types."""
        params: Dict[str, Any] = {}
        value: Any
        for param, value in raw_params:
            if value in ["None", "*"]:
                value = None
            elif "," in value:
                value = value.split(",")
            elif re.match(r"[0-9]+-[0-9]+", value):
                start, end = value.split("-")
                value = range(int(start), int(end) + 1)
            params[param] = value
        return cls.cast_many(params)

    @classmethod
    def compress(
        cls, setups: Union["PlotSetupGroup", Sequence["PlotSetup"]]
    ) -> "PlotSetup":
        setups = list(setups)
        # SR_TMP <
        plot_variables = [
            setup.panels.collect_equal("plot_variable") for setup in setups
        ]
        if len(set(plot_variables)) != 1:
            raise ValueError(
                f"cannot compress setups: plot_variable differs: {plot_variables}"
            )
        # SR_TMP >
        dcts = [setup.dict() for setup in setups]
        panels_dcts = [panels_dct for dct in dcts for panels_dct in dct.pop("panels")]
        comprd_dct = compress_multival_dicts(dcts, cls_seq=tuple)
        comprd_panels_dct = compress_multival_dicts(panels_dcts, cls_seq=tuple)
        if isinstance(comprd_panels_dct["dimensions"], Sequence):
            comprd_panels_dct["dimensions"] = compress_multival_dicts(
                comprd_panels_dct["dimensions"], cls_seq=tuple
            )
        comprd_dct["panels"] = [comprd_panels_dct]
        return cls.create(comprd_dct)


class PlotSetupGroup:
    """A group of ``Setup`` objects."""

    def __init__(self, setups: Collection[PlotSetup]) -> None:
        """Create an instance of ``SetupGroup``."""
        if not setups:
            raise ValueError(f"setups {type(setups).__name__} is empty")
        if not isinstance(setups, Collection) or (
            setups and not isinstance(next(iter(setups)), PlotSetup)
        ):
            raise ValueError(
                "setups is not an collection of Setup objects, but a"
                f" {type(setups).__name__} of {type(next(iter(setups))).__name__}"
            )
        self._setups: List[PlotSetup] = list(setups)

        # Collect shared setup params
        self.infile: str
        self.ens_member_ids: Optional[Tuple[int, ...]]
        for attr, param in [
            ("infile", "infile"),
            ("ens_member_ids", "model.ens_member_id"),
        ]:
            try:
                value = self.collect_equal(param)
            except UnequalSetupParamValuesError as e:
                raise ValueError(
                    f"value of '{param}' differs between {len(self)} setups: "
                    + ", ".join(map(sfmt, self.collect(param)))
                ) from e
            else:
                setattr(self, attr, value)

    def compress(self) -> PlotSetup:
        return PlotSetup.compress(self)

    def compress_partially(self, param: Optional[str] = None) -> "PlotSetupGroup":
        if param == "outfile":
            grouped_setups: Dict[PlotSetup, List[PlotSetup]] = {}
            for setup in self:
                key = setup.derive({"outfile": "none"})
                if key not in grouped_setups:
                    grouped_setups[key] = []
                grouped_setups[key].append(setup)
            new_setup_lst: List[PlotSetup] = []
            for setup_lst_i in grouped_setups.values():
                outfiles: List[str] = []
                for setup in setup_lst_i:
                    if isinstance(setup.outfile, str):
                        outfiles.append(setup.outfile)
                    else:
                        outfiles.extend(setup.outfile)
                new_setup_lst.append(
                    setup_lst_i[0].derive({"outfile": tuple(outfiles)})
                )
            return PlotSetupGroup(new_setup_lst)
        else:
            raise NotImplementedError(f"{type(self).__name__}.compress('{param}')")

    def derive(self, params: Mapping[str, Any]) -> "PlotSetupGroup":
        return type(self)([setup.derive(params) for setup in self])

    @overload
    def decompress(
        self,
        select: Optional[Collection[str]] = ...,
        skip: Optional[Collection[str]] = ...,
        *,
        internal: Literal[False],
    ) -> List["PlotSetupGroup"]:
        ...

    @overload
    def decompress(
        self,
        select: Optional[Collection[str]] = ...,
        skip: Optional[Collection[str]] = ...,
        *,
        internal: Literal[True] = True,
    ) -> "PlotSetupGroup":
        ...

    def decompress(
        self,
        select=None,
        skip=None,
        *,
        internal=True,
    ):
        """Create a group object for each decompressed setup object.

        Args:
            select (optional): List of parameter names to select for
                decompression; all others will be skipped; parameters named in
                both ``select`` and ``skip`` will be skipped.

            skip (optional): List of parameter names to skip; if they have list
                values, those are retained as such; parameters named in both
                ``skip`` and ``select`` will be skipped.

            internal (optional): Decompress setup group internally and return
                one group containing the decompressed setup objectss; otherwise,
                a separate group is returned for each decompressed setup object.

        """
        if (select, skip) == (None, None):
            return [setup.decompress() for setup in self]
        sub_setup_lst_lst: List[List[PlotSetup]] = []
        for setup in self:
            sub_setups = setup.decompress(select, skip)
            if not sub_setup_lst_lst:
                sub_setup_lst_lst = [[sub_setup] for sub_setup in sub_setups]
            else:
                # SR_TMP <
                assert len(sub_setups) == len(
                    sub_setup_lst_lst
                ), f"{len(sub_setups)} != {len(sub_setup_lst_lst)}"
                # SR_TMP >
                for idx, sub_setup in enumerate(sub_setups):
                    sub_setup_lst_lst[idx].append(sub_setup)
        if internal:
            return PlotSetupGroup(
                [
                    sub_setup
                    for sub_setup_lst in sub_setup_lst_lst
                    for sub_setup in sub_setup_lst
                ]
            )
        return [PlotSetupGroup(sub_setup_lst) for sub_setup_lst in sub_setup_lst_lst]

    # pylint: disable=R0912  # too-many-branches (>12)
    def collect(
        self, param: str, flatten: bool = False, exclude_nones: bool = False
    ) -> List[Any]:
        """Collect all unique values of a parameter for all setups.

        Args:
            param: Name of parameter.

            flatten (optional): Unpack values that are collection of sub-values.

            exclude_nones (optional): Exclude values -- and, if ``flatten`` is
                true, also sub-values -- that are None.

        """
        values: List[Any] = []
        for var_setup in self:
            if is_plot_setup_param(param):
                value = getattr(var_setup, param)
            elif is_model_setup_param(param):
                value = getattr(var_setup.model, param.replace("model.", ""))
            elif is_plot_panel_setup_param(param) or is_dimensions_param(param):
                value = var_setup.panels.collect(param)
            else:
                raise ValueError(f"invalid param '{param}'")
            if flatten and isinstance(value, Collection) and not isinstance(value, str):
                for sub_value in value:
                    if exclude_nones and sub_value is None:
                        continue
                    if sub_value not in values:
                        values.append(sub_value)
            else:
                if exclude_nones and value is None:
                    continue
                if value not in values:
                    values.append(value)
        return values

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        if is_plot_panel_setup_param(param) or is_dimensions_param(param):
            values = self.collect(param, flatten=True)
        else:
            values = self.collect(param)
        if not values:
            return None
        if not all(value == values[0] for value in values[1:]):
            raise UnequalSetupParamValuesError(param, values)
        return next(iter(values))

    @overload
    def group(self, param: str) -> Dict[Any, "PlotSetupGroup"]:
        ...

    @overload
    def group(self, param: Sequence[str]) -> Dict[Tuple[Any, ...], "PlotSetupGroup"]:
        ...

    def group(self, param):
        """Group setups by the value of one or more parameters."""
        if not isinstance(param, str):
            grouped: Dict[Tuple[Any, ...], "PlotSetupGroup"] = {}
            params: List[str] = list(param)
            for value, sub_setups in self.group(params[0]).items():
                if len(params) == 1:
                    grouped[(value,)] = sub_setups
                elif len(params) > 1:
                    for values, sub_sub_setups in sub_setups.group(params[1:]).items():
                        key = tuple([value] + list(values))
                        grouped[key] = sub_sub_setups
                else:
                    raise NotImplementedError(f"{len(param)} sub_params", param)
            return grouped
        else:
            grouped_raw: Dict[Any, List[PlotSetup]] = {}
            for setup in self:
                value = get_setup_param_value(setup, param)
                try:
                    hash(value)
                except TypeError as e:
                    raise Exception(
                        f"cannot group by param '{param}': value has unhashable"
                        f" type {type(value).__name__}: {value}"
                    ) from e
                if value not in grouped_raw:
                    grouped_raw[value] = []
                grouped_raw[value].append(setup)
            grouped: Dict[Any, "PlotSetupGroup"] = {
                value: type(self)(setups) for value, setups in grouped_raw.items()
            }
            return grouped

    @overload
    def complete_dimensions(
        self,
        raw_dimensions: Mapping[str, Mapping[str, Any]],
        species_ids: Sequence[int],
        *,
        inplace: Literal[False] = ...,
    ) -> "PlotSetupGroup":
        ...

    @overload
    def complete_dimensions(
        self,
        raw_dimensions: Mapping[str, Mapping[str, Any]],
        species_ids: Sequence[int],
        *,
        inplace: Literal[True],
    ) -> None:
        ...

    def complete_dimensions(self, raw_dimensions, species_ids, *, inplace=False):
        """Complete unconstrained dimensions based on available indices."""
        obj = self if inplace else self.copy()
        for plot_setup in obj:
            for panel_setup in plot_setup.panels:
                panel_setup.complete_dimensions(
                    raw_dimensions, species_ids, inplace=True
                )
        return None if inplace else obj

    def override_output_suffixes(self, suffix: Union[str, Collection[str]]) -> None:
        """Override output file suffixes one or more times.

        If multiple suffixes are passed, all setups are multiplied as many
        times.

        Args:
            suffix: One or more replacement suffix.

        """
        suffixes: List[str] = list([suffix] if isinstance(suffix, str) else suffix)
        if not suffixes:
            raise ValueError("must pass one or more suffixes")
        new_setups: List[PlotSetup] = []
        for setup in self:
            old_outfiles: List[str] = (
                [setup.outfile]
                if isinstance(setup.outfile, str)
                else list(setup.outfile)
            )
            new_outfiles: List[str] = []
            for old_outfile in list(old_outfiles):
                if any(old_outfile.endswith(f".{suffix}") for suffix in ["png", "pdf"]):
                    old_outfile = ".".join(old_outfile.split(".")[:-1])
                for suffix_i in suffixes:
                    if suffix_i.startswith("."):
                        suffix_i = suffix_i[1:]
                    new_outfiles.append(f"{old_outfile}.{suffix_i}")
            new_setup = setup.derive(
                {
                    "outfile": next(iter(new_outfiles))
                    if len(new_outfiles) == 1
                    else tuple(new_outfiles)
                }
            )
            new_setups.append(new_setup)
        self._setups = new_setups

    def copy(self) -> "PlotSetupGroup":
        return type(self)([setup.copy() for setup in self])

    def dicts(self) -> List[Dict[str, Any]]:
        return [setup.dict() for setup in self]

    def __repr__(self) -> str:
        try:
            return PlotSetupGroupFormatter(self).repr()
        # pylint: disable=W0703  # broad-except
        except Exception as e:
            return (
                f"<{type(self).__name__}.__repr__:"
                f" exception in PlotSetupGroupFormatter(self).repr(): {repr(e)}"
                f"\n{pformat(self.dicts())}>"
            )

    def __len__(self) -> int:
        return len(self._setups)

    def __iter__(self) -> Iterator[PlotSetup]:
        return iter(self._setups)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dc._MISSING_TYPE):
            return False
        try:
            other_dicts = other.dicts()  # type: ignore
        except AttributeError:
            other_dicts = [dict(obj) for obj in other]  # type: ignore
        self_dicts = self.dicts()
        return all(
            [
                all(obj in other_dicts for obj in self_dicts),
                all(obj in self_dicts for obj in other_dicts),
            ]
        )

    @classmethod
    def create(
        cls, setups: Collection[Union[PlotSetup, Mapping[str, Any]]]
    ) -> "PlotSetupGroup":
        setup_lst: List[PlotSetup] = []
        for obj in setups:
            if not isinstance(obj, PlotSetup):
                obj = PlotSetup.create(obj)
            setup_lst.append(obj)
        return cls(setup_lst)

    @classmethod
    def merge(cls, setups_lst: Sequence["PlotSetupGroup"]) -> "PlotSetupGroup":
        return cls([setup for setups in setups_lst for setup in setups])


class PlotSetupGroupFormatter(PlotPanelSetupGroupFormatter):
    """Format a human-readable representation of a ``PlotSetupGroup``."""

    def _group_params(self, same: Dict[str, Any], diff: Dict[str, Any]) -> None:
        self._group_plot_params(same, diff)
        same_panels: Dict[str, Any] = {}
        diff_panels: Dict[str, Any] = {}
        self._group_panel_params(same_panels, diff_panels)
        self._group_dims_params(same_panels, diff_panels)
        if same_panels:
            same["panels"] = same_panels
        if diff_panels:
            diff["panels"] = diff_panels

    def _group_plot_params(self, same: Dict[str, Any], diff: Dict[str, Any]) -> None:
        for param in PlotSetup.get_params():
            if param == "panels":
                continue  # Handled by method _group_panel_params
            try:
                value = self.obj.collect_equal(param)
            except UnequalSetupParamValuesError:
                diff[param] = self.obj.collect(param)
            else:
                same[param] = value
