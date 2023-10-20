# pylint: disable=C0302  # too-many-lines (>1000)
"""Plot setup and setup files.

The setup parameters that are exposed in the setup files are all described in
the docstring of the class method ``Setup.create``.

"""
# Standard library
import dataclasses as dc
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
from srutils.dict import compress_multival_dicts
from srutils.dict import decompress_multival_dict
from srutils.dict import merge_dicts
from srutils.exceptions import InvalidParameterNameError
from srutils.format import sfmt

# Local
from ..utils.exceptions import UnequalSetupParamValuesError
from .base_setup import BaseSetup
from .files_setup import FilesSetup
from .files_setup import is_files_setup_param
from .layout_setup import is_layout_setup_param
from .layout_setup import LayoutSetup
from .model_setup import is_model_setup_param
from .model_setup import ModelSetup
from .plot_panel_setup import is_plot_panel_setup_param
from .plot_panel_setup import PlotPanelSetup
from .plot_panel_setup import PlotPanelSetupGroup
from .plot_panel_setup import PlotPanelSetupGroupFormatter


# SR_TMP <<< TODO cleaner solution
def is_plot_setup_param(param: str, recursive: bool = False) -> bool:
    if recursive:
        raise NotImplementedError("recursive")
    return param in PlotSetup.get_params()


# SR_TMP <<< TODO cleaner solution
def get_setup_param_value(setup: "PlotSetup", param: str) -> Any:
    if is_plot_setup_param(param):
        return getattr(setup, param)
    elif is_files_setup_param(param):
        return getattr(setup.files, param.replace("files.", ""))
    elif is_layout_setup_param(param):
        return getattr(setup.layout, param.replace("layout.", ""))
    elif is_model_setup_param(param):
        return getattr(setup.model, param.replace("model.", ""))
    elif is_plot_panel_setup_param(param, recursive=True):
        return setup.panels.collect_equal(param)
    raise ValueError("invalid input setup parameter", param)


# SR_TODO Clean up docstring -- where should format key hints go?
@dc.dataclass(frozen=True)
class PlotSetup(BaseSetup):
    """Setup of a whole plot.

    See docstring of ``Setup.create`` for details on parameters.

    Note that until all the dimensions have been completed etc., a ``PlotSetup``
    object may ultimately correspond to multiple plots, but at plotting time,
    each ``PlotSetup`` object corresponds to one plot (which may be multiple
    times with different names/formats, though).

    """

    files: FilesSetup
    layout: LayoutSetup = dc.field(default_factory=LayoutSetup)
    model: ModelSetup = dc.field(default_factory=ModelSetup)
    panels: PlotPanelSetupGroup = dc.field(
        default_factory=lambda: PlotPanelSetupGroup([PlotPanelSetup()])
    )

    def __post_init__(self) -> None:
        # Wrap check in function to locally disable pylint check
        # pylint: disable=E1101  # no-member ("Instance of 'Field' has no '...' member")
        def _check_plot_type():
            simulation_type = self.model.simulation_type
            ens_variables = sorted(set(self.panels.collect("ens_variable")))
            if simulation_type == "deterministic" and ens_variables != ["none"]:
                # pylint: disable=C0209  # consider-using-f-string (v2.11.1)
                raise ValueError(
                    "ens_variable(s) not 'none' for deterministic simulation: "
                    + ", ".join(map("'{}'".format, ens_variables))
                )
            elif simulation_type == "ensemble" and ens_variables == ["none"]:
                raise ValueError("ens_variable is 'none' for ensemble simulation")

        _check_plot_type()

        # Check number of panels
        n_panel_choices = [4]
        n_panels = len(self.panels)
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        if self.layout.multipanel_param is None:
            if n_panels != 1:
                raise ValueError(
                    f"number of panels ({n_panels}) must be 1  for single-panel plots"
                    "(multipanel_param is None)"
                )
        elif n_panels not in n_panel_choices:
            # pylint: disable=E1101  # no-member [pylint 2.7.4]
            # (pylint 2.7.4 does not support dataclasses.field)
            raise ValueError(
                f"wrong number of panels ({n_panels}); supported: {n_panel_choices}"
                f" (multipanel_param '{self.layout.multipanel_param}')"
            )

        # Check types of sub-setups
        param_types = [
            ("layout", LayoutSetup),
            ("model", ModelSetup),
            ("panels", PlotPanelSetupGroup),
        ]
        for name, cls in param_types:
            val = getattr(self, name)
            if not isinstance(val, cls):
                raise ValueError(
                    f"'{name}' has wrong type: expected {cls.__name__}, got "
                    + type(val).__name__
                )

    def collect(self, param: str, *, unique: bool = False) -> Any:
        """Collect the value(s) of a parameter.

        Args:
            param: Name of parameter.

            unique (optional): Return duplicate sub-values only once.

        """
        if is_plot_setup_param(param):
            value = getattr(self, param)
        elif is_files_setup_param(param):
            value = getattr(self.files, param.replace("files.", ""))
        elif is_layout_setup_param(param):
            value = getattr(self.layout, param.replace("layout.", ""))
        elif is_model_setup_param(param):
            value = getattr(self.model, param.replace("model.", ""))
        elif is_plot_panel_setup_param(param, recursive=True):
            # pylint: disable=E1101  # no-member [pylint 2.7.4]
            # (pylint 2.7.4 does not support dataclasses.field)
            value = self.panels.collect(param, unique=unique)
        else:
            raise ValueError(f"invalid param '{param}'")
        return value

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
            # Check that ens_member_id is not internally decompressed
            ens_member_id = self.collect("ens_member_id")
            selected = not select or (
                "model.ens_member_id" in select or "ens_member_id" in select
            )
            skipped = "model.ens_member_id" in (skip or []) or "ens_member_id" in (
                skip or []
            )
            if ens_member_id and (not select or selected) and not skipped:
                raise ValueError(
                    f"cannot decompress ensemble setup (ens_member_id={ens_member_id})"
                    " with internal=True by 'model.ens_member_id'; either skip"
                    " 'model.ens_member_id' or pass internal=False"
                )

        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        if self.layout.plot_type == "multipanel":
            skip = list(skip or []) + [self.layout.multipanel_param]

        def group_params(
            params: Optional[Collection[str]],
        ) -> Union[
            Tuple[None, None, None, None, None],
            Tuple[List[str], List[str], List[str], List[str], List[str]],
        ]:
            if params is None:
                return None, None, None, None, None
            files: List[str] = []
            layout: List[str] = []
            model: List[str] = []
            panels: List[str] = []
            other: List[str] = []
            for param in params:
                if is_files_setup_param(param):
                    files.append(param.replace("files.", ""))
                elif is_layout_setup_param(param):
                    layout.append(param.replace("layout.", ""))
                elif is_model_setup_param(param):
                    model.append(param.replace("model.", ""))
                elif is_plot_panel_setup_param(param, recursive=True):
                    panels.append(param)
                else:
                    other.append(param)
            return files, layout, model, panels, other

        # Group select and skip parameters
        (
            select_files,
            select_layout,
            select_model,
            select_panels,
            select_outer,
        ) = group_params(select)
        (
            skip_files,
            skip_layout,
            skip_model,
            skip_panels,
            skip_outer,
        ) = group_params(skip)

        # PlotSetup.files
        files_dct = self.dict().pop("files")
        files_dcts = decompress_multival_dict(files_dct, select_files, skip_files)

        # PlotSetup.layout
        layout_dct = self.dict().pop("layout")
        layout_dcts = decompress_multival_dict(layout_dct, select_layout, skip_layout)

        # PlotSetup.model
        model_dct = self.dict().pop("model")
        model_dcts = decompress_multival_dict(model_dct, select_model, skip_model)

        # PlotSetup.panels
        panels_dcts_lst: List[List[Dict[str, Any]]] = []
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        # pylint: disable=E1133  # not-an-iterable
        # (pylint 2.10.2 doesn't recognize PlotPanelSetupGroup as iterable)
        for panels in self.panels.decompress(
            select_panels, skip_panels, internal=False
        ):
            panels_dcts_lst.append(panels.dicts())

        # PlotSetup.*
        outer_dct = self.dict()
        outer_dct.pop("panels")
        outer_dcts = decompress_multival_dict(outer_dct, select_outer, skip_outer)

        # Merge expanded dicts
        dcts: List[Dict[str, Any]] = []
        for outer_dct_i in outer_dcts:
            for files_dct in files_dcts:
                for layout_dct in layout_dcts:
                    for model_dct in model_dcts:
                        for panels_dcts in panels_dcts_lst:
                            dcts.append(
                                {
                                    **deepcopy(outer_dct_i),
                                    "files": deepcopy(files_dct),
                                    "layout": deepcopy(layout_dct),
                                    "model": deepcopy(model_dct),
                                    "panels": deepcopy(panels_dcts),
                                }
                            )

        if internal:
            return PlotSetupGroup.create(dcts)
        return list(map(type(self).create, dcts))

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
            params = list(params)
            if not params:
                setup_lst = [self.copy()]
            else:
                setup_lst = [self.derive(sub_params) for sub_params in params]
            return PlotSetupGroup(setup_lst)
        elif isinstance(params, Mapping):
            params = dict(params)
            if isinstance(params.get("panels"), Mapping):
                params["panels"] = [params["panels"]]
            self_dct = self.dict()
            # Ensure correct simulation type
            model_params = params.get("model", [])
            if (
                "simulation_type" not in model_params
                and "ens_member_id" in model_params
            ):
                if (
                    model_params["ens_member_id"] not in [None, "none"]
                    and len(model_params["ens_member_id"]) > 1
                ):
                    params["model"]["simulation_type"] = "ensemble"
            # Reset 'dimensions.variable'; re-derived from 'plot_variable'
            for panel_dct in self_dct["panels"]:
                panel_dct["dimensions"].pop("variable")
            dct = merge_dicts(self_dct, params, overwrite_seqs=True)
            if len(dct.get("panels", [])) == 1:
                dct["panels"] = next(iter(dct["panels"]))
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
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        return {
            **dc.asdict(self),
            "files": self.files.dict() if rec else self.files,
            "layout": self.layout.dict() if rec else self.layout,
            "model": self.model.dict() if rec else self.model,
            "panels": self.panels.dicts() if rec else self.panels,
        }

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        dct = self.dict(rec=False)
        files = dct.pop("files")
        layout = dct.pop("layout")
        model = dct.pop("model")
        panels = dct.pop("panels")
        return tuple(
            list(dct.items())
            + [
                ("files", files.tuple()),
                ("layout", layout.tuple()),
                ("model", model.tuple()),
                ("panels", panels.tuple()),
            ]
        )

    # Explicitly inherit to prevent @dataclass from changing it
    def __eq__(self, other: Any) -> bool:
        # pylint: disable=W0235  # useless-super-delegation (v2.11.1)
        # see also https://github.com/PyCQA/pylint/issues/3651
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.tuple())

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

            color_style: Color style of the plot type. Defaults to "auto".

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

            release: Index of release (zero-based). Use the format key
                '{release}' to embed it in ``outfile``.

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

        files_params: Dict[str, Any] = dict(params.pop("files", {}))
        layout_params: Dict[str, Any] = dict(params.pop("layout", {}))
        model_params: Dict[str, Any] = dict(params.pop("model", {}))
        panels_params: Union[Dict[str, Any], List[Dict[str, Any]]]

        raw_panels_params: Any = params.pop("panels", {})
        if isinstance(raw_panels_params, Mapping):
            panels_params = dict(raw_panels_params)
        elif isinstance(raw_panels_params, Sequence) and all(
            isinstance(val, Mapping) for val in raw_panels_params
        ):
            panels_params = list(map(dict, raw_panels_params))
        else:
            raw_type = type(raw_panels_params).__name__
            raise ValueError(
                f"params 'panels' has invalid type '{raw_type}'; expecting "
                "mapping or sequence thereof"
            )

        params = cls.cast_many(params)

        files_setup = FilesSetup.create(files_params)
        params["files"] = files_setup

        model_setup = ModelSetup.create(model_params)
        params["model"] = model_setup

        layout_setup = LayoutSetup.create(
            layout_params, simulation_type=model_setup.simulation_type
        )
        params["layout"] = layout_setup

        panels = PlotPanelSetupGroup.create(
            panels_params, multipanel_param=layout_setup.multipanel_param
        )
        params["panels"] = panels

        return cls(**params)

    @classmethod
    def as_setup(cls, obj: Union[Mapping[str, Any], "PlotSetup"]) -> "PlotSetup":
        if isinstance(obj, cls):
            return obj
        return cls.create(obj)  # type: ignore

    @classmethod
    def cast(cls, param: str, value: Any, recursive: bool = True) -> Any:
        """Cast a parameter to the appropriate type."""
        if is_plot_setup_param(param):
            return super().cast(param, value)
        elif recursive:
            if is_files_setup_param(param):
                return FilesSetup.cast(param.replace("files.", ""), value)
            elif is_layout_setup_param(param):
                return LayoutSetup.cast(param.replace("layout.", ""), value)
            elif is_model_setup_param(param):
                return ModelSetup.cast(param.replace("model.", ""), value)
            elif is_plot_panel_setup_param(param, recursive=True):
                return PlotPanelSetup.cast(param, value, recursive=True)
        raise InvalidParameterNameError(f"{param} ({type(value).__name__}: {value})")

    @classmethod
    def cast_many(
        cls,
        params: Union[Collection[Tuple[str, Any]], Mapping[str, Any]],
        recursive: bool = True,
    ) -> Dict[str, Any]:
        return super().cast_many(params, recursive)

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
            ("infile", "files.input"),
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
        if param == "files.output":
            grouped_setups: Dict[PlotSetup, List[PlotSetup]] = {}
            for setup in self:
                key = setup.derive({"files": {"output": "none"}})
                if key not in grouped_setups:
                    grouped_setups[key] = []
                grouped_setups[key].append(setup)
            new_setup_lst: List[PlotSetup] = []
            for setup_lst_i in grouped_setups.values():
                outfiles: List[str] = []
                for setup in setup_lst_i:
                    if isinstance(setup.files.output, str):
                        outfiles.append(setup.files.output)
                    else:
                        outfiles.extend(setup.files.output)
                new_setup_lst.append(
                    setup_lst_i[0].derive({"files": {"output": tuple(outfiles)}})
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
        self,
        param: str,
        *,
        exclude_nones: bool = False,
        flatten: bool = False,
        unique: bool = False,
    ) -> List[Any]:
        """Collect all values of a parameter for all setups.

        Args:
            param: Name of parameter.

            exclude_nones (optional): Exclude values -- and, if ``flatten`` is
                true, also sub-values -- that are None.

            flatten (optional): Unpack values that are collection of sub-values.

            unique (optional): Return duplicate values only once.

        """
        values: List[Any] = []
        for setup in self:
            value = setup.collect(param, unique=unique)
            if flatten and isinstance(value, Collection) and not isinstance(value, str):
                for sub_value in value:
                    if exclude_nones and sub_value is None:
                        continue
                    if not unique or sub_value not in values:
                        values.append(sub_value)
            else:
                if exclude_nones and value is None:
                    continue
                if not unique or value not in values:
                    values.append(value)
        return values

    def collect_equal(self, param: str) -> Any:
        """Collect the value of a parameter that is shared by all setups."""
        flatten = is_plot_panel_setup_param(param, recursive=True)
        values = self.collect(param, unique=True, flatten=flatten)
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
                [setup.files.output]
                if isinstance(setup.files.output, str)
                else list(setup.files.output)
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
                    "files": {
                        "output": next(iter(new_outfiles))
                        if len(new_outfiles) == 1
                        else tuple(new_outfiles)
                    }
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
        cls,
        setups: Union[
            Mapping[str, Any],
            Collection[Mapping[str, Any]],
            Collection[PlotSetup],
        ],
    ) -> "PlotSetupGroup":
        def prepare_setup_dcts(dct: Mapping[str, Any]) -> List[Dict[str, Any]]:
            def unpack_dimension(
                panels_dcts: Sequence[Mapping[str, Any]],
                flag_name: str,
                dim_name: str,
            ) -> List[Dict[str, Any]]:
                unpacked_panels_dcts: List[Dict[str, Any]] = []
                for panels_dct in panels_dcts:
                    if not panels_dct.get(flag_name, False):
                        try:
                            value = panels_dct["dimensions"][dim_name]
                        except KeyError:
                            pass
                        else:
                            if isinstance(value, Sequence):
                                for sub_value in value:
                                    unpacked_panels_dcts.append(
                                        merge_dicts(
                                            panels_dct,
                                            {"dimensions": {dim_name: sub_value}},
                                            overwrite_seqs=True,
                                        )
                                    )
                                continue
                    unpacked_panels_dcts.append(dict(panels_dct))
                return unpacked_panels_dcts

            dct = dict(dct)
            try:
                panels_obj = dct.pop("panels")
            except KeyError:
                return [dct]
            panels_dcts: List[Dict[str, Any]]
            if isinstance(panels_obj, Mapping):
                panels_dcts = [dict(**panels_obj)]
            elif isinstance(panels_obj, Sequence) and not isinstance(panels_obj, str):
                panels_dcts = list(map(dict, panels_obj))
            else:
                raise TypeError(
                    f"dct['panels'] has unexpected type {type(panels_obj).__name__}"
                )
            panels_dcts = unpack_dimension(panels_dcts, "combine_levels", "level")
            panels_dcts = unpack_dimension(panels_dcts, "combine_species", "species_id")
            dcts: List[Dict[str, Any]] = []
            for panels_dct in panels_dcts:
                dcts.append({**dct, "panels": panels_dct})
            return dcts

        if isinstance(setups, Mapping):
            setups = [setups]
        setup_lst: List[PlotSetup] = []
        for obj in setups:
            if isinstance(obj, PlotSetup):
                setup_lst.append(obj)
            else:
                # SR_TMP < TODO move logic of prepare_setup_dcts down
                dcts: Sequence[Mapping[str, Any]]
                if obj.get("layout", {}).get("plot_type") == "multipanel":
                    dcts = [obj]
                else:
                    dcts = prepare_setup_dcts(obj)
                # SR_TMP >
                for dct in dcts:
                    setup = PlotSetup.create(dct)
                    setup_lst.append(setup)
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
