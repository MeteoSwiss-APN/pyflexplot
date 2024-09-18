# pylint: disable=C0302  # too-many-lines (>1000)
"""Read setup files."""
# Standard library
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import TypeAlias

# Third-party
import toml

# First-party
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards
from srutils.exceptions import InvalidParameterNameError
from srutils.exceptions import InvalidParameterValueError

# Local
from .dimensions import Dimensions
from .dimensions import is_dimensions_param
from .files_setup import FilesSetup
from .files_setup import is_files_setup_param
from .layout_setup import is_layout_setup_param
from .layout_setup import LayoutSetup
from .model_setup import is_model_setup_param
from .model_setup import ModelSetup
from .plot_panel_setup import is_ensemble_params_param
from .plot_panel_setup import is_plot_panel_setup_param
from .plot_panel_setup import PlotPanelSetup
from .plot_setup import is_plot_setup_param
from .plot_setup import PlotSetup
from .plot_setup import PlotSetupGroup


class SetupFile:
    """Setup file to be read from and/or written to disk."""

    def __init__(self, path: Union[Path, str]) -> None:
        """Create an instance of ``SetupFile``."""
        self.path: Path = Path(path)

    # pylint: disable=R0914  # too-many-locals
    def read(
        self,
        *,
        override: Optional[Mapping[str, Any]] = None,
    ) -> PlotSetupGroup:
        """Read the setup from a text file in TOML format."""
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                raw_data = toml.load(f)
            except Exception as e:
                raise Exception(
                    f"error parsing TOML file {self.path} ({type(e).__name__}: {e})"
                ) from e
        if not raw_data:
            raise ValueError("empty setup file", self.path)
        semi_raw_data = nested_dict_resolve_wildcards(
            raw_data, double_criterion=lambda key: key.endswith("+")
        )
        raw_params_lst = decompress_nested_dict(
            semi_raw_data, branch_end_criterion=lambda key: not key.startswith("_")
        )
        if override is not None:
            raw_params_lst, old_raw_params_lst = [], raw_params_lst
            for old_raw_params in old_raw_params_lst:
                raw_params = {**old_raw_params, **override}
                if raw_params not in raw_params_lst:
                    raw_params_lst.append(raw_params)
        raw_params_lst = self.prepare_raw_params(raw_params_lst)
        if not raw_params_lst:
            raise Exception(f"no setups defined in file {self.path}")
        setups = PlotSetupGroup.create(raw_params_lst)
        return setups


    @classmethod
    def read_many(
        cls,
        paths: Sequence[Union[Path, str]],
        override: Mapping[str, Any],
    ) -> List[PlotSetupGroup]:
        
        key_t: TypeAlias = Tuple[str, Tuple[int, ...]]
        setups_by_infiles: Dict[key_t, List[PlotSetup]] = {}

        for path in paths:
            setups = cls(path).read(override=override)
            for setup in setups:
                key: key_t = (setup.files.input, setup.model.ens_member_id)
                if key not in setups_by_infiles:
                    setups_by_infiles[key] = []
                if setup not in setups_by_infiles[key]:
                    setups_by_infiles[key].append(setup)

        return [PlotSetupGroup(setup_lst) for setup_lst in setups_by_infiles.values()]

    # pylint: disable=R0912  # too-many-branches (>12)
    @classmethod
    def prepare_raw_params(
        cls, raw_params: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Group a flat parameter dict into nested structure and cast values."""
        raw_params_lst: List[Mapping[str, Any]]
        if isinstance(raw_params, Mapping):
            raw_params_lst = [raw_params]
        else:
            raw_params_lst = list(raw_params)
        params_lst = []
        for raw_params_i in raw_params_lst:
            params: Dict[str, Any] = {}
            panels: Dict[str, Any] = {}
            ens_params: Dict[str, Any] = {}
            for param, value in raw_params_i.items():
                param = cls.prepare_raw_param_name(param)
                value = cls.preproc_raw_param_value(value)
                if param == "multipanel_param":
                    value = cls.prepare_raw_param_name(value)
                if is_plot_setup_param(param):
                    params[param] = value
                elif is_files_setup_param(param):
                    if "files" not in params:
                        params["files"] = {}
                    param = param.replace("files.", "")
                    params["files"][param] = value
                elif is_layout_setup_param(param):
                    if "layout" not in params:
                        params["layout"] = {}
                    param = param.replace("layout.", "")
                    params["layout"][param] = value
                elif is_model_setup_param(param):
                    if "model" not in params:
                        params["model"] = {}
                    param = param.replace("model.", "")
                    params["model"][param] = value
                elif is_plot_panel_setup_param(param):
                    panels[param] = value
                elif is_dimensions_param(param):
                    if "dimensions" not in panels:
                        panels["dimensions"] = {}
                    param = param.replace("dimensions.", "")
                    panels["dimensions"][param] = value
                elif is_ensemble_params_param(param):
                    param = param.replace("ens_params.", "")
                    ens_params[param] = value
                else:
                    raise InvalidParameterNameError(param)
            if ens_params:
                panels["ens_params"] = ens_params
            if panels:
                params["panels"] = panels
            params_lst.append(params)
        return params_lst

    @classmethod
    def prepare_raw_param_name(cls, raw_name: str) -> str:
        if raw_name.startswith("ens_param_"):
            return f"ens_params.{raw_name.replace('ens_param_', '')}"
        return {
            "infile": "files.input",
            "layout_type": "layout.type",
            "model": "model.name",
            "outfile": "files.output",
            "outfile_time_format": "files.output_time_format",
        }.get(raw_name, raw_name)

    @classmethod
    def preproc_raw_param_value(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value in ["None", "*"]:
            value = None
        elif "," in value:
            value = value.split(",")
        elif re.match(r"[0-9]+-[0-9]+", value):
            start, end = value.split("-")
            value = range(int(start), int(end) + 1)
        return value

    # pylint: disable=R0911  # too-many-return-statements (>6)
    @classmethod
    def is_valid_raw_param_name(cls, name: str) -> bool:
        """Check a raw parameter with optional raw value for validity."""
        name = cls.prepare_raw_param_name(name)
        if is_plot_setup_param(name):
            return True
        elif is_files_setup_param(name):
            return True
        elif is_layout_setup_param(name):
            return True
        elif is_model_setup_param(name):
            return True
        elif is_plot_panel_setup_param(name):
            return True
        elif is_dimensions_param(name):
            return True
        else:
            return False

    @classmethod
    def is_valid_raw_param_value(
        cls, name: str, raw_value: Optional[str] = None
    ) -> bool:
        """Check a raw parameter with optional raw value for validity."""
        if not cls.is_valid_raw_param_name(name):
            return False
        name = cls.prepare_raw_param_name(name)
        raw_value = cls.preproc_raw_param_value(raw_value)
        if is_plot_setup_param(name):
            if name in ["files", "layout", "model", "dimensions"]:
                fct = lambda name, raw_value: raw_value  # noqa
            else:
                fct = PlotSetup.cast
        elif is_files_setup_param(name):
            name = name.replace("files.", "")
            fct = FilesSetup.cast
        elif is_layout_setup_param(name):
            name = name.replace("layout.", "")
            fct = LayoutSetup.cast
        elif is_model_setup_param(name):
            name = name.replace("model.", "")
            fct = ModelSetup.cast
        elif is_plot_panel_setup_param(name):
            fct = PlotPanelSetup.cast
        elif is_dimensions_param(name):
            name = name.replace("dimensions.", "")
            fct = Dimensions.cast
        else:
            return False
        try:
            fct(name, raw_value)
        except InvalidParameterValueError:
            return False
        return True
