# pylint: disable=C0302  # too-many-lines (>1000)
"""Read setup files."""
# Standard library
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import toml

# First-party
from srutils.dict import decompress_nested_dict
from srutils.dict import nested_dict_resolve_wildcards

# Local
from .plot_setup import PlotSetup
from .plot_setup import PlotSetupGroup
from .plot_setup import prepare_raw_params


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
        only: Optional[int] = None,
    ) -> PlotSetupGroup:
        """Read the setup from a text file in TOML format."""
        with open(self.path, "r") as f:
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
        raw_params_lst = prepare_raw_params(raw_params_lst)
        setups = PlotSetupGroup.create(raw_params_lst)
        if only is not None:
            if only < 0:
                raise ValueError(f"only must not be negative ({only})")
            setups = PlotSetupGroup(list(setups)[:only])
        return setups

    def write(self, *args, **kwargs) -> None:
        """Write the setup to a text file in TOML format."""
        raise NotImplementedError(f"{type(self).__name__}.write")

    @classmethod
    def read_many(
        cls,
        paths: Sequence[Union[Path, str]],
        override: Optional[Mapping[str, Any]] = None,
        only: Optional[int] = None,
        each_only: Optional[int] = None,
    ) -> List[PlotSetupGroup]:
        if only is not None:
            if only < 0:
                raise ValueError("only must not be negative", only)
            each_only = only
        elif each_only is not None:
            if each_only < 0:
                raise ValueError("each_only must not be negative", each_only)
        KeyT = Tuple[str, Optional[Tuple[int, ...]]]
        setups_by_infiles: Dict[KeyT, List[PlotSetup]] = {}
        n_setups = 0
        for path in paths:
            setups = cls(path).read(override=override, only=each_only)
            for setup in setups:
                if only is not None and n_setups >= only:
                    break
                key: KeyT = (setup.infile, setup.model.ens_member_id)
                if key not in setups_by_infiles:
                    setups_by_infiles[key] = []
                if setup not in setups_by_infiles[key]:
                    setups_by_infiles[key].append(setup)
                    n_setups += 1
        return [PlotSetupGroup(setup_lst) for setup_lst in setups_by_infiles.values()]