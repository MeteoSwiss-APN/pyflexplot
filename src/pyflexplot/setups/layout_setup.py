"""Layout setup."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Mapping
from typing import Optional

# Local
from .base_setup import BaseSetup


# SR_TMP <<< TODO cleaner solution
def is_layout_setup_param(param: str, recursive: bool = False) -> bool:
    if recursive:
        raise NotImplementedError("recursive")
    return param.replace("layout.", "") in LayoutSetup.get_params()


# SR_TMP TODO pull common base class out of LayoutSetup, ModelSetup etc.
@dc.dataclass
class LayoutSetup(BaseSetup):
    color_style: str = "auto"
    plot_type: str = "auto"
    multipanel_param: Optional[str] = None
    scale_fact: float = 1.0
    type: str = "auto"

    def __post_init__(self) -> None:
        # Check plot_type
        choices = ["auto", "multipanel"]
        assert self.plot_type in choices, self.plot_type

        # Check multipanel_param
        if self.multipanel_param is not None:
            multipanel_param_choices = ["ens_variable", "ens_params.pctl"]
            if self.multipanel_param not in multipanel_param_choices:
                raise NotImplementedError(
                    f"unknown multipanel_param '{self.multipanel_param}'"
                    f"; choices: {', '.join(multipanel_param_choices)}"
                )

    # pylint: disable=W0221  # arguments-differ (simulation_type)
    @classmethod
    def create(
        cls,
        params: Mapping[str, Any],
        *,
        simulation_type: str = "deterministic",
    ) -> "LayoutSetup":
        params = cls.cast_many(params)
        if params.get("type", "auto") == "auto":
            if simulation_type == "deterministic":
                params["type"] = "post_vintage"
            elif simulation_type == "ensemble":
                params["type"] = "post_vintage_ens"
            else:
                raise ValueError(f"invalid simulation_type '{simulation_type}'")
        params = cls.cast_many(params)
        return cls(**params)
