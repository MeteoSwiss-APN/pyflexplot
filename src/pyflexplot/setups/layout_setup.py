"""Layout setup."""
# Standard library
import dataclasses as dc
from typing import Optional

# Local
from .base_setup import BaseSetup


# SR_TMP <<< TODO cleaner solution
def is_layout_setup_param(param: str) -> bool:
    return param.replace("layout.", "") in LayoutSetup.get_params()


# SR_TMP TODO pull common base class out of LayoutSetup, ModelSetup etc.
@dc.dataclass
class LayoutSetup(BaseSetup):
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
            multipanel_param_choices = ["ens_variable"]
            if self.multipanel_param not in multipanel_param_choices:
                raise NotImplementedError(
                    f"unknown multipanel_param '{self.multipanel_param}'"
                    f"; choices: {', '.join(multipanel_param_choices)}"
                )

        # Check type
        layouts = [
            "auto",
            "vintage",
            "post_vintage",
            "post_vintage_ens",
        ]
        if self.type not in layouts:
            raise ValueError(
                f"invalid type '{self.type}'; choices: "
                + ", ".join(map("'{}'".format, layouts))
            )
