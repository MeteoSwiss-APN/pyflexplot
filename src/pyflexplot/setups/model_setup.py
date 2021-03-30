"""Model setup."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple

# Local
from .base_setup import BaseSetup


# SR_TMP <<< TODO cleaner solution
def is_model_setup_param(param: str) -> bool:
    return param.replace("model.", "") in ModelSetup.get_params()


# SR_TMP TODO pull common base class out of ModelSetup, LayoutSetup .
@dc.dataclass
class ModelSetup(BaseSetup):
    name: str = "N/A"
    base_time: Optional[int] = None
    ens_member_id: Optional[Tuple[int, ...]] = None
    simulation_type: str = "N/A"

    @classmethod
    def _create_mod_params_post_cast(cls, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Modify params in ``create`` after typecasting."""
        params = dict(params)
        if "simulation_type" not in params:
            if params.get("ens_member_id"):
                params["simulation_type"] = "ensemble"
            else:
                params["simulation_type"] = "deterministic"
        return params
