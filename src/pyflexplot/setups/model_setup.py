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
def is_model_setup_param(param: str, recursive: bool = False) -> bool:
    if recursive:
        raise NotImplementedError("recursive")
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
        members: Optional[list] = params.get("ens_member_id", [])
        multiple_members = members is not None and len(members) > 1
        if "simulation_type" not in params:
            if multiple_members:
                params["simulation_type"] = "ensemble"
            else:
                params["simulation_type"] = "deterministic"
        if params["simulation_type"] == "deterministic" and multiple_members:
            raise ValueError(
                f"deterministic simulation cannot have multiple members: {members}"
            )
        return params
