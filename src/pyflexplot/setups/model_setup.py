# pylint: disable=C0302  # too-many-lines (>1000)
"""Model setup."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

# First-party
from srutils.dataclasses import cast_field_value


# SR_TMP <<< TODO cleaner solution
def is_model_setup_param(param: str) -> bool:
    return param.replace("model.", "") in ModelSetup.get_params()


@dc.dataclass
class ModelSetup:
    name: str = "N/A"
    base_time: Optional[int] = None
    ens_member_id: Optional[Tuple[int, ...]] = None
    simulation_type: str = "N/A"

    def dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def tuple(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.dict().items())

    @classmethod
    def create(cls, params: Mapping[str, Any]) -> "ModelSetup":
        params = cls.cast_many(params)
        if "simulation_type" not in params:
            if params.get("ens_member_id"):
                params["simulation_type"] = "ensemble"
            else:
                params["simulation_type"] = "deterministic"
        return cls(**params)

    # SR_TMP Identical to CoreDimensions.cast
    @classmethod
    def cast(cls, param: str, value: Any) -> Any:
        return cast_field_value(
            cls,
            param,
            value,
            auto_wrap=True,
            bool_mode="intuitive",
            timedelta_unit="hours",
            unpack_str=False,
        )

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

    # SR_TMP Identical to CoreSetup.get_params and CoreDimensions.get_params
    @classmethod
    def get_params(cls) -> List[str]:
        return list(cls.__dataclass_fields__)  # type: ignore  # pylint: disable=E1101
